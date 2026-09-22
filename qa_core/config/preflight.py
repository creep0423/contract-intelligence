"""主链路运行环境前置校验：Milvus、MySQL、本地模型、LLM Key 等基础条件不满足直接报错。

设计决策（fail-fast）：
- 在服务接收任何用户请求之前，验证外部依赖和配置项。这里的核心权衡是启动时多花
  几百毫秒做全面检查，换取"线上零配置事故"的保障——如果等到第一个用户请求才暴露
  Milvus 连接失败或模型路径错误，故障影响范围会从"启动失败"扩散到"服务降级、
  隐式报错、数据不一致"。
- 检查顺序按成本排列：纯内存操作（占位符检测）→ 文件系统调用（路径检查）→
  网络 I/O（TCP 连接检查），让最快速的检查先阻断，网络层面的问题最后暴露。

调用顺序：进程启动时 -> preflight。
"""

from __future__ import annotations

import socket
import importlib
from pathlib import Path
from urllib.parse import urlparse

from qa_core.config.settings import DEMO_IDENTITY_ENVIRONMENTS, TRUSTED_IDENTITY_ENVIRONMENTS, get_settings
from qa_core.governance.kb_versions import get_kb_version_store
from qa_core.scenarios.registry import ACTIVE_SCENARIO_IDS, get_scenario_registry, resolve_scenario


PLACEHOLDER_VALUES = {"", "replace-with-real-key", "replace-with-random-token", "changeme", "change-me"}
PLACEHOLDER_MARKERS = ("请替换", "replace", "changeme", "change-me", "your-", "placeholder")

def _is_placeholder(value: str | None) -> bool:
    """判断配置值是否为空或仍是示例占位符。

    统一转小写后依次与精确占位符值和模糊占位符标记匹配，覆盖"replace-with-real-key"
    等常见开发疏忽。

    参数：
        value: 待检查的配置值。

    返回：
        True 表示值是占位符或空值，需要阻断启动。

    调用顺序：启动配置或前置校验 -> _is_placeholder()。
    """
    normalized = str(value or "").strip()
    # 统一转小写后依次与精确占位符值和模糊占位符标记匹配，覆盖"replace-with-real-key"等常见开发疏忽
    lower_value = normalized.lower()
    return lower_value in PLACEHOLDER_VALUES or any(marker in lower_value for marker in PLACEHOLDER_MARKERS)


def require_demo_session_environment(settings) -> None:
    """演示会话环境守卫：只有非生产环境允许开启浏览器演示会话。

    该检查刻意放在前置校验里而不是请求处理里：生产配置一旦误开
    CONTRACT_BROWSER_DEMO_SESSION，服务直接拒绝启动，不会带病收发流量。

    调用顺序：validate_runtime_environment() -> require_demo_session_environment()。
    """
    if settings.contract_browser_demo_session and str(settings.env).lower() not in DEMO_IDENTITY_ENVIRONMENTS:
        raise RuntimeError(
            "CONTRACT_BROWSER_DEMO_SESSION 只允许在 dev/test/uat 环境启用；生产环境必须保持关闭。"
        )


def _require_tcp(name: str, host: str, port: int, timeout: float = 3.0) -> None:
    """校验 TCP 端口可连接。

    这里只做连接性检查，不做业务读写。真实集合、表结构和模型预热会在后续 warmup 中
    完成。把端口检查放在这里，是为了让"服务没启动"这类基础问题在最早阶段暴露。

    参数：
        name: 服务名称，用于错误信息提示。
        host: 主机地址。
        port: 端口号。
        timeout: 连接超时时间（秒），默认 3 秒。

    调用顺序：启动配置或前置校验 -> _require_tcp()。
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return  # TCP 三次握手成功即通过，不做业务探活避免引入外部依赖初始化耗时
    except OSError as exc:
        # 关键依赖缺失时直接阻断启动，避免服务看似正常但核心链路不通
        raise RuntimeError(f"{name} 不可连接：{host}:{port}。请先启动必需环境。") from exc


def _require_path(name: str, raw_path: str) -> None:
    """校验本地目录或文件存在。

    参数：
        name: 路径名称，用于错误信息提示。
        raw_path: 待检查的路径字符串。

    调用顺序：启动配置或前置校验 -> _require_path()。
    """
    path = Path(raw_path)
    if not path.exists():
        raise RuntimeError(f"{name} 不存在：{path}")


def _require_milvus_uri() -> None:
    """校验 Milvus URI 格式和 TCP 可达性。

    从 URL 中解析 host 和 port，URI 缺少 hostname 时说明配置格式错误
    （如漏写协议前缀），直接阻断避免连接超时浪费启动时间。

    调用顺序：validate_runtime_environment() -> _require_milvus_uri()。
    """
    settings = get_settings()
    # URI 格式示例：http://localhost:19530，从中提取 host 和 port 用于 TCP 探活
    parsed = urlparse(settings.milvus_uri)
    host = parsed.hostname
    port = parsed.port or 19530
    if not host:
        # URI 缺少 hostname 时说明配置格式错误（如漏写协议前缀），直接阻断避免连接超时浪费启动时间
        raise RuntimeError(f"MILVUS_URI 无效：{settings.milvus_uri}")
    _require_tcp("Milvus", host, port)


def _require_redis() -> None:
    """校验 Redis 连接和 Python 客户端依赖。

    redis Python 包是可选依赖（不在 requirements.txt 中强制安装），
    仅当 cache_enabled=True 时才需要。缺少时给出明确的安装提示。

    调用顺序：validate_runtime_environment() -> _require_redis()。
    """
    settings = get_settings()
    try:
        importlib.import_module("redis")
    except ImportError as exc:
        raise RuntimeError("CACHE_ENABLED=true 时必须安装 redis Python 依赖。") from exc
    _require_tcp("Redis", settings.redis_host, settings.redis_port)


def validate_runtime_environment() -> dict[str, object]:
    """校验主链路基础前置条件，任一不满足则抛出 RuntimeError。

    【fail-fast 设计】
    在服务接收任何用户请求之前，验证外部依赖和配置项。这里的核心权衡是：
    启动时多花几百毫秒做全面检查，换取"线上零配置事故"的保障——如果等到第一个用户
    请求才暴露 Milvus 连接失败或模型路径错误，故障影响范围会从"启动失败"扩散到
    "服务降级、隐式报错、数据不一致"。

    【检查顺序说明】
    1. 占位符检测（API Key / Token）——纯内存操作，零成本，最先拦截最常见的人为错误。
    2. 本地路径检查（模型目录、场景目录、FAQ 文件）——文件系统调用，比网络 I/O 快一到
       两个数量级，优先暴露开发环境的常见配置遗漏。
    3. TCP 连接检查（Milvus、MySQL、Redis）——网络 I/O 最慢且可能 hang，放在最后，
       让前面快速失败的检查先阻断，网络层面的问题留到最后集中暴露。

    active KB version 属于 schema bootstrap 之后的业务状态检查，由
    validate_active_kb_versions() 单独完成，避免前置校验阶段夹带表结构假设。

    LLM 真实连通性不放在这里阻断启动：Key 必须配置，但供应商欠费、额度耗尽或临时网络
    抖动会被写入运行状态，由 /health 和 /api/admin/status 暴露；需要生成答案时仍会正常
    调用 LLM，失败时由业务错误处理返回。

    调用顺序：启动配置或前置校验 -> validate_runtime_environment()。
    """
    # 获取场景注册器并解析当前场景的 TOML 配置
    settings = get_settings()
    registry = get_scenario_registry()
    if settings.active_scenario_id not in ACTIVE_SCENARIO_IDS:
        raise RuntimeError("ACTIVE_SCENARIO_ID 必须是 tender_contract_risk；当前产品仅启用合同履约风控场景。")
    scenario = resolve_scenario()

    # 阶段1：占位符检测（纯内存操作，最先拦截最常见的人为配置遗漏）
    if _is_placeholder(settings.llm_api_key):
        # llm_api_key 缺失时直接阻断，避免 LLM 调用环节报"鉴权失败"混淆问题根因
        raise RuntimeError("DASHSCOPE_API_KEY 未配置。当前架构必须通过 LangChain ChatOpenAI 调用真实 LLM。")
    if _is_placeholder(settings.admin_api_token):
        # admin token 缺失时阻断，管理接口需要显式令牌防止未授权访问
        raise RuntimeError("ADMIN_API_TOKEN 未配置。管理接口必须显式设置令牌。")
    if settings.env.lower() in TRUSTED_IDENTITY_ENVIRONMENTS and _is_placeholder(settings.contract_trusted_upstream_token):
        raise RuntimeError(
            "CONTRACT_TRUSTED_UPSTREAM_TOKEN 未配置。生产环境必须由可信网关注入合同身份上下文。"
        )
    # 演示会话只在非生产环境可用：生产配置误开时直接拒绝启动，避免出现可在生产生效的身份旁路
    require_demo_session_environment(settings)
    if settings.env.lower() in TRUSTED_IDENTITY_ENVIRONMENTS and "*" in settings.cors_allow_origins:
        raise RuntimeError("生产环境 CORS_ALLOW_ORIGINS 不得包含通配符 *")

    # 阶段2：本地路径检查（文件系统调用比网络 I/O 快一到两个数量级，优先暴露开发环境配置遗漏）
    _require_path("Embedding 模型目录", settings.embedding_model_path)
    _require_path("Reranker 模型目录", settings.reranker_model_path)
    _require_path("BERT 意图模型目录", settings.intent_model_path)
    _require_path("BERT 意图模型标签文件", str(Path(settings.intent_model_path) / "intent_labels.json"))
    if not (Path(settings.intent_model_path) / "model.safetensors").exists() and not (
        Path(settings.intent_model_path) / "pytorch_model.bin"
    ).exists():
        raise RuntimeError(f"BERT 意图模型权重文件不存在：{settings.intent_model_path}")

    if not Path(settings.scenario_config_dir).exists():
        raise RuntimeError(f"SCENARIO_CONFIG_DIR 不存在：{settings.scenario_config_dir}")
    if scenario.scenario_id not in {item.scenario_id for item in registry.list_runtime_scenarios()}:
        # 场景 ID 无效说明 TOML 配置与注册器不匹配，避免运行期产生"未知场景"报错
        raise RuntimeError(f"ACTIVE_SCENARIO_ID 无效：{settings.active_scenario_id}")
    _require_path("场景文档目录", scenario.data_root)
    _require_path("场景 FAQ 文件", scenario.faq_csv_path)

    # 阶段3：网络依赖连通性检查（网络 I/O 最慢，放在最后让前面快速失败的检查先阻断）
    _require_milvus_uri()
    _require_tcp("MySQL", settings.mysql_host, settings.mysql_port)
    if settings.cache_enabled:
        _require_redis()

    return {
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.display_name,
        "llm_model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
        "milvus_uri": settings.milvus_uri,
        "mysql": f"{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}",
        "redis": f"{settings.redis_host}:{settings.redis_port}/{settings.redis_db}" if settings.cache_enabled else "disabled",
        "embedding_model_path": settings.embedding_model_path,
        "reranker_model_path": settings.reranker_model_path,
        "intent_model_path": settings.intent_model_path,
        "intent_model_version": settings.intent_model_version,
        "available_scenarios": [item.scenario_id for item in registry.list_runtime_scenarios()],
    }


def validate_active_kb_versions(scenario_id: str | None = None) -> dict[str, object]:
    """校验当前场景存在可用 active 知识库版本。

    调用顺序：启动配置或前置校验 -> validate_active_kb_versions()。
    """
    scenario = resolve_scenario(scenario_id)
    version_store = get_kb_version_store(scenario.scenario_id)
    try:
        # 从版本清单表查询标记为 active 的版本号，不存在时上层会抛出 ValueError
        active_version = version_store.resolve_active_version()
    except ValueError as exc:
        # 无 active 版本意味着入库流程未执行或未激活，直接阻断以避免空知识库检索
        raise RuntimeError(
            f"{exc}。请先执行入库并激活版本，例如 "
            "scripts/rebuild_kb_version.py --new-version --force --quality-gate --activate。"
        ) from exc
    return{
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.display_name,
        "active_kb_version": active_version,
    }
