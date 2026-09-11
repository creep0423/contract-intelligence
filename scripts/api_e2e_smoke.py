"""真实 API E2E 验收脚本。

`acceptance_smoke.py` 重点验证页面、管理摘要和 WebSocket 流式事件。这个脚本更偏 API
合同验收：逐个检查当前一期暴露的 HTTP 管理、质量报告和 LangSmith 状态接口能否稳定返回预期字段。
LangSmith 未启用不代表本地服务不可用；本脚本只要求状态接口结构正确，是否启用写入
details，供正式企业环境另行确认。
"""

from __future__ import annotations

# ── 标准库 ──
# argparse: 命令行参数解析（--base-url, --admin-token, --scenario 等）
import argparse
# json: 标准 JSON 序列化
import json
# sys: 系统功能（sys.path 修改 + sys.exit 退出码）
import sys
# pathlib.Path: 文件路径操作
from pathlib import Path
# urllib: HTTP 请求（用于访问 REST API 接口）
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

# ── 项目根路径 ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── 导入核心模块 ──
# get_settings: 读取当前运行配置（含 ADMIN_API_TOKEN）
from qa_core.config.settings import get_settings
# configure_utf8_stdio: Windows UTF-8 输出保护
# write_optional_json: 可选 JSON 报告写入
from scripts.common import configure_utf8_stdio, write_optional_json


def headers(token: str) -> dict[str, str]:
    """构造管理接口 header。

    调用顺序：命令行入口 -> headers()。
    """
    return {"X-Admin-Token": token} if token else {}


def resolve_admin_token(raw_token: str | None) -> str:
    """解析管理接口令牌。

    使用场景：
    - 本地手动烟测时，通常只传 `--base-url`，不希望每次都把管理令牌写在命令行里；
    - 质量检查脚本会统一调用本脚本，如果命令行没有显式传令牌，就应该复用当前运行配置中的
      `ADMIN_API_TOKEN`；
    - CI 或临时环境仍可以通过 `--admin-token` 覆盖，便于验证不同服务实例。

    这里不会把令牌写入报告，避免验收文件泄露敏感信息。

    调用顺序：命令行入口 -> resolve_admin_token()。
    """
    return (raw_token or get_settings().admin_api_token or "").strip()


def fetch_json(base_url: str, path: str, token: str = "") -> dict:
    """读取 JSON 接口并返回 dict。

    调用顺序：命令行入口 -> fetch_json()。
    """
    req = urlrequest.Request(base_url.rstrip("/") + path, headers=headers(token))
    try:
        with urlrequest.urlopen(req, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"API check failed: {path}: {exc}") from exc


def check_fields(payload: dict, fields: list[str]) -> bool:
    """检查响应是否包含必需字段。

    调用顺序：命令行入口 -> check_fields()。
    """
    return all(field in payload for field in fields)


def run_checks(args: argparse.Namespace) -> dict:
    """执行 API 合同验收。

    逐接口检查当前一期暴露的 HTTP 管理、质量报告和 LangSmith 状态接口。
    每个接口验证两个维度：
      1. HTTP 响应是否可正常解析（fetch_json 失败会直接抛异常）
      2. 响应 JSON 中是否包含必需字段（check_fields）

    调用顺序：命令行入口 -> run_checks()。
    """
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    admin_token = resolve_admin_token(args.admin_token)

    # ── 基础接口（无需认证）──
    # 1. 健康检查：服务是否存活 + 当前 active 场景
    health = fetch_json(args.base_url, "/health")
    checks["health"] = check_fields(health, ["status", "active_scenario_id", "llm"])
    details["health"] = health

    # 2. 场景列表：是否能返回已注册场景配置
    scenarios = fetch_json(args.base_url, "/api/scenarios")
    checks["scenarios"] = bool(scenarios.get("scenarios"))
    details["scenario_count"] = len(scenarios.get("scenarios") or [])

    # ── 管理接口（需 X-Admin-Token）──
    # 3. 知识库版本列表：版本切面和元数据存储状态
    versions = fetch_json(args.base_url, f"/api/kb_versions?scenario_id={args.scenario}", admin_token)
    checks["kb_versions"] = check_fields(versions, ["scenario_id", "versions", "metadata_store"])
    details["kb_active"] = versions.get("effective_active_version")

    # 4. 管理状态摘要：场景、active 版本、缓存、意图模型、LLM 和 LangSmith 状态的汇总
    admin_status = fetch_json(args.base_url, "/api/admin/status", admin_token)
    checks["admin_status"] = check_fields(
        admin_status,
        ["status", "scenarios", "active_kb_versions", "cache", "intent_model", "llm", "langsmith"],
    )
    details["admin_scenario_count"] = len(admin_status.get("scenarios") or [])

    # 5. LLM 状态：模型、Base URL、最近探测和可用性是否可见
    llm_status = fetch_json(args.base_url, "/api/admin/llm", admin_token)
    checks["llm_status"] = check_fields(llm_status, ["status", "ok", "model", "base_url", "has_api_key"])
    details["llm_status"] = llm_status.get("status")
    details["llm_model"] = llm_status.get("model")

    # 6. 缓存状态：Redis、TTL、命中统计和 namespace epoch 是否可见
    cache_status = fetch_json(args.base_url, f"/api/admin/cache/status?scenario_id={args.scenario}", admin_token)
    checks["cache_status"] = check_fields(cache_status, ["enabled", "redis", "config", "stats", "namespaces"])
    details["cache_enabled"] = cache_status.get("enabled")
    details["cache_redis_ok"] = (cache_status.get("redis") or {}).get("ok")

    # 7. 意图模型状态：模型版本、标签、评测报告和治理闭环是否可见
    intent_model = fetch_json(args.base_url, "/api/admin/intent_model", admin_token)
    intent_payload = intent_model.get("payload") or {}
    intent_model_payload = intent_payload.get("model") or {}
    checks["intent_model"] = check_fields(intent_model, ["available", "payload"]) and check_fields(
        intent_payload,
        ["ok", "artifact_ok", "runtime_ok", "model", "decision_policy", "closure"],
    )
    details["intent_model_ok"] = intent_payload.get("ok")
    details["intent_model_version"] = intent_model_payload.get("model_version")

    # 8. LangSmith 状态：可观测性配置是否可见（启用与否不影响服务可用性）
    langsmith = fetch_json(args.base_url, "/api/admin/langsmith", admin_token)
    checks["langsmith"] = check_fields(langsmith, ["enabled", "project", "endpoint", "project_url"])
    details["langsmith_enabled"] = langsmith.get("enabled")
    details["langsmith_has_api_key"] = langsmith.get("has_api_key")
    details["langsmith_project"] = langsmith.get("project")

    # 9. 入库质量报告：最近 5 条入库报告是否存在
    ingestion = fetch_json(args.base_url, f"/api/admin/ingestion_reports?scenario_id={args.scenario}&limit=5", admin_token)
    checks["ingestion_reports"] = "reports" in ingestion
    details["ingestion_report_count"] = len(ingestion.get("reports") or [])

    # 10. 门禁报告：最近 5 条门禁判定记录
    gates = fetch_json(args.base_url, "/api/admin/gate_reports?limit=5", admin_token)
    checks["gate_reports"] = check_fields(gates, ["reports", "langsmith"])
    details["gate_report_count"] = len(gates.get("reports") or [])

    # 11. 性能报告：最近 5 条性能基线记录
    performance = fetch_json(args.base_url, "/api/admin/performance_reports?limit=5", admin_token)
    checks["performance_reports"] = check_fields(performance, ["reports", "langsmith"])
    details["performance_report_count"] = len(performance.get("reports") or [])

    # 12. 企业资料治理报告：真实度 + dirty samples + overlay 就绪
    enterprise_governance = fetch_json(args.base_url, "/api/admin/enterprise_governance", admin_token)
    checks["enterprise_governance"] = check_fields(enterprise_governance, ["report_type", "data_realism", "overlay_readiness", "langsmith"])

    # 13. 知识库版本对比报告：新旧版本召回对比结果
    kb_version_compare = fetch_json(args.base_url, "/api/admin/kb_version_compare", admin_token)
    checks["kb_version_compare"] = check_fields(kb_version_compare, ["report_type", "comparison", "langsmith"])

    return {
        "report_type": "api_e2e_smoke",
        "ok": all(checks.values()),
        "base_url": args.base_url,
        "checks": checks,
        "details": details,
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    调用顺序：命令行入口 -> build_parser()。
    """
    parser = argparse.ArgumentParser(description="Run HTTP API E2E checks.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--admin-token", default="", help="为空时自动读取当前运行配置中的 ADMIN_API_TOKEN。")
    parser.add_argument("--scenario", default="enterprise_knowledge")
    parser.add_argument("--output", default="", help="可选 JSON 报告输出路径。")
    return parser


def main() -> None:
    """执行 API E2E 检查。

    调用顺序：命令行入口 -> main()。
    """
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args()
    payload = run_checks(args)
    write_optional_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
