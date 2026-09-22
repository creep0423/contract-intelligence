"""FastAPI 共享依赖：管理令牌校验和轻量限流。

属于协议层横切能力，通过 FastAPI Depends 注入到路由函数中。
限流使用进程内滑动窗口算法，不依赖外部 Redis，适合单进程部署场景。
多进程部署时需切换到 Redis 中心化限流。

设计决策：
- 滑动窗口比固定窗口更平滑，不会在整点边界出现突发流量。
- 滑动窗口只在每次请求时惰性清理过期记录，省去定时器开销。
- 管理接口使用独立于用户 token 的专用令牌鉴权，避免被滥用破坏生产数据。
"""

from __future__ import annotations
import hmac
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from fastapi import Header, Request, WebSocket
from qa_core.api.contract_demo_session import resolve_demo_contract_access
from qa_core.contracts.schemas import ContractAccessContext
from qa_core.api.error_handlers import raise_server_error, raise_too_many_requests, raise_unauthorized
from qa_core.config.settings import TRUSTED_IDENTITY_ENVIRONMENTS, get_settings
settings = get_settings()
RATE_BUCKETS: dict[str, deque[float]] = defaultdict(deque)

def client_key(scope: Request | WebSocket) -> str:
    """从 HTTP/WebSocket 连接中提取客户端 IP 作为限流 key。（★★ 理解）

    优先使用 scope.client.host（FastAPI 自动从 ASGI scope 提取的客户端地址），
    仅在 local 开发或没有 client 信息时回退到 "local" 字符串。

    参数：
        scope: FastAPI Request 或 WebSocket 对象，包含底层 ASGI scope 信息。

    返回：
        客户端 IP 字符串或 "local"。

    调用顺序：FastAPI 路由层 -> client_key()。
    """
    # 原因：scope.client 可能为 None（如测试环境或某些 ASGI server），需要安全访问
    client = getattr(scope, "client", None)
    if client and getattr(client, "host", None):
        return str(client.host)
    return "local"

def check_rate_limit(key: str) -> bool:
    """执行进程内滑动窗口限流，避免频繁请求打爆 LLM/Milvus。（★★★ 核心）

    算法说明：
      1. 读取配置文件中的 api_rate_limit_per_minute（每分钟最大请求数）。
      2. 如果 limit <= 0，表示不限流，直接返回 True。
      3. 从 RATE_BUCKETS 字典中获取或创建该 key 对应的请求时间戳双端队列。
      4. 移除 60 秒之前的所有过期时间戳（惰性清理）。
      5. 如果队列长度已达上限，返回 False（限流）。
      6. 否则将当前时间戳加入队列并返回 True。

    参数：
        key: 限流标识（通常是客户端 IP）。

    返回：
        True 表示请求允许通过，False 表示触发限流。

    调用顺序：FastAPI 路由层 -> check_rate_limit()。
    """
    limit = max(int(settings.api_rate_limit_per_minute or 0), 0)
    # 原因：滑动窗口比固定窗口更平滑，不会在整点边界出现突发流量；
    # 滑动窗口只在每次请求时惰性清理过期记录，省去定时器开销
    if limit <= 0:
        return True
    # ── 步骤 1：获取当前时间 ──
    now = time.time()
    bucket = RATE_BUCKETS[key]
    # ── 步骤 2：清理 60 秒前的过期时间戳 ──
    while bucket and now - bucket[0] >= 60:
        bucket.popleft()
    # ── 步骤 3：判断是否超限 ──
    if len(bucket) >= limit:
        return False
    # ── 步骤 4：记录当前请求时间 ──
    bucket.append(now)
    return True

def enforce_http_rate_limit(request: Request) -> None:
    """HTTP 请求限流 FastAPI 依赖注入器，超限时返回 429。

    参数：
        request: FastAPI Request 对象，由依赖注入框架自动传入。

    异常：
        HTTPException(429): 超出限流阈值时抛出。

    调用顺序：FastAPI 路由层 -> enforce_http_rate_limit()。
    """
    # 原因：作为 FastAPI Depends 使用，路由层只需在参数中声明
    # `_: None = Depends(enforce_http_rate_limit)` 即可启用限流
    if not check_rate_limit(client_key(request)):
        raise_too_many_requests("请求过于频繁，请稍后再试。")

def require_admin_token(x_admin_token: str | None = Header(default=None)) -> None:
    """校验管理接口令牌，未配置或令牌不匹配时返回 500/401。（★★★ 核心）

    参数：
        x_admin_token: 从 HTTP Header `x-admin-token` 中提取的管理令牌。

    异常：
        HTTPException(500): ADMIN_API_TOKEN 环境变量未配置。
        HTTPException(401): 请求令牌与配置不匹配。

    调用顺序：FastAPI 路由层 -> require_admin_token()。
    """
    # 原因：管理接口（重索引、版本激活、数据清理）一旦被滥用可能破坏生产数据，
    # 需要独立于用户 token 的专用令牌鉴权
    expected = settings.admin_api_token.strip()
    if not expected:
        raise_server_error("ADMIN_API_TOKEN 未配置")
    if x_admin_token != expected:
        raise_unauthorized("管理接口令牌无效")


def resolve_contract_access(
    request: Request,
    x_tenant_id: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
    x_visibility: str | None = Header(default=None),
    x_user_roles: str | None = Header(default=None),
    x_contract_upstream_token: str | None = Header(default=None),
) -> ContractAccessContext:
    """由统一可信上游入口解析 HTTP 合同身份，body/query 永不参与权限决策。

    未携带可信上游令牌时，允许非生产环境的演示会话以服务端配置的固定演示身份访问
    （见 contract_demo_session）；一旦携带令牌，就只走可信上游校验，避免身份降级。

    调用顺序：FastAPI 路由层 -> resolve_contract_access()。
    """
    if not _presented_upstream_token(x_contract_upstream_token):
        demo_access = resolve_demo_contract_access(request)
        if demo_access is not None:
            return demo_access
    return _resolve_contract_access_headers(request.headers, x_contract_upstream_token)


def resolve_websocket_contract_access(websocket: WebSocket) -> ContractAccessContext:
    """WebSocket 与 HTTP 使用完全相同的可信身份解析和校验。"""
    origin = str(websocket.headers.get("origin") or "").rstrip("/")
    allowed_origins = {str(item).rstrip("/") for item in settings.cors_allow_origins}
    if origin and "*" not in allowed_origins and origin not in allowed_origins:
        raise_unauthorized("WebSocket Origin 不受信任")
    presented_token = websocket.headers.get("x-contract-upstream-token")
    if not _presented_upstream_token(presented_token):
        demo_access = resolve_demo_contract_access(websocket)
        if demo_access is not None:
            return demo_access
    return _resolve_contract_access_headers(websocket.headers, presented_token)


def _presented_upstream_token(presented_token: str | None) -> bool:
    """判断请求是否显式携带了可信上游令牌；携带即不允许回退到演示会话。"""
    return bool(str(presented_token or "").strip())


def _resolve_contract_access_headers(
    headers: Mapping[str, str],
    presented_token: str | None,
) -> ContractAccessContext:
    expected = settings.contract_trusted_upstream_token.strip()
    if expected and not hmac.compare_digest(str(presented_token or ""), expected):
        raise_unauthorized("合同身份上游校验失败")
    if settings.env.lower() in TRUSTED_IDENTITY_ENVIRONMENTS and not expected:
        raise_server_error("生产环境未配置合同可信上游令牌")
    roles = [item.strip() for item in headers.get("x-user-roles", "public").split(",") if item.strip()]
    return ContractAccessContext(
        tenant_id=headers.get("x-tenant-id", "default"),
        user_id=headers.get("x-user-id", "anonymous"),
        visibility=headers.get("x-visibility", "private"),
        user_roles=roles or ["public"],
    )
