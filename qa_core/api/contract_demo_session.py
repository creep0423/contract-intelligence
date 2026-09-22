"""UAT / 本地演示专用的浏览器同源会话身份。

为什么需要这个模块：
- 生产与 UAT 的合同接口身份由可信上游网关注入 `x-contract-upstream-token` 和身份 Header；
  本项目的 Real UAT Compose 没有网关组件，浏览器既拿不到也不应该持有可信上游令牌；
- 为了让受保护的合同工作台可以在本地演示环境里正常打开，这里提供一条"服务端签发会话票据"
  的路径：页面响应下发签名票据，浏览器凭票据获得固定的演示身份。

安全边界（全部为 fail-closed）：
- 只有当 `APP_ENV` 属于 dev/test/uat 且显式开启 `CONTRACT_BROWSER_DEMO_SESSION` 时才会生效；
  生产环境即使误开该开关，也会在启动前置校验阶段直接失败，不会带病启动；
- 票据使用 HMAC-SHA256 签名，密钥复用服务端可信上游令牌（浏览器无法读取），
  并把签发时的环境与演示身份一起签名，因此配置变更后旧票据立即失效；
- 演示身份完全来自服务端配置，演示路径不读取任何客户端提交的身份 Header，
  浏览器无法借此切换到其他租户；
- 请求一旦携带 `x-contract-upstream-token`，只走可信上游校验路径，不存在身份降级。
"""

from __future__ import annotations

# base64: 票据负载编解码
import base64
# hashlib: HMAC-SHA256 摘要算法
import hashlib
# hmac: 常量时间签名比较，避免比较耗时泄露信息
import hmac
# json: 票据负载序列化
import json
# secrets: 票据随机数，保证每次签发的票据都不同
import secrets
# time: 票据过期时间计算
import time
from collections.abc import Mapping

from fastapi import Request, WebSocket
from pydantic import ValidationError

from qa_core.config.settings import DEMO_IDENTITY_ENVIRONMENTS, Settings, get_settings
from qa_core.contracts.schemas import ContractAccessContext


DEMO_SESSION_COOKIE = "contract_demo_session"
DEMO_SESSION_VERSION = "v1"
_IDENTITY_FIELDS = ("env", "tenant", "user", "visibility", "roles")


def demo_session_enabled(settings: Settings) -> bool:
    """判断当前配置是否允许浏览器演示会话。

    三个条件必须同时满足：显式开启开关、环境属于非生产环境、可信上游令牌已配置
    （票据签名需要密钥）。任一条件不满足都返回 False，调用方继续走可信上游校验。

    调用顺序：页面响应/合同身份依赖 -> demo_session_enabled()。
    """
    return (
        bool(settings.contract_browser_demo_session)
        and str(settings.env).lower() in DEMO_IDENTITY_ENVIRONMENTS
        and bool(_demo_secret(settings))
    )


def issue_demo_session(settings: Settings) -> str | None:
    """签发演示会话票据；未启用演示会话时返回 None（不写任何 Cookie）。

    调用顺序：页面路由 -> issue_demo_session()。
    """
    if not demo_session_enabled(settings):
        return None
    payload = _encode(
        json.dumps(
            {
                **_demo_identity(settings),
                "exp": int(time.time()) + int(settings.contract_demo_session_ttl_seconds),
                "nonce": secrets.token_urlsafe(12),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return f"{DEMO_SESSION_VERSION}.{payload}.{_signature(_demo_secret(settings), payload)}"


def demo_session_ticket_matches(settings: Settings, ticket: str | None) -> bool:
    """校验票据签名、签发上下文与过期时间；任一项不符都按未通过处理。

    调用顺序：resolve_demo_contract_access() -> demo_session_ticket_matches()。
    """
    if not demo_session_enabled(settings):
        return False
    parts = str(ticket or "").strip().split(".")
    if len(parts) != 3 or parts[0] != DEMO_SESSION_VERSION:
        return False
    _, payload, presented = parts
    if not hmac.compare_digest(presented, _signature(_demo_secret(settings), payload)):
        return False
    decoded = _decode(payload)
    if decoded is None:
        return False
    try:
        envelope = json.loads(decoded)
    except json.JSONDecodeError:
        return False
    if not isinstance(envelope, dict):
        return False
    identity = _demo_identity(settings)
    if any(envelope.get(field) != identity[field] for field in _IDENTITY_FIELDS):
        return False
    expires = envelope.get("exp")
    return isinstance(expires, int) and expires > int(time.time())


def resolve_demo_contract_access(
    scope: Request | WebSocket,
    *,
    settings: Settings | None = None,
) -> ContractAccessContext | None:
    """演示会话有效时返回固定的演示身份，否则返回 None。

    返回 None 表示"本请求不适用演示身份"，调用方必须继续执行可信上游校验，
    不能把 None 当作放行。

    调用顺序：合同身份依赖 -> resolve_demo_contract_access()。
    """
    resolved = settings or get_settings()
    if not demo_session_ticket_matches(resolved, _cookie_value(scope, DEMO_SESSION_COOKIE)):
        return None
    identity = _demo_identity(resolved)
    try:
        return ContractAccessContext(
            tenant_id=identity["tenant"],
            user_id=identity["user"],
            visibility=identity["visibility"],
            user_roles=identity["roles"] or ["public"],
        )
    except ValidationError:
        # 配置不合法时按未通过处理，避免把非法身份带入合同查询。
        return None


def _demo_secret(settings: Settings) -> str:
    """票据签名密钥：复用服务端可信上游令牌，永不发送到浏览器。"""
    return str(settings.contract_trusted_upstream_token).strip()


def _demo_identity(settings: Settings) -> dict[str, object]:
    """演示身份只来自服务端配置，不接受任何客户端输入。"""
    return {
        "env": str(settings.env).lower(),
        "tenant": str(settings.contract_demo_tenant_id).strip(),
        "user": str(settings.contract_demo_user_id).strip(),
        "visibility": str(settings.contract_demo_visibility).strip().lower(),
        "roles": [str(role).strip() for role in settings.contract_demo_user_roles if str(role).strip()],
    }


def _signature(secret: str, payload: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        f"{DEMO_SESSION_VERSION}.{payload}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _encode(raw: str) -> str:
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


def _decode(raw: str) -> str | None:
    try:
        return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def _cookie_value(scope: Request | WebSocket, name: str) -> str | None:
    cookies = getattr(scope, "cookies", None)
    if not isinstance(cookies, Mapping):
        return None
    value = cookies.get(name)
    return str(value) if value else None
