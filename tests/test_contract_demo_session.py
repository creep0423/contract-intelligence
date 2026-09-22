"""浏览器演示会话身份的边界测试。

覆盖三类风险：
1. 演示会话默认关闭，未开启时合同接口必须继续拒绝无身份请求；
2. 票据必须防伪造、防篡改、带过期时间，并与签发环境/演示身份绑定；
3. 演示路径不得读取客户端身份 Header，也不得在携带可信上游令牌时降级。
"""

from __future__ import annotations

import base64
import json
import time

import pytest
from fastapi import Request
from starlette.websockets import WebSocket

from qa_core.api import contract_demo_session
from qa_core.api import dependencies as api_dependencies
from qa_core.api import pages
from qa_core.config.preflight import require_demo_session_environment
from qa_core.config.settings import Settings


_ENV_ALIASES = {
    "env": "APP_ENV",
    "contract_trusted_upstream_token": "CONTRACT_TRUSTED_UPSTREAM_TOKEN",
    "contract_browser_demo_session": "CONTRACT_BROWSER_DEMO_SESSION",
    "contract_demo_tenant_id": "CONTRACT_DEMO_TENANT_ID",
    "contract_demo_user_id": "CONTRACT_DEMO_USER_ID",
    "contract_demo_visibility": "CONTRACT_DEMO_VISIBILITY",
    "contract_demo_user_roles": "CONTRACT_DEMO_USER_ROLES",
    "contract_demo_session_ttl_seconds": "CONTRACT_DEMO_SESSION_TTL_SECONDS",
}


def _settings(**overrides) -> Settings:
    """按环境变量别名构造设置对象，避免依赖本机 .env 内容。"""
    values = {
        "env": "uat",
        "contract_trusted_upstream_token": "uat-secret",
        "contract_browser_demo_session": True,
        "contract_demo_tenant_id": "demo-tenant",
        "contract_demo_user_id": "demo-user",
        "contract_demo_visibility": "private",
        "contract_demo_user_roles": ["legal", "public"],
        "contract_demo_session_ttl_seconds": 3600,
    }
    values.update(overrides)
    return Settings(_env_file=None, **{_ENV_ALIASES[field]: value for field, value in values.items()})


async def _noop_receive() -> dict:
    return {"type": "websocket.disconnect", "code": 1000}


async def _noop_send(_: dict) -> None:
    return None


def _request(*, cookie: str | None = None, headers: dict[str, str] | None = None) -> Request:
    raw_headers = [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in (headers or {}).items()]
    if cookie:
        raw_headers.append((b"cookie", f"{contract_demo_session.DEMO_SESSION_COOKIE}={cookie}".encode("latin-1")))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/contracts",
            "query_string": b"",
            "headers": raw_headers,
        }
    )


def _websocket(*, cookie: str | None = None, headers: dict[str, str] | None = None) -> WebSocket:
    raw_headers = [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in (headers or {}).items()]
    if cookie:
        raw_headers.append((b"cookie", f"{contract_demo_session.DEMO_SESSION_COOKIE}={cookie}".encode("latin-1")))
    return WebSocket(
        {
            "type": "websocket",
            "path": "/api/contracts/example/stream",
            "query_string": b"",
            "headers": raw_headers,
        },
        receive=_noop_receive,
        send=_noop_send,
    )


def _unsigned_ticket(settings: Settings, *, expires_in: int) -> str:
    """用模块内部编码方式构造一张过期/伪造票据，用于验证签名与过期检查。"""
    payload = contract_demo_session._encode(
        json.dumps(
            {
                **contract_demo_session._demo_identity(settings),
                "exp": int(time.time()) + expires_in,
                "nonce": "test",
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return f"{contract_demo_session.DEMO_SESSION_VERSION}.{payload}.{contract_demo_session._signature('uat-secret', payload)}"


def test_demo_session_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("CONTRACT_BROWSER_DEMO_SESSION", raising=False)
    settings = Settings(_env_file=None)

    assert settings.contract_browser_demo_session is False
    assert contract_demo_session.demo_session_enabled(settings) is False
    assert contract_demo_session.issue_demo_session(settings) is None
    assert contract_demo_session.resolve_demo_contract_access(_request(cookie="anything"), settings=settings) is None


def test_demo_session_requires_non_production_environment() -> None:
    for environment in ("production", "prod", "prd", "staging"):
        settings = _settings(env=environment)
        assert contract_demo_session.demo_session_enabled(settings) is False
        assert contract_demo_session.issue_demo_session(settings) is None


def test_preflight_rejects_demo_session_outside_non_production_environments() -> None:
    with pytest.raises(RuntimeError):
        require_demo_session_environment(_settings(env="production"))
    for environment in ("dev", "test", "uat"):
        require_demo_session_environment(_settings(env=environment))


def test_demo_session_requires_trusted_upstream_token() -> None:
    settings = _settings(contract_trusted_upstream_token="")

    assert contract_demo_session.demo_session_enabled(settings) is False
    assert contract_demo_session.issue_demo_session(settings) is None
    assert contract_demo_session.resolve_demo_contract_access(_request(cookie="v1.x.y"), settings=settings) is None


def test_demo_ticket_signature_and_expiry_are_enforced() -> None:
    settings = _settings()
    ticket = contract_demo_session.issue_demo_session(settings)
    assert ticket is not None
    assert contract_demo_session.demo_session_ticket_matches(settings, ticket) is True

    version, payload, signature = ticket.split(".")
    assert contract_demo_session.demo_session_ticket_matches(settings, f"{version}.{payload}.{signature[:-1]}0") is False
    assert contract_demo_session.demo_session_ticket_matches(settings, f"{version}.{payload}x.{signature}") is False
    assert contract_demo_session.demo_session_ticket_matches(settings, "v9.abc.def") is False
    assert contract_demo_session.demo_session_ticket_matches(settings, None) is False

    forged = base64.urlsafe_b64encode(json.dumps({"tenant": "other-tenant"}).encode("utf-8")).decode("ascii").rstrip("=")
    assert contract_demo_session.demo_session_ticket_matches(settings, f"{version}.{forged}.{signature}") is False


def test_demo_ticket_expires() -> None:
    settings = _settings()
    expired = _unsigned_ticket(settings, expires_in=-1)
    assert contract_demo_session.demo_session_ticket_matches(settings, expired) is False


def test_demo_ticket_is_bound_to_configured_identity_and_environment() -> None:
    ticket = contract_demo_session.issue_demo_session(_settings())

    assert contract_demo_session.demo_session_ticket_matches(_settings(contract_demo_tenant_id="other-tenant"), ticket) is False
    assert contract_demo_session.demo_session_ticket_matches(_settings(contract_demo_user_id="other-user"), ticket) is False
    assert contract_demo_session.demo_session_ticket_matches(_settings(contract_demo_visibility="internal"), ticket) is False
    assert contract_demo_session.demo_session_ticket_matches(_settings(contract_demo_user_roles=["legal"]), ticket) is False
    assert contract_demo_session.demo_session_ticket_matches(_settings(contract_trusted_upstream_token="rotated-secret"), ticket) is False


def test_demo_session_uses_configured_identity_and_ignores_client_headers() -> None:
    settings = _settings()
    ticket = contract_demo_session.issue_demo_session(settings)
    access = contract_demo_session.resolve_demo_contract_access(
        _request(
            cookie=ticket,
            headers={
                "x-tenant-id": "attacker-tenant",
                "x-user-id": "attacker-user",
                "x-visibility": "public",
                "x-user-roles": "admin",
            },
        ),
        settings=settings,
    )

    assert access is not None
    assert access.tenant_id == "demo-tenant"
    assert access.user_id == "demo-user"
    assert access.visibility == "private"
    assert access.user_roles == ["legal", "public"]


def test_demo_session_resolves_for_websocket_with_same_rules(monkeypatch) -> None:
    settings = _settings()
    monkeypatch.setattr(api_dependencies, "settings", settings)
    monkeypatch.setattr(contract_demo_session, "get_settings", lambda: settings)
    ticket = contract_demo_session.issue_demo_session(settings)

    access = api_dependencies.resolve_websocket_contract_access(
        _websocket(cookie=ticket, headers={"origin": "http://127.0.0.1:28100"})
    )
    assert access.tenant_id == "demo-tenant"

    with pytest.raises(Exception) as rejected:
        api_dependencies.resolve_websocket_contract_access(
            _websocket(cookie=ticket, headers={"origin": "http://evil.example"})
        )
    assert getattr(rejected.value, "status_code", None) == 401


def test_presented_upstream_token_disables_demo_fallback(monkeypatch) -> None:
    settings = _settings()
    monkeypatch.setattr(api_dependencies, "settings", settings)
    monkeypatch.setattr(contract_demo_session, "get_settings", lambda: settings)
    ticket = contract_demo_session.issue_demo_session(settings)

    with pytest.raises(Exception) as rejected:
        api_dependencies.resolve_contract_access(
            _request(cookie=ticket, headers={"x-contract-upstream-token": "wrong-token"}),
            x_contract_upstream_token="wrong-token",
        )
    assert getattr(rejected.value, "status_code", None) == 401


def test_protected_path_rejects_missing_identity_when_demo_session_disabled(monkeypatch) -> None:
    settings = _settings(contract_browser_demo_session=False)
    monkeypatch.setattr(api_dependencies, "settings", settings)
    monkeypatch.setattr(contract_demo_session, "get_settings", lambda: settings)

    with pytest.raises(Exception) as rejected:
        api_dependencies.resolve_contract_access(_request(cookie="v1.forged.signature"), x_contract_upstream_token=None)
    assert getattr(rejected.value, "status_code", None) == 401


def test_demo_page_sets_httponly_cookie_only_when_enabled(monkeypatch) -> None:
    enabled = _settings()
    monkeypatch.setattr(pages, "get_settings", lambda: enabled)
    response = pages._demo_aware_page("static/contracts.html", _request())
    header = response.headers.get("set-cookie", "")

    assert contract_demo_session.DEMO_SESSION_COOKIE in header
    assert "HttpOnly" in header
    assert "SameSite=lax" in header
    assert "Path=/" in header
    assert "uat-secret" not in response.body.decode("utf-8")
    assert "uat-secret" not in header

    monkeypatch.setattr(pages, "get_settings", lambda: _settings(env="production"))
    assert pages._demo_aware_page("static/contracts.html", _request()).headers.get("set-cookie") is None
