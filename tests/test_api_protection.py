"""API 基础保护能力测试。

当前项目要求管理令牌是必需环境配置，不再提供“本地为空则放行”的降级路径。这里直接
测试 `qa_core.api.dependencies` 的依赖函数，避免启动真实服务。
"""

from __future__ import annotations

import unittest

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from qa_core.api import dependencies as api_deps
from qa_core.api.error_handlers import register_api_exception_handlers
from qa_core.api.error_handlers import raise_bad_request, raise_not_found, raise_too_many_requests


class ApiProtectionTests(unittest.TestCase):
    """验证管理令牌和限流行为可控。

    调用顺序：pytest/unittest 测试入口 -> ApiProtectionTests。
    """

    def test_admin_token_requires_configured_token(self) -> None:
        """验证 admin_token 为空时 require_admin_token 抛出 500。

        调用顺序：pytest/unittest 测试入口 -> ApiProtectionTests.test_admin_token_requires_configured_token()。
        """
        original = api_deps.settings.admin_api_token
        api_deps.settings.admin_api_token = ""
        try:
            with self.assertRaises(HTTPException) as ctx:
                api_deps.require_admin_token(None)
            self.assertEqual(ctx.exception.status_code, 500)
        finally:
            api_deps.settings.admin_api_token = original

    def test_admin_token_rejects_wrong_token_when_enabled(self) -> None:
        """验证配置了 token 后，错误 token 导致 401。

        调用顺序：pytest/unittest 测试入口 -> ApiProtectionTests.test_admin_token_rejects_wrong_token_when_enabled()。
        """
        original = api_deps.settings.admin_api_token
        api_deps.settings.admin_api_token = "secret"
        try:
            with self.assertRaises(HTTPException) as ctx:
                api_deps.require_admin_token("bad")
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertIsNone(api_deps.require_admin_token("secret"))
        finally:
            api_deps.settings.admin_api_token = original

    def test_rate_limit_can_block_after_limit(self) -> None:
        """验证超过限流阈值后请求被拒绝。

        调用顺序：pytest/unittest 测试入口 -> ApiProtectionTests.test_rate_limit_can_block_after_limit()。
        """
        original_limit = api_deps.settings.api_rate_limit_per_minute
        api_deps.settings.api_rate_limit_per_minute = 2
        api_deps.RATE_BUCKETS.clear()
        try:
            self.assertTrue(api_deps.check_rate_limit("unit-test"))
            self.assertTrue(api_deps.check_rate_limit("unit-test"))
            self.assertFalse(api_deps.check_rate_limit("unit-test"))
        finally:
            api_deps.settings.api_rate_limit_per_minute = original_limit
            api_deps.RATE_BUCKETS.clear()

    def test_value_error_becomes_http_400(self) -> None:
        """验证 ValueError 被转换为 HTTP 400 响应。

        调用顺序：pytest/unittest 测试入口 -> ApiProtectionTests.test_value_error_becomes_http_400()。
        """
        app = FastAPI()
        register_api_exception_handlers(app)

        @app.get("/value-error")
        def value_error_route():
            """测试辅助路由：始终抛出 ValueError，用于验证 HTTP 400 转换。

            调用顺序：pytest/unittest 测试入口 -> ApiProtectionTests.value_error_route()。
            """
            raise ValueError("业务分类不合法")

        response = TestClient(app).get("/value-error")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "业务分类不合法"})

    def test_unexpected_error_becomes_stable_http_500(self) -> None:
        """验证 RuntimeError 返回稳定的 500 错误，不泄露堆栈细节。

        调用顺序：pytest/unittest 测试入口 -> ApiProtectionTests.test_unexpected_error_becomes_stable_http_500()。
        """
        app = FastAPI()
        register_api_exception_handlers(app)

        @app.get("/unexpected-error")
        def unexpected_error_route():
            """测试辅助路由：始终抛出 RuntimeError，用于验证 HTTP 500 稳定返回。

            调用顺序：pytest/unittest 测试入口 -> ApiProtectionTests.unexpected_error_route()。
            """
            raise RuntimeError("database password leaked in stack")

        response = TestClient(app, raise_server_exceptions=False).get("/unexpected-error")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "服务内部错误，请查看后端日志"})

    def test_http_error_helpers_keep_status_code_semantics(self) -> None:
        """验证错误助手函数返回正确的 HTTP 状态码。

        调用顺序：pytest/unittest 测试入口 -> ApiProtectionTests.test_http_error_helpers_keep_status_code_semantics()。
        """
        cases = [
            (raise_bad_request, 400),
            (raise_not_found, 404),
            (raise_too_many_requests, 429),
        ]

        for raiser, status_code in cases:
            with self.subTest(status_code=status_code):
                with self.assertRaises(HTTPException) as ctx:
                    raiser("错误信息")
                self.assertEqual(ctx.exception.status_code, status_code)
                self.assertEqual(ctx.exception.detail, "错误信息")


if __name__ == "__main__":
    unittest.main()
