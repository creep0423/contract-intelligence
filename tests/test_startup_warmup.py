"""应用启动预热的异步边界测试。"""

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import AsyncMock, patch


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI is not installed in this interpreter")
class StartupWarmupTests(unittest.IsolatedAsyncioTestCase):
    """验证服务不会在检索栈预热前完成启动。"""

    async def test_lifespan_waits_for_retrieval_warmup(self) -> None:
        """FastAPI lifespan 在接收流量前等待 BGE、Reranker 和 Milvus 就绪。"""
        import app

        with (
            patch("app.validate_runtime_environment", return_value={"ok": True}),
            patch("app.bootstrap_mysql_schema", return_value={"ok": True}),
            patch("app.validate_active_kb_versions", return_value={"ok": True}),
            patch("app.warmup_intent_decision_gateway", return_value={"model": "ready"}) as warmup_intent,
            patch("app.start_retrieval_warmup_background", return_value={"status": "running"}) as start_retrieval,
            patch("app.wait_for_retrieval_warmup", return_value={"status": "ready"}) as wait_retrieval,
            patch("app.refresh_llm_status_background", new_callable=AsyncMock),
        ):
            async with app.lifespan(app.app):
                pass

        warmup_intent.assert_called_once()
        start_retrieval.assert_called_once()
        wait_retrieval.assert_called_once()


if __name__ == "__main__":
    unittest.main()
