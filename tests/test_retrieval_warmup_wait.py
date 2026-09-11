"""检索预热等待逻辑的独立单元测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from qa_core.retrieval import factory


class _FinishedThread:
    """模拟已经结束的预热线程。"""

    def join(self, timeout=None) -> None:
        """记录等待参数，模拟线程 join 完成。"""
        self.timeout = timeout

    def is_alive(self) -> bool:
        """返回线程已经结束。"""
        return False


class RetrievalWarmupWaitTests(unittest.TestCase):
    """验证服务仅在 retrieval warmup ready 后继续启动。"""

    def test_returns_summary_only_after_ready(self) -> None:
        """预热完成时返回预热摘要。"""
        state = {
            "status": "ready",
            "started_at": None,
            "finished_at": None,
            "elapsed_seconds": 1.2,
            "summary": {"collection_count": 16},
            "error": None,
        }
        with (
            patch.object(factory, "_warmup_state", state),
            patch.object(factory, "_warmup_thread", _FinishedThread()),
        ):
            self.assertEqual(factory.wait_for_retrieval_warmup(), {"collection_count": 16})

    def test_raises_when_background_warmup_failed(self) -> None:
        """后台预热失败时向启动流程抛出明确错误。"""
        state = {
            "status": "failed",
            "started_at": None,
            "finished_at": None,
            "elapsed_seconds": 1.2,
            "summary": None,
            "error": "Milvus unavailable",
        }
        with (
            patch.object(factory, "_warmup_state", state),
            patch.object(factory, "_warmup_thread", _FinishedThread()),
        ):
            with self.assertRaisesRegex(RuntimeError, "Milvus unavailable"):
                factory.wait_for_retrieval_warmup()


if __name__ == "__main__":
    unittest.main()
