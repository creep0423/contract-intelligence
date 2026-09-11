"""聊天历史存储的轻量单元测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from qa_core.memory.history import ChatHistoryStore


class ChatHistoryStoreTests(unittest.TestCase):
    """验证历史存储复用共享 SQLAlchemy engine，而不是每次创建新连接字符串引擎。

    调用顺序：pytest/unittest 测试入口 -> ChatHistoryStoreTests。
    """

    def test_for_session_reuses_store_engine(self) -> None:
        """验证 for_session 复用 ChatHistoryStore 的 SQLAlchemy engine。

        调用顺序：pytest/unittest 测试入口 -> ChatHistoryStoreTests.test_for_session_reuses_store_engine()。
        """
        store = ChatHistoryStore()
        fake_engine = object()
        store._engine = fake_engine

        with patch("qa_core.memory.history.SQLChatMessageHistory") as history_cls:
            store.for_session("unit-session")

        history_cls.assert_called_once()
        kwargs = history_cls.call_args.kwargs
        self.assertIs(kwargs["connection"], fake_engine)
        self.assertEqual(kwargs["session_id"], "unit-session")
        self.assertEqual(kwargs["table_name"], "chat_messages")


if __name__ == "__main__":
    unittest.main()
