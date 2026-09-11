"""文档增量复用契约测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from qa_core.indexing.service import _manifest_matches_current_settings


class IngestionIncrementalContractTests(unittest.TestCase):
    """验证文档增量复用必须同时满足文件、模型和 chunk schema 契约。"""

    def test_manifest_match_requires_exact_chunk_schema_version(self) -> None:
        """chunk_schema_version 不一致时，不能复用旧 manifest。"""

        settings = SimpleNamespace(
            embedding_model_version="bge-m3-local-v1",
            chunk_schema_version="parent_child_validity_v2",
        )
        old_record = SimpleNamespace(
            fingerprint="same-file",
            embedding_model_version="bge-m3-local-v1",
            chunk_schema_version="parent_child_v1",
        )
        current_record = SimpleNamespace(
            fingerprint="same-file",
            embedding_model_version="bge-m3-local-v1",
            chunk_schema_version="parent_child_validity_v2",
        )

        self.assertFalse(_manifest_matches_current_settings(old_record, "same-file", settings))
        self.assertTrue(_manifest_matches_current_settings(current_record, "same-file", settings))


if __name__ == "__main__":
    unittest.main()
