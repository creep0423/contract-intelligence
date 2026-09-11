"""MySQL 控制面存储测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

import pytest
from sqlalchemy import text

from qa_core.governance.chunk_versions import KB_CHUNK_VERSIONS_TABLE, ChunkVersionIndex
from qa_core.governance.kb_versions import (
    KB_ACTIVATION_TABLE,
    KB_ACTIVE_TABLE,
    KB_VERSIONS_TABLE,
    KnowledgeBaseVersionStore,
)
from qa_core.indexing.manifest import INDEX_MANIFEST_TABLE, IndexManifest
from qa_core.storage.bootstrap import bootstrap_mysql_schema


pytestmark = pytest.mark.mysql_integration


class MySqlMetadataStoreTests(unittest.TestCase):
    """验证知识库版本和文档 manifest 都落在 MySQL。

    调用顺序：pytest/unittest 测试入口 -> MySqlMetadataStoreTests。
    """

    scenario_id = "enterprise_knowledge"

    def setUp(self) -> None:
        """测试前置：初始化 MySQL schema 并清理遗留数据。

        调用顺序：pytest/unittest 测试入口 -> MySqlMetadataStoreTests.setUp()。
        """
        bootstrap_mysql_schema()
        self.version_store = KnowledgeBaseVersionStore(self.scenario_id)
        self.original_pointer = self.version_store._active_pointer()
        self._cleanup_versions()
        self._cleanup_activation_history()

        self.manifest = IndexManifest()
        self._cleanup_manifest()
        self.chunk_index = ChunkVersionIndex()
        self._cleanup_chunk_versions()

    def tearDown(self) -> None:
        """测试后置：清理测试数据并恢复原始指针状态。

        调用顺序：pytest/unittest 测试入口 -> MySqlMetadataStoreTests.tearDown()。
        """
        self._cleanup_versions()
        self._cleanup_activation_history()
        self._cleanup_chunk_versions()
        with self.version_store.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO {KB_ACTIVE_TABLE}
                        (scenario_id, active_kb_version, previous_kb_version)
                    VALUES
                        (:scenario_id, :active_kb_version, :previous_kb_version)
                    ON DUPLICATE KEY UPDATE
                        active_kb_version=VALUES(active_kb_version),
                        previous_kb_version=VALUES(previous_kb_version)
                    """
                ),
                {
                    "scenario_id": self.scenario_id,
                    "active_kb_version": self.original_pointer[0],
                    "previous_kb_version": self.original_pointer[1],
                },
            )
        self._cleanup_manifest()

    def _cleanup_versions(self) -> None:
        """清理测试用的版本记录。

        调用顺序：pytest/unittest 测试入口 -> MySqlMetadataStoreTests._cleanup_versions()。
        """
        with self.version_store.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    DELETE FROM {KB_VERSIONS_TABLE}
                    WHERE scenario_id=:scenario_id AND kb_version LIKE 'kb_mysql_unit_%'
                    """
                ),
                {"scenario_id": self.scenario_id},
            )

    def _cleanup_activation_history(self) -> None:
        """清理测试用的激活历史记录。

        调用顺序：pytest/unittest 测试入口 -> MySqlMetadataStoreTests._cleanup_activation_history()。
        """
        with self.version_store.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    DELETE FROM {KB_ACTIVATION_TABLE}
                    WHERE scenario_id=:scenario_id
                      AND (from_kb_version LIKE 'kb_mysql_unit_%'
                           OR to_kb_version LIKE 'kb_mysql_unit_%')
                    """
                ),
                {"scenario_id": self.scenario_id},
            )

    def _cleanup_chunk_versions(self) -> None:
        """清理测试用的块版本记录。

        调用顺序：pytest/unittest 测试入口 -> MySqlMetadataStoreTests._cleanup_chunk_versions()。
        """
        with self.chunk_index.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    DELETE FROM {KB_CHUNK_VERSIONS_TABLE}
                    WHERE scenario_id=:scenario_id AND chunk_id LIKE 'mysql-unit-%'
                    """
                ),
                {"scenario_id": self.scenario_id},
            )

    def _cleanup_manifest(self) -> None:
        """清理测试用的 Manifest 记录。

        调用顺序：pytest/unittest 测试入口 -> MySqlMetadataStoreTests._cleanup_manifest()。
        """
        with self.manifest.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    DELETE FROM {INDEX_MANIFEST_TABLE}
                    WHERE scenario_id=:scenario_id AND kb_version LIKE 'kb_mysql_unit_%'
                    """
                ),
                {"scenario_id": self.scenario_id},
            )

    def test_kb_versions_use_mysql_control_plane(self) -> None:
        """验证知识库版本使用 MySQL 控制平面存储。

        调用顺序：pytest/unittest 测试入口 -> MySqlMetadataStoreTests.test_kb_versions_use_mysql_control_plane()。
        """
        v1 = self.version_store.ensure_version("kb_mysql_unit_v1")
        v2 = self.version_store.ensure_version("kb_mysql_unit_v2")
        self.version_store.activate_version(v1.kb_version)
        self.version_store.activate_version(v2.kb_version)

        payload = self.version_store.as_payload()

        self.assertEqual(payload["metadata_store"], "mysql")
        self.assertEqual(payload["active_version"], "kb_mysql_unit_v2")
        self.assertEqual(self.version_store.get("kb_mysql_unit_v1").status, "STAGED")
        self.assertEqual(self.version_store.get("kb_mysql_unit_v2").status, "ACTIVE")

    def test_activation_history_supports_multi_step_rollback_audit(self) -> None:
        """验证激活历史支持多步回滚审计。

        调用顺序：pytest/unittest 测试入口 -> MySqlMetadataStoreTests.test_activation_history_supports_multi_step_rollback_audit()。
        """
        v1 = self.version_store.ensure_version("kb_mysql_unit_v1")
        v2 = self.version_store.ensure_version("kb_mysql_unit_v2")
        v3 = self.version_store.ensure_version("kb_mysql_unit_v3")

        self.version_store.activate_version(v1.kb_version, reason="init", activated_by="unit")
        self.version_store.activate_version(v2.kb_version, reason="promote", activated_by="unit")
        self.version_store.activate_version(v3.kb_version, reason="promote", activated_by="unit")
        self.version_store.activate_version(v1.kb_version, reason="rollback test", activated_by="unit")

        history = self.version_store.list_activation_history(limit=4)

        self.assertEqual(history[0]["from_kb_version"], "kb_mysql_unit_v3")
        self.assertEqual(history[0]["to_kb_version"], "kb_mysql_unit_v1")
        self.assertEqual(history[0]["action"], "rollback")
        self.assertEqual(history[0]["reason"], "rollback test")
        self.assertEqual(history[0]["activated_by"], "unit")
        self.assertEqual(self.version_store.as_payload()["activation_history"][0]["to_kb_version"], "kb_mysql_unit_v1")

    def test_index_manifest_uses_mysql_records(self) -> None:
        """验证索引 Manifest 使用 MySQL 存储记录。

        调用顺序：pytest/unittest 测试入口 -> MySqlMetadataStoreTests.test_index_manifest_uses_mysql_records()。
        """
        path = Path("scenarios/enterprise_knowledge/data/hr_data/onboarding.md")
        self.manifest.update(
            "hr",
            path,
            "fingerprint-1",
            ["chunk-a", "chunk-b"],
            scenario_id=self.scenario_id,
            kb_version="kb_mysql_unit_v1",
            embedding_model_version="embed-v1",
            chunk_schema_version="schema-v1",
        )

        record = self.manifest.get("hr", path, "kb_mysql_unit_v1", self.scenario_id)
        records = self.manifest.iter_records(scenario_id=self.scenario_id, kb_version="kb_mysql_unit_v1")

        self.assertIsNotNone(record)
        self.assertEqual(record.chunk_ids, ["chunk-a", "chunk-b"])
        self.assertEqual(len(records), 1)
        removed = self.manifest.remove_by_key(record.key)
        self.assertEqual(removed.fingerprint, "fingerprint-1")
        self.assertIsNone(self.manifest.get("hr", path, "kb_mysql_unit_v1", self.scenario_id))


    def test_record_incremental_base_is_stored_in_version_stats(self) -> None:
        """验证增量基础版本记录在版本统计信息中。

        调用顺序：pytest/unittest 测试入口 -> MySqlMetadataStoreTests.test_record_incremental_base_is_stored_in_version_stats()。
        """
        self.version_store.ensure_version("kb_mysql_unit_base")
        self.version_store.ensure_version("kb_mysql_unit_candidate")

        version = self.version_store.record_incremental_base("kb_mysql_unit_candidate", "kb_mysql_unit_base")
        version = self.version_store.record_ingest_result(
            version.kb_version,
            content_type="doc",
            count=3,
            source="hr",
            extra_stats={"last_doc_reused_count": 2, "last_doc_reembedded_count": 1},
        )

        self.assertEqual(version.stats["incremental_base_kb_version"], "kb_mysql_unit_base")
        self.assertEqual(version.stats["incremental_mode"], "reference_delta_validity_window")
        self.assertEqual(version.stats["last_doc_reused_count"], 2)
        self.assertEqual(version.stats["last_doc_reembedded_count"], 1)
        self.assertIn("hr", version.sources)

    def test_chunk_version_index_tracks_visible_validity_window(self) -> None:
        """验证块版本索引追踪可见的有效性窗口。

        调用顺序：pytest/unittest 测试入口 -> MySqlMetadataStoreTests.test_chunk_version_index_tracks_visible_validity_window()。
        """
        self.chunk_index.upsert_chunks(
            ["mysql-unit-a", "mysql-unit-b"],
            scenario_id=self.scenario_id,
            source="hr",
            kb_version="kb_mysql_unit_v1",
            valid_from_seq=1,
            file_path="hr/onboarding.md",
        )
        self.chunk_index.expire_chunks(
            ["mysql-unit-a"],
            scenario_id=self.scenario_id,
            source="hr",
            kb_version="kb_mysql_unit_v1",
            valid_from_seq=1,
            valid_to_seq=2,
            file_path="hr/onboarding.md",
        )
        self.chunk_index.upsert_chunks(
            ["mysql-unit-c"],
            scenario_id=self.scenario_id,
            source="hr",
            kb_version="kb_mysql_unit_v2",
            valid_from_seq=2,
            file_path="hr/onboarding.md",
        )

        visible_v1 = {item.chunk_id for item in self.chunk_index.list_visible(scenario_id=self.scenario_id, active_seq=1)}
        visible_v2 = {item.chunk_id for item in self.chunk_index.list_visible(scenario_id=self.scenario_id, active_seq=2)}

        self.assertEqual(visible_v1, {"mysql-unit-a", "mysql-unit-b"})
        self.assertEqual(visible_v2, {"mysql-unit-b", "mysql-unit-c"})


if __name__ == "__main__":
    unittest.main()
