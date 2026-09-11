"""Metadata mode contract tests for FAQ snapshots and document increments."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from qa_core.indexing.document_normalizer import normalize_documents
from qa_core.indexing.faq_ingestion import faq_documents_from_csv


class IngestionMetadataModeTests(unittest.TestCase):
    """Ensure FAQ and document rows explain their versioning semantics."""

    def test_faq_metadata_marks_snapshot_versioning(self) -> None:
        """FAQ rows are version snapshots even though they carry shared version fields."""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "faq.csv"
            path.write_text(
                "question,answer,source\n"
                "哪些数据属于敏感个人信息？,身份证件和金融账户等属于敏感个人信息。,privacy\n",
                encoding="utf-8",
            )
            docs, ids = faq_documents_from_csv(
                str(path),
                kb_version="kb_test_v17",
                version_seq=17,
                scenario_id="compliance_qa",
            )

        self.assertEqual(len(docs), 1)
        self.assertEqual(len(ids), 1)
        metadata = docs[0].metadata
        self.assertEqual(metadata["source_type"], "faq")
        self.assertEqual(metadata["record_type"], "faq")
        self.assertEqual(metadata["versioning_mode"], "snapshot")
        self.assertEqual(metadata["version_filter_mode"], "kb_version_exact")
        self.assertEqual(metadata["kb_version"], "kb_test_v17")
        self.assertEqual(metadata["valid_from_seq"], 17)
        self.assertEqual(metadata["valid_to_seq"], 0)

    def test_document_metadata_marks_reference_incremental_versioning(self) -> None:
        """Document chunks use validity windows for cross-version visibility."""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "briefing.txt"
            path.write_text("供应商声明缺失，需补充声明扫描件。", encoding="utf-8")
            normalized = normalize_documents(
                [Document(page_content="供应商声明缺失，需补充声明扫描件。", metadata={})],
                path,
                "audit",
                "kb_test_v17",
                "compliance_qa",
                17,
            )

        metadata = normalized[0].metadata
        self.assertEqual(metadata["source_type"], "doc")
        self.assertEqual(metadata["record_type"], "doc_chunk")
        self.assertEqual(metadata["versioning_mode"], "reference_incremental")
        self.assertEqual(metadata["version_filter_mode"], "validity_window")
        self.assertEqual(metadata["kb_version"], "kb_test_v17")
        self.assertEqual(metadata["valid_from_seq"], 17)
        self.assertEqual(metadata["valid_to_seq"], 0)


if __name__ == "__main__":
    unittest.main()
