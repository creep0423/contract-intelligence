"""Quality-gate tests for local RAG quality rules.

Local reports, eval_sets and gate scripts are the default quality loop.
LangSmith remains an optional tracing and collaboration adapter.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from qa_core.config.settings import get_settings
from qa_core.indexing.chunking import split_documents
from qa_core.indexing.document_loaders import _use_docling_for, load_file
from qa_core.indexing.document_normalizer import normalize_documents
from qa_core.observability.langsmith_adapter import langsmith_enabled, langsmith_status
from qa_core.quality.conflicts import detect_faq_document_conflicts
from qa_core.quality.chunk import analyze_chunk_quality
from qa_core.quality.ingestion import build_ingestion_quality_report
from scripts.quality.check_evaluation_gate import EvaluationGateThresholds
from scripts.quality.check_evaluation_gate import available_eval_sets
from scripts.quality.check_evaluation_gate import evaluate_report_against_gate as evaluate_evaluation_gate
from scripts.quality.check_evaluation_gate import load_evaluation_report
from scripts.quality.check_evaluation_gate import project_path
from scripts.quality.check_ingestion_quality_gate import IngestionQualityThresholds
from scripts.quality.check_ingestion_quality_gate import evaluate_report_against_gate as evaluate_ingestion_gate
from scripts.rebuild_kb_version import build_parser as build_rebuild_kb_version_parser
from scripts.rebuild_kb_version import validate_args as validate_rebuild_kb_version_args


def _clean_ingestion_report() -> dict:
    """调用顺序：pytest/unittest 测试入口 -> _clean_ingestion_report()。
    """
    return {
        "scenario_id": "enterprise_knowledge",
        "files_total": 1,
        "loaded_files_count": 1,
        "unsupported_files_count": 0,
        "failed_files_count": 0,
        "empty_files_count": 0,
        "low_quality_issues_count": 0,
        "duplicate_chunks_count": 0,
        "faq_quality": {
            "exists": True,
            "empty_question_rows": 0,
            "empty_answer_rows": 0,
            "duplicate_questions": 0,
            "invalid_sources": 0,
        },
        "chunk_quality": {
            "low_quality_issue_count": 0,
            "duplicate_chunk_count": 0,
        },
        "faq_document_conflicts": {"conflict_count": 0},
        "metadata_quality": {
            "missing_kb_version_count": 0,
            "missing_embedding_model_version_count": 0,
            "missing_reranker_model_version_count": 0,
            "missing_chunk_schema_version_count": 0,
            "missing_data_scope_count": 0,
        },
        "table_files_count": 0,
        "ocr_risk_files_count": 0,
        "image_risk_files_count": 0,
        "image_risk_blocking_files_count": 0,
        "kb_version": "kb_test",
        "embedding_model_version": "bge-m3-local-v1",
        "chunk_schema_version": "parent_child_validity_v2",
    }


class QualityGateTests(unittest.TestCase):
    """Project-specific quality rules that remain local.

    调用顺序：pytest/unittest 测试入口 -> QualityGateTests。
    """

    def tearDown(self) -> None:
        """清理配置缓存，避免环境变量补丁影响后续测试。

        调用顺序：pytest/unittest 测试入口 -> QualityGateTests.tearDown()。
        """
        get_settings.cache_clear()

    def test_all_frozen_scenarios_have_multiformat_data(self) -> None:
        """验证所有冻结的业务场景包含多格式数据文件。

        调用顺序：pytest/unittest 测试入口 -> QualityGateTests.test_all_frozen_scenarios_have_multiformat_data()。
        """
        required_suffixes = {".md", ".csv", ".xlsx", ".docx", ".pptx", ".pdf"}
        scenario_roots = sorted(path for path in Path("scenarios").iterdir() if (path / "scenario.toml").exists())

        self.assertEqual(len(scenario_roots), 8)
        for scenario_root in scenario_roots:
            data_root = scenario_root / "data"
            suffixes = {path.suffix.lower() for path in data_root.rglob("*") if path.is_file()}
            self.assertTrue(required_suffixes.issubset(suffixes), f"{scenario_root.name} 缺少格式：{sorted(required_suffixes - suffixes)}")

            for suffix in sorted(required_suffixes - {".md"}):
                sample = next(path for path in data_root.rglob("*") if path.is_file() and path.suffix.lower() == suffix)
                docs = load_file(sample)
                content = "\n".join(doc.page_content for doc in docs)
                self.assertTrue(content.strip(), f"{scenario_root.name} 的 {sample.name} 未解析出正文")

    def test_ingestion_gate_rejects_faq_document_conflicts(self) -> None:
        """验证摄入门禁拒绝有 FAQ 文档冲突的报告。

        调用顺序：pytest/unittest 测试入口 -> QualityGateTests.test_ingestion_gate_rejects_faq_document_conflicts()。
        """
        report = _clean_ingestion_report()
        report["faq_document_conflicts"] = {"conflict_count": 1}
        result = evaluate_ingestion_gate(report, IngestionQualityThresholds())
        self.assertFalse(result["ok"])
        self.assertEqual(result["failures"][0]["metric"], "faq_document_conflicts")

    def test_ingestion_gate_passes_clean_report(self) -> None:
        """验证干净的摄入报告通过门禁检查。

        调用顺序：pytest/unittest 测试入口 -> QualityGateTests.test_ingestion_gate_passes_clean_report()。
        """
        result = evaluate_ingestion_gate(_clean_ingestion_report(), IngestionQualityThresholds())
        self.assertTrue(result["ok"])

    def test_rebuild_activation_enables_ingestion_quality_gate(self) -> None:
        """验证重建脚本激活版本时自动开启入库质量门禁。

        调用顺序：pytest/unittest 测试入口 -> QualityGateTests.test_rebuild_activation_enables_ingestion_quality_gate()。
        """
        parser = build_rebuild_kb_version_parser()
        args = parser.parse_args(["--scenario", "enterprise_knowledge", "--new-version", "--activate"])
        validate_rebuild_kb_version_args(parser, args)
        self.assertTrue(args.quality_gate)

    def test_rebuild_activation_rejects_skipped_quality_report(self) -> None:
        """验证重建脚本不允许跳过质量报告后直接激活版本。

        调用顺序：pytest/unittest 测试入口 -> QualityGateTests.test_rebuild_activation_rejects_skipped_quality_report()。
        """
        parser = build_rebuild_kb_version_parser()
        args = parser.parse_args(["--scenario", "enterprise_knowledge", "--new-version", "--skip-quality-report", "--activate"])
        with self.assertRaises(SystemExit):
            validate_rebuild_kb_version_args(parser, args)

    def test_evaluation_gate_passes_clean_existing_report(self) -> None:
        """验证干净的评测报告通过评估门禁。

        调用顺序：pytest/unittest 测试入口 -> QualityGateTests.test_evaluation_gate_passes_clean_existing_report()。
        """
        report = {
            "dataset": "eval_sets/multi_scenario_smoke.json",
            "total": 2,
            "errors": 0,
            "recall_at_k": 1.0,
            "mrr": 1.0,
            "avg_keyword_coverage": 1.0,
            "hit_type_accuracy": 1.0,
            "source_inference_accuracy": 1.0,
            "prompt_profile_accuracy": 1.0,
            "faq_direct_accuracy": 1.0,
            "scenario_isolation_accuracy": 1.0,
            "avg_elapsed_ms": 1200.0,
            "rows": [
                {
                    "scenario_id": "enterprise_knowledge",
                    "expected_effective_source": "finance",
                    "expected_hit_type": "rag",
                    "source_recall_hit": True,
                    "mrr": 1.0,
                    "keyword_coverage": 1.0,
                    "hit_type_matched": True,
                    "source_inference_matched": True,
                    "prompt_profile_matched": True,
                }
            ],
        }
        result = evaluate_evaluation_gate(report, EvaluationGateThresholds())
        self.assertTrue(result["ok"], result["failures"])

    def test_evaluation_gate_missing_report_raises_clear_file_error(self) -> None:
        """验证缺失的评测报告抛出清晰的 FileNotFoundError。

        调用顺序：pytest/unittest 测试入口 -> QualityGateTests.test_evaluation_gate_missing_report_raises_clear_file_error()。
        """
        missing_path = Path("reports") / "evaluation" / "__missing_eval_report__.json"
        with self.assertRaisesRegex(FileNotFoundError, "评测报告不存在"):
            load_evaluation_report(missing_path)

    def test_evaluation_gate_paths_resolve_from_project_root(self) -> None:
        """验证评测门禁路径从项目根目录解析。

        调用顺序：pytest/unittest 测试入口 -> QualityGateTests.test_evaluation_gate_paths_resolve_from_project_root()。
        """
        dataset_path = project_path(Path("eval_sets") / "multi_scenario_smoke.json")
        self.assertTrue(dataset_path.is_absolute())
        self.assertTrue(dataset_path.exists())
        eval_sets = {path.replace("\\", "/") for path in available_eval_sets()}
        self.assertIn("eval_sets/multi_scenario_smoke.json", eval_sets)

    def test_faq_document_conflict_uses_chinese_search_segmentation(self) -> None:
        """验证 FAQ 文档冲突检测使用中文搜索分词。

        调用顺序：pytest/unittest 测试入口 -> QualityGateTests.test_faq_document_conflict_uses_chinese_search_segmentation()。
        """
        docs = [
            Document(
                page_content="管理员忘记密码时，可以在登录页选择忘记密码，并通过绑定邮箱重置。",
                metadata={"source": "account", "file_name": "password.md"},
            ),
            Document(
                page_content="成员离职后应先禁用账号，再回收角色权限、应用授权、API Token 和数据导出权限。",
                metadata={"source": "account", "file_name": "member_offboarding.md"},
            ),
        ]
        report = detect_faq_document_conflicts("scenarios/saas_support/faq.csv", docs)
        missing_account = [
            item
            for item in report["items"]
            if item["source"] == "account" and item["issue"] == "no_related_document"
        ]
        self.assertEqual(missing_account, [])

    def test_chunk_quality_uses_real_length_and_unique_ratio_rules(self) -> None:
        """验证短文本与低字符唯一率使用代码声明的真实阈值。"""
        chunks = [
            Document(page_content="短文本", metadata={"source": "hr"}),
            Document(page_content="A" * 60, metadata={"source": "hr"}),
            Document(page_content="状态：完成", metadata={"source": "finance", "content_type": "table_row"}),
        ]

        issues, _ = analyze_chunk_quality(chunks)
        issue_pairs = {(item["index"], item["issue"]) for item in issues}

        self.assertIn((1, "too_short"), issue_pairs)
        self.assertIn((2, "low_unique_ratio"), issue_pairs)
        self.assertNotIn((3, "too_short"), issue_pairs)

    def test_faq_document_conflict_detects_numeric_and_polarity_mismatch(self) -> None:
        """验证冲突门禁比较真实数字事实和肯定/否定口径。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            faq_path = Path(temp_dir) / "faq.csv"
            faq_path.write_text(
                "question,answer,source\n报销超过多少需要审批,超过5000元可以报销,finance\n",
                encoding="utf-8",
            )
            docs = [
                Document(
                    page_content="报销超过3000元不得直接报销。",
                    metadata={"source": "finance", "file_name": "expense.md"},
                )
            ]

            report = detect_faq_document_conflicts(faq_path, docs)

        self.assertEqual(report["issue_counts"]["numeric_mismatch"], 1)
        self.assertEqual(report["issue_counts"]["polarity_mismatch"], 1)

    def test_csv_table_loader_keeps_row_and_header_semantics(self) -> None:
        """验证 CSV 表格加载器保留行和表头语义。

        调用顺序：pytest/unittest 测试入口 -> QualityGateTests.test_csv_table_loader_keeps_row_and_header_semantics()。
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "materials.csv"
            path.write_text("材料,状态,金额\n验收记录,缺失,50000\n付款申请,待复核,120000\n", encoding="utf-8")
            docs = load_file(path)
        self.assertEqual(len(docs), 2)
        self.assertIn("表头：材料 / 状态 / 金额", docs[0].page_content)
        self.assertIn("- 状态：缺失", docs[0].page_content)
        self.assertEqual(docs[0].metadata["content_type"], "table_row")
        self.assertEqual(docs[0].metadata["row_number"], 1)

    def test_docling_backend_is_configurable_and_keeps_tables_native(self) -> None:
        """Docling 只接管复杂文档解析，CSV/Excel 仍保留项目行级表格语义。

        调用顺序：pytest/unittest 测试入口 -> QualityGateTests.test_docling_backend_is_configurable_and_keeps_tables_native()。
        """
        with patch.dict("os.environ", {"DOCUMENT_PARSER_BACKEND": "docling"}):
            get_settings.cache_clear()
            self.assertTrue(_use_docling_for(Path("handbook.pdf")))
            self.assertTrue(_use_docling_for(Path("slides.pptx")))
            self.assertFalse(_use_docling_for(Path("materials.csv")))

        with patch.dict("os.environ", {"DOCUMENT_PARSER_BACKEND": "native"}):
            get_settings.cache_clear()
            self.assertFalse(_use_docling_for(Path("handbook.pdf")))

    def test_table_chunk_is_not_split_like_normal_text(self) -> None:
        """验证表格块不会被当作普通文本进行拆分。

        调用顺序：pytest/unittest 测试入口 -> QualityGateTests.test_table_chunk_is_not_split_like_normal_text()。
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "materials.csv"
            path.write_text("材料,状态,金额\n验收记录,缺失,50000\n", encoding="utf-8")
            normalized = normalize_documents(load_file(path), path, "quality", "kb_test", "engineering_project_qa")
            chunks, ids = split_documents(normalized)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(ids), 1)
        self.assertEqual(chunks[0].metadata["content_type"], "table_row")
        self.assertEqual(chunks[0].page_content, normalized[0].page_content.strip())
        self.assertEqual(chunks[0].metadata["parent_content"], normalized[0].page_content.strip())
        self.assertIn("验收记录", chunks[0].metadata["parent_content"])

    def test_ingestion_quality_report_marks_table_and_ocr_risk_files(self) -> None:
        """验证摄入质量报告标记表格文件和 OCR 风险文件。

        调用顺序：pytest/unittest 测试入口 -> QualityGateTests.test_ingestion_quality_report_marks_table_and_ocr_risk_files()。
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            quality_dir = data_dir / "quality_data"
            quality_dir.mkdir(parents=True)
            (quality_dir / "materials.csv").write_text("材料,状态\n隐蔽验收记录,缺失\n", encoding="utf-8")
            (quality_dir / "scan_noise.txt").write_text("扫描件 OCR 识别存在 O 和 0 混淆、断行和错字。", encoding="utf-8")
            faq_path = root / "faq.csv"
            faq_path.write_text("question,answer,source\n隐蔽工程验收需要哪些资料,需要隐蔽验收记录。,quality\n", encoding="utf-8")
            report = build_ingestion_quality_report(
                scenario_id="engineering_project_qa",
                data_dir=str(data_dir),
                faq_csv=str(faq_path),
                kb_version="kb_test",
            )
        self.assertEqual(report["table_files_count"], 1)
        self.assertEqual(report["ocr_risk_files_count"], 1)
        result = evaluate_ingestion_gate(report, IngestionQualityThresholds(max_low_quality_issues=10))
        self.assertFalse(result["ok"])
        self.assertIn("ocr_risk_files", {item["metric"] for item in result["failures"]})

    def test_image_file_requires_ocr_review_before_activation(self) -> None:
        """验证独立图片文件会进入图片风险报告，并被入库门禁阻断。

        调用顺序：pytest/unittest 测试入口 -> QualityGateTests.test_image_file_requires_ocr_review_before_activation()。
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            quality_dir = data_dir / "quality_data"
            quality_dir.mkdir(parents=True)
            # 1x1 PNG。文件内容是否真实可看不是重点，门禁依据是图片载体必须先 OCR/复核。
            (quality_dir / "site_photo.png").write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
                b"\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfeA\x7f\x9b\x8b"
                b"\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            faq_path = root / "faq.csv"
            faq_path.write_text("question,answer,source\n现场照片是否齐全,需要人工复核。,quality\n", encoding="utf-8")
            report = build_ingestion_quality_report(
                scenario_id="engineering_project_qa",
                data_dir=str(data_dir),
                faq_csv=str(faq_path),
                kb_version="kb_test",
            )

        self.assertEqual(report["image_risk_files_count"], 1)
        self.assertEqual(report["image_risk_blocking_files_count"], 1)
        self.assertEqual(report["image_risk_files"][0]["severity"], "block")
        result = evaluate_ingestion_gate(
            report,
            IngestionQualityThresholds(
                max_unsupported_files=1,
                max_low_quality_issues=10,
                max_faq_document_conflicts=1,
            ),
        )
        self.assertFalse(result["ok"])
        self.assertIn("image_risk_blocking_files", {item["metric"] for item in result["failures"]})

    def test_reviewed_ocr_markdown_is_allowed_after_manual_review(self) -> None:
        """验证人工复核后的 OCR Markdown 文件通过质量门禁。

        调用顺序：pytest/unittest 测试入口 -> QualityGateTests.test_reviewed_ocr_markdown_is_allowed_after_manual_review()。
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            quality_dir = data_dir / "quality_data"
            quality_dir.mkdir(parents=True)
            (quality_dir / "reviewed_scan_ocr.md").write_text(
                "\n".join(
                    [
                        "# OCR 清洗候选稿：reviewed_scan.pdf",
                        "",
                        "- 原始文件：reviewed_scan.pdf",
                        "- OCR 平均置信度：0.96",
                        "- 复核状态：已复核",
                        "",
                        "人工复核：通过",
                        "",
                        "隐蔽工程照片状态为待补交，责任人为项目经理。",
                    ]
                ),
                encoding="utf-8",
            )
            faq_path = root / "faq.csv"
            faq_path.write_text("question,answer,source\n隐蔽工程照片谁负责,项目经理负责。,quality\n", encoding="utf-8")
            report = build_ingestion_quality_report(
                scenario_id="engineering_project_qa",
                data_dir=str(data_dir),
                faq_csv=str(faq_path),
                kb_version="kb_test",
            )

        self.assertEqual(report["ocr_risk_files_count"], 0)
        self.assertEqual(report["files_loaded_count"], 1)
        result = evaluate_ingestion_gate(report, IngestionQualityThresholds(max_low_quality_issues=10))
        self.assertTrue(result["ok"], result["failures"])

    def test_langsmith_status_is_available_without_api_key(self) -> None:
        """验证没有 API key 时 LangSmith 状态仍然可查询。

        调用顺序：pytest/unittest 测试入口 -> QualityGateTests.test_langsmith_status_is_available_without_api_key()。
        """
        self.assertFalse(langsmith_enabled())
        status = langsmith_status()
        self.assertEqual(status["provider"], "langsmith")
        self.assertIn("project", status)


if __name__ == "__main__":
    unittest.main()
