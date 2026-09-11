"""Regression guards for V1.0 maintenance ingestion call bindings."""

from __future__ import annotations

from pathlib import Path
import unittest


def _call_block(path: str, marker: str) -> str:
    """获取调用块中从标记到右括号之间的代码片段。

    调用顺序：pytest/unittest 测试入口 -> _call_block()。
    """
    source = Path(path).read_text(encoding="utf-8")
    return source.split(marker, 1)[1].split(")", 1)[0]


class V1MaintenanceBindingTests(unittest.TestCase):
    """验证 V1.0 维护版摄入调用的参数绑定。

    调用顺序：pytest/unittest 测试入口 -> V1MaintenanceBindingTests。
    """

    def test_faq_ingestion_passes_scenario_id_as_keyword(self) -> None:
        """验证 FAQ 摄入以关键字参数传入 scenario_id。

        调用顺序：pytest/unittest 测试入口 -> V1MaintenanceBindingTests.test_faq_ingestion_passes_scenario_id_as_keyword()。
        """
        call_block = _call_block(
            "qa_core/indexing/faq_ingestion.py",
            "docs, ids = faq_documents_from_csv(",
        )

        self.assertIn("scenario_id=scenario.scenario_id", call_block)
        self.assertIn("version_seq=version.version_seq", call_block)
        self.assertNotIn("        scenario.scenario_id,", call_block)

    def test_quality_report_passes_data_scope_as_keyword(self) -> None:
        """验证质量报告以关键字参数传入 data_scope。

        调用顺序：pytest/unittest 测试入口 -> V1MaintenanceBindingTests.test_quality_report_passes_data_scope_as_keyword()。
        """
        call_block = _call_block(
            "qa_core/quality/ingestion.py",
            "normalized = normalize_documents(",
        )

        self.assertIn("kb_version=active_kb_version", call_block)
        self.assertIn("scenario_id=scenario.scenario_id", call_block)
        self.assertIn("data_scope=scope", call_block)
        self.assertIn("allowed_roles=allowed_roles", call_block)
        self.assertNotIn("            scope,", call_block)


if __name__ == "__main__":
    unittest.main()
