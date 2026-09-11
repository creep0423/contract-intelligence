"""验证意图模型制品、评测和治理报告闭环。

调用顺序：课程示例、测试或命令行入口 -> 本模块公开接口。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qa_core.intent.governance import build_intent_model_report, write_intent_model_report


class IntentModelGovernanceTests(unittest.TestCase):
    """验证意图模型治理报告和发布闭环。

    调用顺序：业务模块 -> IntentModelGovernanceTests。
    """
    def test_governance_report_contains_release_closure_fields(self) -> None:
        """验证模型治理报告包含制品、运行和评测闭环字段。

        调用顺序：测试或业务入口 -> IntentModelGovernanceTests.test_governance_report_contains_release_closure_fields()。
        """
        report = build_intent_model_report(evaluate=True)

        self.assertEqual(report["report_type"], "intent_model_governance")
        self.assertTrue(report["ok"], report.get("error"))
        self.assertTrue(report["artifact_ok"])
        self.assertTrue(report["runtime_ok"])
        self.assertEqual(report["model"]["model_version"], "bert-intent-v1")
        self.assertEqual(report["model"]["labels"], ["FAQ_QUERY", "KNOWLEDGE_QUERY", "FOLLOW_UP"])
        self.assertGreaterEqual(report["evaluation"]["accuracy"], 0.75)
        self.assertEqual(report["decision_policy"]["policy_version"], "intent-policy-v1-bert")
        self.assertIn("training_script", report["closure"])
        self.assertIn("policy_eval_script", report["closure"])

    def test_write_intent_model_report_writes_json_file(self) -> None:
        """验证模型治理报告能够稳定写入 JSON 文件。

        调用顺序：测试或业务入口 -> IntentModelGovernanceTests.test_write_intent_model_report_writes_json_file()。
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "intent_model_report.json"

            written = write_intent_model_report(output, evaluate=True)

            self.assertEqual(Path(written), output)
            self.assertTrue(output.exists())
            self.assertIn("intent_model_governance", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
