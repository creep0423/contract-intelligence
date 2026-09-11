"""验证意图策略评测集和失败门禁行为。

调用顺序：课程示例、测试或命令行入口 -> 本模块公开接口。
"""

from __future__ import annotations

import unittest

from scripts.intent.evaluate_intent_policy import DEFAULT_DATASET, build_report, evaluate_case, load_policy_cases


class IntentPolicyEvaluationTests(unittest.TestCase):
    """验证意图策略离线评测和门禁逻辑。

    调用顺序：业务模块 -> IntentPolicyEvaluationTests。
    """
    def test_default_policy_calibration_dataset_passes(self) -> None:
        """验证冻结意图策略样本通过默认门禁。

        调用顺序：测试或业务入口 -> IntentPolicyEvaluationTests.test_default_policy_calibration_dataset_passes()。
        """
        cases = load_policy_cases(DEFAULT_DATASET)

        report = build_report(cases)

        self.assertTrue(report["ok"])
        self.assertGreaterEqual(report["total"], 10)
        self.assertEqual(report["critical_failure_count"], 0)
        self.assertEqual(report["metrics"]["intent_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["question_category_accuracy"], 1.0)
        self.assertIn("threshold_snapshot", report)
        self.assertIn("calibration_guidance", report)

    def test_low_score_default_case_requires_conservative_plan(self) -> None:
        """验证低分默认意图触发保守检索计划。

        调用顺序：测试或业务入口 -> IntentPolicyEvaluationTests.test_low_score_default_case_requires_conservative_plan()。
        """
        case = {
            "case_id": "unit_low_score_guard",
            "question": "帮我分析一下这个问题",
            "scenario_id": "enterprise_knowledge",
            "expected_route": "retrieval",
            "expected_intent": "KNOWLEDGE_QUERY",
            "expected_confidence_max": 0.7,
            "expected_plan_contains": {
                "faq_direct_exact_only": True,
                "faq_direct_threshold": 0.86,
            },
            "critical": True,
        }

        row = evaluate_case(case)

        self.assertTrue(row["ok"], row["failed_checks"])
        self.assertEqual(row["plan"]["faq_direct_exact_only"], True)
        self.assertGreaterEqual(row["plan"]["doc_top_k"], 24)

    def test_failed_expectation_is_reported_as_critical_failure(self) -> None:
        """验证关键策略断言失败会阻断评测门禁。

        调用顺序：测试或业务入口 -> IntentPolicyEvaluationTests.test_failed_expectation_is_reported_as_critical_failure()。
        """
        case = {
            "case_id": "unit_expected_failure",
            "question": "新人入职流程有哪些",
            "scenario_id": "enterprise_knowledge",
            "expected_route": "retrieval",
            "expected_intent": "OUT_OF_SCOPE",
            "critical": True,
        }

        report = build_report([case])

        self.assertFalse(report["ok"])
        self.assertEqual(report["critical_failure_count"], 1)
        self.assertEqual(report["critical_failures"][0]["case_id"], "unit_expected_failure")
        self.assertTrue(report["critical_failures"][0]["failed_checks"])


if __name__ == "__main__":
    unittest.main()
