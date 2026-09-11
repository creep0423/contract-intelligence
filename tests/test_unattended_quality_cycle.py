"""无人值守 V1 质量周期的编排契约测试。"""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.quality.run_v1_quality_cycle import QualityCycleLock, build_steps


def _args(**overrides) -> argparse.Namespace:
    values = {
        "docker": True,
        "scenario": "enterprise_knowledge",
        "dataset": "eval_sets/multi_scenario_smoke.json",
        "limit": 20,
        "include_performance": True,
        "performance_dataset": "eval_sets/phase1_performance_baseline.json",
        "performance_limit": 6,
        "evaluation_report": "reports/evaluation/test.json",
        "evaluation_gate": "reports/verification/test-gate.json",
        "bad_cases": "eval_sets/test-bad.json",
        "feedback_bad_cases": "eval_sets/test-feedback.json",
        "intent_policy_report": "reports/intent_policy/test.json",
        "threshold_faq_dataset": "eval_sets/threshold_calibration_cases.json",
        "threshold_intent_dataset": "eval_sets/intent_policy_cases.jsonl",
        "threshold_calibration_report": "reports/threshold_calibration/test.json",
        "runtime_param_recommendation_report": "reports/runtime_params/test.json",
        "performance_report": "reports/performance/test.json",
        "performance_gate": "reports/verification/test-performance-gate.json",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class UnattendedQualityCycleTests(unittest.TestCase):
    """验证无人值守质量周期的步骤和互斥控制。

    调用顺序：业务模块 -> UnattendedQualityCycleTests。
    """
    def test_build_steps_contains_gates_and_does_not_promote_or_train(self) -> None:
        """验证无人值守周期只评测和产出候选，不自动训练或发布。

        调用顺序：测试或业务入口 -> UnattendedQualityCycleTests.test_build_steps_contains_gates_and_does_not_promote_or_train()。
        """
        names = [name for name, _ in build_steps(_args())]
        self.assertEqual(
            names,
            [
                "core_chain_evaluation",
                "evaluation_gate",
                "intent_policy_evaluation",
                "threshold_calibration",
                "runtime_param_recommendation",
                "evaluation_bad_case_draft",
                "feedback_bad_case_draft",
                "performance_baseline",
                "performance_gate",
            ],
        )
        commands = [" ".join(command) for _, command in build_steps(_args())]
        self.assertFalse(any("promote_bad_cases_to_regression.py" in command for command in commands))
        self.assertFalse(any("train_intent_bert.py" in command for command in commands))
        self.assertTrue(all(command[:6] == ["docker", "compose", "--env-file", ".env.compose", "run", "--rm"] for command in [command for _, command in build_steps(_args())]))

    def test_lock_prevents_overlapping_runs_and_cleans_up(self) -> None:
        """验证质量周期互斥锁阻止并发运行并在结束后释放。

        调用顺序：测试或业务入口 -> UnattendedQualityCycleTests.test_lock_prevents_overlapping_runs_and_cleans_up()。
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycle.lock"
            with QualityCycleLock(path):
                self.assertTrue(path.exists())
                with self.assertRaises(RuntimeError):
                    with QualityCycleLock(path):
                        pass
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
