# -*- coding: utf-8 -*-
"""V1 封版验收脚本编排测试。"""

from __future__ import annotations

import argparse
import unittest

from scripts import verify_v1_release


def _args(**overrides) -> argparse.Namespace:
    """构造 V1 封版验收脚本的默认测试参数。"""
    defaults = {
        "include_runtime_tests": False,
        "include_evaluation": False,
        "include_performance": False,
        "include_docker": False,
        "include_api": False,
        "base_url": "http://127.0.0.1:8000",
        "admin_token": "",
        "scenario": "enterprise_knowledge",
        "evaluation_dataset": str(verify_v1_release.DEFAULT_EVALUATION_DATASET),
        "evaluation_limit": 20,
        "evaluation_report_path": str(verify_v1_release.DEFAULT_EVALUATION_REPORT_PATH),
        "evaluation_gate_path": str(verify_v1_release.DEFAULT_EVALUATION_GATE_PATH),
        "evaluation_bad_cases_path": str(verify_v1_release.DEFAULT_EVALUATION_BAD_CASES_PATH),
        "performance_dataset": str(verify_v1_release.DEFAULT_PERFORMANCE_DATASET),
        "performance_limit": 6,
        "performance_report_path": str(verify_v1_release.DEFAULT_PERFORMANCE_REPORT_PATH),
        "performance_gate_path": str(verify_v1_release.DEFAULT_PERFORMANCE_GATE_PATH),
        "performance_no_warmup": False,
        "performance_max_error_rate": 0.0,
        "performance_max_avg_total_ms": 15000.0,
        "performance_max_p95_total_ms": 30000.0,
        "performance_max_avg_first_token_ms": 8000.0,
        "performance_max_p95_first_token_ms": 15000.0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class V1ReleaseVerificationTests(unittest.TestCase):
    """验证 V1 封版步骤编排稳定。

    调用顺序：pytest/unittest 测试入口 -> V1ReleaseVerificationTests。
    """

    def test_include_evaluation_adds_core_chain_gate_and_bad_case_steps(self) -> None:
        """验证开启评测时会额外加入评测、门禁和 Bad Case 步骤。

        调用顺序：pytest/unittest 测试入口 -> V1ReleaseVerificationTests.test_include_evaluation_adds_core_chain_gate_and_bad_case_steps()。
        """
        args = _args(include_evaluation=True)

        steps = verify_v1_release.build_steps(args)
        names = [name for name, _ in steps]

        self.assertIn("evaluation_core_chain", names)
        self.assertIn("evaluation_gate", names)
        self.assertIn("evaluation_bad_cases", names)
        self.assertLess(names.index("evaluation_core_chain"), names.index("evaluation_gate"))
        self.assertLess(names.index("evaluation_gate"), names.index("evaluation_bad_cases"))

    def test_include_evaluation_with_docker_uses_compose_api_runner(self) -> None:
        """验证开启 Docker 后评测步骤切换到 Compose api 容器执行。"""
        args = _args(include_evaluation=True, include_docker=True)

        steps = dict(verify_v1_release.build_steps(args))
        command = steps["evaluation_core_chain"]

        self.assertGreaterEqual(len(command), 8)
        self.assertEqual(command[:6], ["docker", "compose", "--env-file", ".env.compose", "run", "--rm"])
        self.assertEqual(command[6], "api")
        self.assertEqual(command[7], "python")
        self.assertIn("scripts/evaluate_core_chain.py", command)

    def test_include_performance_adds_baseline_and_gate_steps(self) -> None:
        """验证开启性能验收时会额外加入基线采集和门禁步骤。"""
        args = _args(include_performance=True)

        steps = verify_v1_release.build_steps(args)
        names = [name for name, _ in steps]

        self.assertIn("performance_baseline", names)
        self.assertIn("performance_gate", names)
        self.assertLess(names.index("performance_baseline"), names.index("performance_gate"))
        self.assertLess(names.index("docker_compose_config"), names.index("performance_baseline"))

    def test_include_performance_with_docker_uses_compose_api_runner(self) -> None:
        """验证开启 Docker 后性能步骤切换到 Compose api 容器执行。"""
        args = _args(include_performance=True, include_docker=True)

        steps = dict(verify_v1_release.build_steps(args))
        command = steps["performance_baseline"]

        self.assertGreaterEqual(len(command), 8)
        self.assertEqual(command[:6], ["docker", "compose", "--env-file", ".env.compose", "run", "--rm"])
        self.assertEqual(command[6], "api")
        self.assertEqual(command[7], "python")
        self.assertIn("scripts/quality/collect_performance_baseline.py", command)
        self.assertIn("--allow-errors", command)


if __name__ == "__main__":
    unittest.main()
