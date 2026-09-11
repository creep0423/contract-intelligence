"""V1 无人值守质量周期。

该入口只自动执行可审计的评测动作：主链路评测、意图策略校准、质量/性能门禁
以及 Bad Case 草稿导出。它不会自动修改正式回归集，也不会自动训练或激活模型，
这些动作必须经过人工复核和发布审批。

Windows 任务计划程序和 Linux cron 都只需要定时调用本脚本即可。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import PROJECT_ROOT, configure_utf8_stdio, run_command_step, utc_now, write_json_file


DEFAULT_DATASET = "eval_sets/multi_scenario_smoke.json"
DEFAULT_REPORT = "reports/verification/v1_quality_cycle_latest.json"
DEFAULT_EVALUATION_REPORT = "reports/evaluation/v1_quality_cycle_core_chain.json"
DEFAULT_EVALUATION_GATE = "reports/verification/v1_quality_cycle_evaluation_gate.json"
DEFAULT_BAD_CASES = "eval_sets/v1_quality_cycle_bad_cases.json"
DEFAULT_FEEDBACK_BAD_CASES = "eval_sets/v1_quality_cycle_feedback_bad_cases.json"
DEFAULT_INTENT_POLICY_REPORT = "reports/intent_policy/v1_quality_cycle_latest.json"
DEFAULT_PERFORMANCE_REPORT = "reports/performance/v1_quality_cycle_latest.json"
DEFAULT_PERFORMANCE_GATE = "reports/verification/v1_quality_cycle_performance_gate.json"
DEFAULT_THRESHOLD_CALIBRATION = "reports/threshold_calibration/v1_quality_cycle_candidate.json"
DEFAULT_LOCK = "reports/verification/v1_quality_cycle.lock"


class QualityCycleLock:
    """跨平台的简单文件锁，避免定时任务重叠执行。"""

    def __init__(self, path: Path, stale_after_seconds: int = 12 * 60 * 60) -> None:
        self.path = path
        self.stale_after_seconds = stale_after_seconds

    def __enter__(self) -> "QualityCycleLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            if age < self.stale_after_seconds:
                raise RuntimeError(f"质量周期正在运行或锁文件未过期：{self.path}")
            self.path.unlink()
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(self.path, flags)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"created_at": utc_now(), "pid": os.getpid()}, ensure_ascii=False))
        except FileExistsError as exc:
            raise RuntimeError(f"质量周期正在运行：{self.path}") from exc
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.path.unlink(missing_ok=True)


def _runner(use_docker: bool, *args: str) -> list[str]:
    if use_docker:
        return ["docker", "compose", "--env-file", ".env.compose", "run", "--rm", "api", "python", *args]
    return [sys.executable, *args]


def build_steps(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """构造无人值守质量周期的确定性步骤。"""
    runner = lambda *command: _runner(args.docker, *command)
    steps: list[tuple[str, list[str]]] = [
        (
            "core_chain_evaluation",
            runner(
                "scripts/evaluate_core_chain.py",
                "--dataset", args.dataset,
                "--limit", str(args.limit),
                "--output", args.evaluation_report,
            ),
        ),
        (
            "evaluation_gate",
            runner(
                "scripts/check_evaluation_gate.py",
                "--report", args.evaluation_report,
                "--gate-output", args.evaluation_gate,
            ),
        ),
        (
            "intent_policy_evaluation",
            runner(
                "scripts/evaluate_intent_policy.py",
                "--output", args.intent_policy_report,
                "--fail-on-critical",
            ),
        ),
        (
            "threshold_calibration",
            runner(
                "scripts/calibrate_thresholds.py",
                "--faq-dataset", args.threshold_faq_dataset,
                "--intent-dataset", args.threshold_intent_dataset,
                "--output", args.threshold_calibration_report,
                "--fail-on-insufficient",
            ),
        ),
        (
            "evaluation_bad_case_draft",
            runner(
                "scripts/extract_bad_cases_from_report.py",
                "--report", args.evaluation_report,
                "--output", args.bad_cases,
            ),
        ),
        (
            "feedback_bad_case_draft",
            runner(
                "scripts/export_feedback_bad_cases.py",
                "--scenario", args.scenario,
                "--output", args.feedback_bad_cases,
            ),
        ),
    ]
    if args.include_performance:
        steps.extend(
            [
                (
                    "performance_baseline",
                    runner(
                        "scripts/collect_performance_baseline.py",
                        "--dataset", args.performance_dataset,
                        "--limit", str(args.performance_limit),
                        "--output", args.performance_report,
                        "--allow-errors",
                    ),
                ),
                (
                    "performance_gate",
                    runner(
                        "scripts/check_performance_gate.py",
                        "--report", args.performance_report,
                        "--gate-output", args.performance_gate,
                    ),
                ),
            ]
        )
    return steps


def parse_args() -> argparse.Namespace:
    """解析当前命令行工具的运行参数。

    调用顺序：业务模块或命令行入口 -> parse_args()。
    """
    parser = argparse.ArgumentParser(description="Run the unattended V1 quality cycle.")
    parser.add_argument("--docker", action="store_true", help="在 Compose api 容器中执行评测。")
    parser.add_argument("--scenario", default="enterprise_knowledge")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--include-performance", action="store_true")
    parser.add_argument("--performance-dataset", default="eval_sets/phase1_performance_baseline.json")
    parser.add_argument("--performance-limit", type=int, default=6)
    parser.add_argument("--evaluation-report", default=DEFAULT_EVALUATION_REPORT)
    parser.add_argument("--evaluation-gate", default=DEFAULT_EVALUATION_GATE)
    parser.add_argument("--bad-cases", default=DEFAULT_BAD_CASES)
    parser.add_argument("--feedback-bad-cases", default=DEFAULT_FEEDBACK_BAD_CASES)
    parser.add_argument("--intent-policy-report", default=DEFAULT_INTENT_POLICY_REPORT)
    parser.add_argument("--threshold-faq-dataset", default="eval_sets/threshold_calibration_cases.json")
    parser.add_argument("--threshold-intent-dataset", default="eval_sets/intent_policy_cases.jsonl")
    parser.add_argument("--threshold-calibration-report", default=DEFAULT_THRESHOLD_CALIBRATION)
    parser.add_argument("--performance-report", default=DEFAULT_PERFORMANCE_REPORT)
    parser.add_argument("--performance-gate", default=DEFAULT_PERFORMANCE_GATE)
    parser.add_argument("--output", default=DEFAULT_REPORT)
    parser.add_argument("--lock", default=DEFAULT_LOCK)
    return parser.parse_args()


def main() -> int:
    """执行当前脚本的完整命令行流程。

    调用顺序：业务模块或命令行入口 -> main()。
    """
    configure_utf8_stdio()
    args = parse_args()
    lock_path = PROJECT_ROOT / args.lock
    try:
        with QualityCycleLock(lock_path):
            results = [run_command_step(name, command) for name, command in build_steps(args)]
            payload = {
                "report_type": "v1_unattended_quality_cycle",
                "created_at": utc_now(),
                "ok": all(item.ok for item in results),
                "docker": bool(args.docker),
                "include_performance": bool(args.include_performance),
                "manual_approval_required": True,
                "manual_boundary": [
                    "人工复核 Bad Case 草稿并补齐 expected_* 字段",
                    "人工批准后运行 promote_bad_cases_to_regression.py",
                "人工批准意图模型训练、激活和回滚",
                "人工批准阈值候选后，再运行完整回归和性能门禁",
                ],
                "steps": [
                    {
                        "name": item.name,
                        "ok": item.ok,
                        "returncode": item.returncode,
                        "elapsed_ms": item.elapsed_ms,
                        "command": item.command,
                        "stdout_preview": item.stdout_preview,
                        "stderr_preview": item.stderr_preview,
                    }
                    for item in results
                ],
            }
            write_json_file(PROJECT_ROOT / args.output, payload)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0 if payload["ok"] else 1
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "report_type": "v1_unattended_quality_cycle", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
