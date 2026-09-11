# -*- coding: utf-8 -*-
# ============================================================================
# 多轮追问评测门禁 — 把追问链路指标变成可失败的质量检查
# ============================================================================
# 多轮追问比单轮 RAG 更容易暴露真实工程问题：用户会说”那这个呢”，
# 系统必须结合历史把问题改写清楚，同时仍然保持 source、场景和提示词路由正确。
# 本脚本把这些指标变成可失败的质量检查，避免只看单轮评测就误以为历史链路稳定。
#
# 检查指标（7 项）：
#   1. error_rate              — 错误率（必须为 0）
#   2. recall_at_k             — 追问召回率（≥ 0.8）
#   3. avg_keyword_coverage    — 关键事实覆盖（≥ 0.7）
#   4. followup_source_accuracy — 追问后 source 推断准确率（≥ 0.8）
#   5. prompt_profile_accuracy  — Prompt Profile 路由准确率（≥ 0.7）
#   6. scenario_isolation_accuracy — 跨场景隔离准确率（≥ 1.0）
#   7. avg_elapsed_ms          — 平均耗时上限（≤ 60000ms）
#
# 用法示例：
#   python scripts\check_followup_gate.py
#   python scripts\check_followup_gate.py --report reports/evaluation/xxx_followup.json
#   python scripts\check_followup_gate.py --min-recall-at-k 0.85 --max-error-rate 0.05
# ============================================================================

"""检查多轮追问评测报告是否达到发布门槛。

调用顺序：课程示例、测试或命令行入口 -> 本模块公开接口。
"""

from __future__ import annotations

# argparse: 命令行参数解析
import argparse

# sys: 系统功能（sys.path + sys.exit）
import sys

# dataclasses: 数据类定义（FollowupGateThresholds）
from dataclasses import asdict, dataclass

# pathlib.Path: 文件路径操作
from pathlib import Path

# typing.Any: 任意类型
from typing import Any

# ── 路径设置 ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── 导入公共模块 ──
from scripts.common import print_json, read_json_file, write_json_file, write_optional_json
from scripts.evaluate_followup_chain import default_output_path, evaluate_dataset
from scripts.gate_utils import add_max_failure, add_min_failure


@dataclass(frozen=True)
class FollowupGateThresholds:
    """多轮追问门禁阈值。

    调用顺序：命令行入口 -> FollowupGateThresholds。
    """

    min_recall_at_k: float = 0.8
    min_keyword_coverage: float = 0.7
    min_followup_source_accuracy: float = 0.8
    min_prompt_profile_accuracy: float = 0.7
    min_scenario_isolation_accuracy: float = 1.0
    max_error_rate: float = 0.0
    max_avg_elapsed_ms: float = 60000.0


def _metric(report: dict[str, Any], key: str, default: float = 0.0) -> float:
    """安全读取报告中的数值指标。

    调用顺序：命令行入口 -> _metric()。
    """
    try:
        return float(report.get(key, default))
    except (TypeError, ValueError):
        return default


def evaluate_report_against_gate(
    report: dict[str, Any],
    thresholds: FollowupGateThresholds,
    *,
    report_path: str = "",
) -> dict[str, Any]:
    """根据多轮追问指标判断是否通过门禁。

    调用顺序：命令行入口 -> evaluate_report_against_gate()。
    """
    total = int(report.get("total") or 0)
    errors = int(report.get("errors") or 0)
    error_rate = round(errors / max(total, 1), 4)
    metrics = {
        "total": total,
        "followup_turns": int(report.get("followup_turns") or 0),
        "errors": errors,
        "error_rate": error_rate,
        "recall_at_k": _metric(report, "recall_at_k"),
        "avg_keyword_coverage": _metric(report, "avg_keyword_coverage"),
        "followup_source_accuracy": _metric(report, "followup_source_accuracy", 1.0),
        "prompt_profile_accuracy": _metric(report, "prompt_profile_accuracy", 1.0),
        "scenario_isolation_accuracy": _metric(report, "scenario_isolation_accuracy", 1.0),
        "followup_rewrite_rate": _metric(report, "followup_rewrite_rate"),
        "avg_elapsed_ms": _metric(report, "avg_elapsed_ms"),
    }
    failures: list[dict[str, Any]] = []
    add_max_failure(
        failures,
        metric="error_rate",
        actual=metrics["error_rate"],
        maximum=thresholds.max_error_rate,
        message="多轮追问出现错误，说明历史读取、追问改写或检索链路不稳定。",
    )
    add_min_failure(
        failures,
        metric="recall_at_k",
        actual=metrics["recall_at_k"],
        minimum=thresholds.min_recall_at_k,
        message="追问链路预期来源召回不足。",
    )
    add_min_failure(
        failures,
        metric="avg_keyword_coverage",
        actual=metrics["avg_keyword_coverage"],
        minimum=thresholds.min_keyword_coverage,
        message="追问答案关键事实覆盖不足。",
    )
    add_min_failure(
        failures,
        metric="followup_source_accuracy",
        actual=metrics["followup_source_accuracy"],
        minimum=thresholds.min_followup_source_accuracy,
        message="追问后的 source 推断或继承不稳定，容易召回错业务分类。",
    )
    add_min_failure(
        failures,
        metric="prompt_profile_accuracy",
        actual=metrics["prompt_profile_accuracy"],
        minimum=thresholds.min_prompt_profile_accuracy,
        message="追问中的高风险问题没有稳定进入正确提示词模板。",
    )
    add_min_failure(
        failures,
        metric="scenario_isolation_accuracy",
        actual=metrics["scenario_isolation_accuracy"],
        minimum=thresholds.min_scenario_isolation_accuracy,
        message="多轮追问出现跨场景污染。",
    )
    add_max_failure(
        failures,
        metric="avg_elapsed_ms",
        actual=metrics["avg_elapsed_ms"],
        maximum=thresholds.max_avg_elapsed_ms,
        message="多轮追问平均耗时过高。",
    )
    return {
        "ok": not failures,
        "report_type": "followup_chain_gate",
        "report_path": report_path,
        "dataset": report.get("dataset"),
        "created_at": report.get("created_at"),
        "metrics": metrics,
        "thresholds": asdict(thresholds),
        "failures": failures,
    }


def thresholds_from_args(args: argparse.Namespace) -> FollowupGateThresholds:
    """把命令行参数转换为门禁阈值。

    调用顺序：命令行入口 -> thresholds_from_args()。
    """
    return FollowupGateThresholds(
        min_recall_at_k=args.min_recall_at_k,
        min_keyword_coverage=args.min_keyword_coverage,
        min_followup_source_accuracy=args.min_followup_source_accuracy,
        min_prompt_profile_accuracy=args.min_prompt_profile_accuracy,
        min_scenario_isolation_accuracy=args.min_scenario_isolation_accuracy,
        max_error_rate=args.max_error_rate,
        max_avg_elapsed_ms=args.max_avg_elapsed_ms,
    )


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    调用顺序：命令行入口 -> build_parser()。
    """
    parser = argparse.ArgumentParser(description="Check follow-up evaluation report against gate thresholds.")
    parser.add_argument("--report", default="", help="已有多轮追问评测报告路径。未提供时现场执行。")
    parser.add_argument("--dataset", default=str(Path("eval_sets") / "multi_turn_followup_regression.json"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="", help="现场评测报告输出路径。")
    parser.add_argument("--gate-output", default="", help="门禁摘要输出路径。")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--visibility", default=None)
    parser.add_argument("--user-role", default=None)
    parser.add_argument("--kb-version", default=None)
    parser.add_argument("--min-recall-at-k", type=float, default=0.8)
    parser.add_argument("--min-keyword-coverage", type=float, default=0.7)
    parser.add_argument("--min-followup-source-accuracy", type=float, default=0.8)
    parser.add_argument("--min-prompt-profile-accuracy", type=float, default=0.7)
    parser.add_argument("--min-scenario-isolation-accuracy", type=float, default=1.0)
    parser.add_argument("--max-error-rate", type=float, default=0.0)
    parser.add_argument("--max-avg-elapsed-ms", type=float, default=60000.0)
    return parser


def main() -> None:
    """执行多轮追问门禁。

    调用顺序：命令行入口 -> main()。
    """
    parser = build_parser()
    args = parser.parse_args()
    if args.report:
        report_path = args.report
        report = read_json_file(report_path)
    else:
        output_path = Path(args.output) if args.output else default_output_path(args.dataset)
        report = evaluate_dataset(args)
        report_path = str(output_path)
        write_json_file(output_path, report)
    result = evaluate_report_against_gate(report, thresholds_from_args(args), report_path=report_path)
    write_optional_json(args.gate_output, result)
    print_json(result)
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
