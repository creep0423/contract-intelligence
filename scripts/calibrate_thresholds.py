"""基于标注样本生成 V1 阈值候选策略，不直接修改生产配置。

当前校准两个能够由离线真值可靠判断的核心阈值：
1. FAQ 相似直出的最低分数；
2. BERT 意图模型参与规则仲裁的最低分数。

质量红线、性能 SLA 和缓存 TTL 属于业务/运维策略，不由本脚本自动优化。
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import PROJECT_ROOT, configure_utf8_stdio, print_json, utc_now, write_json_file


DEFAULT_FAQ_DATASET = PROJECT_ROOT / "eval_sets" / "threshold_calibration_cases.json"
DEFAULT_INTENT_DATASET = PROJECT_ROOT / "eval_sets" / "intent_policy_cases.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "threshold_calibration" / "threshold_candidate_latest.json"


@dataclass(frozen=True)
class BinaryObservation:
    """表示一个阈值样本的分数、标签和业务标识。

    调用顺序：业务模块 -> BinaryObservation。
    """
    case_id: str
    score: float
    positive: bool
    false_positive_cost: float = 4.0
    false_negative_cost: float = 1.0


def candidate_values(start: float = 0.50, end: float = 0.95, step: float = 0.01) -> list[float]:
    """从样本分数生成可评估的阈值候选集合。

    调用顺序：业务模块或命令行入口 -> candidate_values()。
    """
    count = int(round((end - start) / step))
    return [round(start + index * step, 4) for index in range(count + 1)]


def evaluate_binary_threshold(observations: list[BinaryObservation], threshold: float) -> dict[str, Any]:
    """计算指定阈值下的精确率、召回率和误直出数量。

    调用顺序：业务模块或命令行入口 -> evaluate_binary_threshold()。
    """
    tp = fp = tn = fn = 0
    weighted_loss = 0.0
    for item in observations:
        predicted = item.score >= threshold
        if predicted and item.positive:
            tp += 1
        elif predicted:
            fp += 1
            weighted_loss += item.false_positive_cost
        elif item.positive:
            fn += 1
            weighted_loss += item.false_negative_cost
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    negatives = fp + tn
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_direct_rate": round(fp / negatives, 4) if negatives else 0.0,
        "weighted_loss": round(weighted_loss, 4),
    }


def select_faq_threshold(
    observations: list[BinaryObservation],
    *,
    min_precision: float,
    min_recall: float,
    max_false_direct_rate: float,
    current_threshold: float = 0.72,
) -> dict[str, Any]:
    """在业务约束内选择 FAQ 直出阈值候选。

    调用顺序：业务模块或命令行入口 -> select_faq_threshold()。
    """
    if not observations:
        return {
            "ok": False,
            "selected": None,
            "constraints": {
                "min_precision": min_precision,
                "min_recall": min_recall,
                "max_false_direct_rate": max_false_direct_rate,
            },
            "scan": [],
        }
    scans = [evaluate_binary_threshold(observations, value) for value in candidate_values()]
    feasible = [
        row
        for row in scans
        if row["precision"] >= min_precision
        and row["recall"] >= min_recall
        and row["false_direct_rate"] <= max_false_direct_rate
    ]
    pool = feasible or scans
    selected = min(
        pool,
        key=lambda row: (
            row["weighted_loss"],
            -row["precision"],
            -row["recall"],
            abs(row["threshold"] - current_threshold),
            -row["threshold"],
        ),
    )
    return {
        "ok": bool(feasible),
        "selected": selected,
        "constraints": {
            "min_precision": min_precision,
            "min_recall": min_recall,
            "max_false_direct_rate": max_false_direct_rate,
        },
        "scan": scans,
    }


def load_json_list(path: str | Path) -> list[dict[str, Any]]:
    """读取并校验由对象列表组成的 JSON 数据集。

    调用顺序：业务模块或命令行入口 -> load_json_list()。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"校准集必须是 JSON list：{path}")
    return [dict(item) for item in payload]


def collect_faq_observations(cases: list[dict[str, Any]]) -> tuple[list[BinaryObservation], list[dict[str, str]]]:
    """从校准数据集中提取 FAQ 直出阈值样本。

    调用顺序：业务模块或命令行入口 -> collect_faq_observations()。
    """
    factory_module = importlib.import_module("qa_core.application.factory")
    service = factory_module.get_qa_service()
    observations: list[BinaryObservation] = []
    failures: list[dict[str, str]] = []
    for index, case in enumerate(cases, start=1):
        case_id = str(case.get("case_id") or f"faq_threshold_{index}")
        try:
            payload = service.debug_retrieval(
                str(case["question"]),
                case.get("source_filter"),
                f"threshold-calibration-{index}",
                scenario_id=str(case.get("scenario_id") or "enterprise_knowledge"),
            )
            faq_sources = list(payload.get("faq_sources") or [])
            score = float(faq_sources[0].get("score") or 0.0) if faq_sources else 0.0
            observations.append(
                BinaryObservation(
                    case_id=case_id,
                    score=score,
                    positive=bool(case.get("expected_faq_direct")),
                    false_positive_cost=float(case.get("false_positive_cost") or 4.0),
                    false_negative_cost=float(case.get("false_negative_cost") or 1.0),
                )
            )
        except Exception as exc:
            failures.append({"case_id": case_id, "error": str(exc)})
    return observations, failures


def _top_candidate(candidates: list[dict[str, Any]], source: str) -> dict[str, Any] | None:
    selected = [item for item in candidates if item.get("source") == source]
    return max(selected, key=lambda item: float(item.get("score") or 0.0)) if selected else None


def _simulate_intent(case: dict[str, Any], row: dict[str, Any], model_min_score: float) -> str:
    intent = row.get("intent") or {}
    candidates = list(intent.get("candidate_intents") or [])
    rule = _top_candidate(candidates, "rule")
    model = _top_candidate(candidates, "model")
    if not rule or not model:
        return str(intent.get("intent") or "")
    rule_intent = str(rule.get("intent") or "")
    model_intent = str(model.get("intent") or "")
    model_score = float(model.get("score") or 0.0)
    has_history = bool(case.get("history"))
    if model_intent == "FOLLOW_UP" and not has_history:
        return rule_intent
    if model_score < model_min_score:
        return rule_intent
    if model_intent == rule_intent:
        return rule_intent
    if str(rule.get("reason") or "") == "default_knowledge":
        return model_intent
    return rule_intent


def select_intent_model_threshold(
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    current_threshold: float = 0.55,
) -> dict[str, Any]:
    """根据意图模型样本选择最低接管分数候选。

    调用顺序：业务模块或命令行入口 -> select_intent_model_threshold()。
    """
    evaluated: list[dict[str, Any]] = []
    for threshold in candidate_values(0.40, 0.90, 0.01):
        correct = weighted_errors = total = 0
        for case, row in zip(cases, rows, strict=True):
            expected = str(case.get("expected_intent") or "")
            if not expected or str(case.get("expected_route") or "retrieval") == "direct_answer":
                continue
            total += 1
            actual = _simulate_intent(case, row, threshold)
            if actual == expected:
                correct += 1
            else:
                weighted_errors += 5 if case.get("critical") else 1
        evaluated.append(
            {
                "threshold": threshold,
                "total": total,
                "accuracy": round(correct / total, 4) if total else 0.0,
                "weighted_errors": weighted_errors,
            }
        )
    selected = min(
        evaluated,
        key=lambda row: (
            row["weighted_errors"],
            -row["accuracy"],
            abs(row["threshold"] - current_threshold),
        ),
    )
    return {"ok": selected["total"] >= 6, "selected": selected, "scan": evaluated}


def build_parser() -> argparse.ArgumentParser:
    """构造当前命令行工具的参数解析器。

    调用顺序：业务模块或命令行入口 -> build_parser()。
    """
    parser = argparse.ArgumentParser(description="Calibrate governed V1 thresholds from labeled samples.")
    parser.add_argument("--faq-dataset", default=str(DEFAULT_FAQ_DATASET.relative_to(PROJECT_ROOT)))
    parser.add_argument("--intent-dataset", default=str(DEFAULT_INTENT_DATASET.relative_to(PROJECT_ROOT)))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)))
    parser.add_argument("--min-faq-precision", type=float, default=0.95)
    parser.add_argument("--min-faq-recall", type=float, default=0.80)
    parser.add_argument("--max-faq-false-direct-rate", type=float, default=0.05)
    parser.add_argument("--min-positive-cases", type=int, default=4)
    parser.add_argument("--min-negative-cases", type=int, default=4)
    parser.add_argument("--fail-on-insufficient", action="store_true")
    return parser


def main() -> int:
    """执行当前脚本的完整命令行流程。

    调用顺序：业务模块或命令行入口 -> main()。
    """
    configure_utf8_stdio()
    args = build_parser().parse_args()
    settings_module = importlib.import_module("qa_core.config.settings")
    decision_module = importlib.import_module("qa_core.intent.decision")
    intent_evaluation_module = importlib.import_module("scripts.evaluate_intent_policy")

    faq_cases = load_json_list(PROJECT_ROOT / args.faq_dataset)
    settings = settings_module.get_settings()
    observations, collection_failures = collect_faq_observations(faq_cases)
    positive_count = sum(1 for item in observations if item.positive)
    negative_count = len(observations) - positive_count
    faq_calibration = select_faq_threshold(
        observations,
        min_precision=args.min_faq_precision,
        min_recall=args.min_faq_recall,
        max_false_direct_rate=args.max_faq_false_direct_rate,
        current_threshold=settings.faq_direct_score_threshold,
    )
    faq_calibration["sample_counts"] = {
        "total": len(observations),
        "positive": positive_count,
        "negative": negative_count,
        "collection_failures": len(collection_failures),
    }
    faq_calibration["sufficient_samples"] = (
        positive_count >= args.min_positive_cases
        and negative_count >= args.min_negative_cases
        and not collection_failures
    )

    intent_cases = intent_evaluation_module.load_policy_cases(PROJECT_ROOT / args.intent_dataset)
    intent_rows = [intent_evaluation_module.evaluate_case(case) for case in intent_cases]
    intent_calibration = select_intent_model_threshold(
        intent_cases,
        intent_rows,
        current_threshold=decision_module.POLICY.model_min_score,
    )
    ok = bool(faq_calibration["ok"] and faq_calibration["sufficient_samples"] and intent_calibration["ok"])
    payload = {
        "report_type": "v1_threshold_calibration_candidate",
        "created_at": utc_now(),
        "ok": ok,
        "applied": False,
        "approval_required": True,
        "current_policy": {
            "faq_direct_score_threshold": settings.faq_direct_score_threshold,
            "intent_model_min_score": decision_module.POLICY.model_min_score,
        },
        "candidate_policy": {
            "faq_direct_score_threshold": (
                faq_calibration["selected"]["threshold"] if faq_calibration["selected"] else None
            ),
            "intent_model_min_score": intent_calibration["selected"]["threshold"],
        },
        "faq_direct": faq_calibration,
        "intent_model": intent_calibration,
        "collection_failures": collection_failures,
        "manual_next_step": "复核候选值，更新配置后运行完整回归与性能门禁；禁止脚本自动修改生产配置。",
    }
    write_json_file(PROJECT_ROOT / args.output, payload)
    print_json({key: value for key, value in payload.items() if key not in {"faq_direct", "intent_model"}})
    if args.fail_on_insufficient and not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
