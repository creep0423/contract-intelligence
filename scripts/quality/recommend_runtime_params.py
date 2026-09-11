"""把评测和阈值校准报告翻译成运行时参数建议。

`calibrate_thresholds.py` 只回答“候选阈值是多少”，本脚本继续回答“当前能不能采纳”。
它只读取 JSON 报告，不加载 QAService、Milvus、Embedding 或 LLM，适合作为发布前复核步骤。
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import PROJECT_ROOT, configure_utf8_stdio, print_json, read_json_file, utc_now, write_json_file


DEFAULT_CALIBRATION_REPORT = PROJECT_ROOT / "reports" / "threshold_calibration" / "threshold_candidate_latest.json"
DEFAULT_EVALUATION_REPORT = PROJECT_ROOT / "reports" / "evaluation" / "core_chain_latest.json"
DEFAULT_INTENT_POLICY_REPORT = PROJECT_ROOT / "reports" / "intent_policy" / "intent_policy_latest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "runtime_params" / "runtime_param_recommendation_latest.json"


def project_path(path: str | Path) -> Path:
    """把相对路径解析到项目根目录下。

    参数：
        path: 命令行传入的路径；绝对路径原样返回，相对路径按项目根目录拼接。

    返回：
        解析后的 Path 对象。

    调用顺序：load_optional_report() / main() -> project_path()。
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_optional_report(path: str | Path | None) -> dict[str, Any] | None:
    """读取可选 JSON 报告；路径为空或文件不存在时返回 None。

    参数：
        path: 报告路径；None、空值或不存在的文件都返回 None。

    返回：
        报告字典；无报告时返回 None。

    调用顺序：main() -> load_optional_report() -> project_path()。
    """
    if not path:
        return None
    report_path = project_path(path)
    if not report_path.exists():
        return None
    return read_json_file(report_path)


def _number(value: Any) -> float | None:
    """把报告里的数值字段转成 float；缺失或非法时返回 None。

    参数：
        value: 报告中的原始数值（str / int / float 或 None）。

    返回：
        浮点数；无法转换时返回 None。

    调用顺序：build_recommendation_report() -> _number()。
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _case_ids(rows: list[dict[str, Any]], limit: int = 5) -> list[str]:
    """提取少量样本 ID，避免建议报告太长。

    参数：
        rows: 评测行列表，优先取 case_id，退回 id。
        limit: 最多提取条数，默认 5。

    返回：
        样本 ID 字符串列表。

    调用顺序：build_core_chain_advice() -> _case_ids()。
    """
    result: list[str] = []
    for row in rows[:limit]:
        case_id = str(row.get("case_id") or row.get("id") or "")
        if case_id:
            result.append(case_id)
    return result


def _core_eval_blockers(evaluation_report: dict[str, Any] | None) -> list[str]:
    """主链路评测失败时，禁止直接采纳阈值候选。

    参数：
        evaluation_report: 主链路评测报告；缺失时按阻断处理。

    返回：
        阻断原因字符串列表；无阻断时为空列表。

    调用顺序：build_recommendation_report() -> _core_eval_blockers()。
    """
    if not evaluation_report:
        return ["missing_core_chain_evaluation_report"]
    errors = int(evaluation_report.get("errors") or 0)
    if errors > 0:
        return [f"core_chain_errors={errors}"]
    if evaluation_report.get("ok") is False:
        return ["core_chain_gate_failed"]
    return []


def _intent_policy_blockers(intent_policy_report: dict[str, Any] | None) -> list[str]:
    """意图策略评测失败时，禁止直接采纳阈值候选。

    参数：
        intent_policy_report: 意图策略评测报告；缺失时按阻断处理。

    返回：
        阻断原因字符串列表；无阻断时为空列表。

    调用顺序：build_recommendation_report() -> _intent_policy_blockers()。
    """
    if not intent_policy_report:
        return ["missing_intent_policy_report"]
    blockers: list[str] = []
    if intent_policy_report.get("ok") is False:
        blockers.append("intent_policy_gate_failed")
    critical_count = int(intent_policy_report.get("critical_failure_count") or 0)
    if critical_count > 0:
        blockers.append(f"intent_policy_critical_failures={critical_count}")
    return blockers


def _recommend_threshold(
    *,
    parameter: str,
    current: float | None,
    candidate: float | None,
    config_target: str,
    blockers: list[str],
    min_delta: float,
    reason: str,
) -> dict[str, Any]:
    """根据候选值和阻断原因生成单个阈值建议。（★★★ 核心）

    四分支 action 矩阵，按固定顺序依次检查：
    1. 当前值缺失 -> review_manually（无法计算差异，只能人工复核）；
    2. 候选值缺失 -> keep_current（校准报告缺数据，维持现状）；
    3. 存在阻断项 -> keep_current（评测失败/样本不足，禁止采纳）；
    4. 差异小于 min_delta -> keep_current（高置信度，变化量太小不值得改）；
    只有四道检查全部通过才落到 apply_candidate_after_review（唯一建议采纳的分支）。

    参数：
        parameter: 参数名（如 FAQ_DIRECT_SCORE_THRESHOLD）。
        current: 当前配置值；None 表示缺失。
        candidate: 校准出的候选值；None 表示缺失。
        config_target: 参数所在配置位置，便于人工定位修改点。
        blockers: 阻断原因列表；非空时禁止采纳候选。
        min_delta: 采纳候选所需的最小变化量。
        reason: 该参数校准来源与目标说明。

    返回：
        单条建议字典，含 action / current / candidate / recommended / confidence /
        config_target / reason。

    调用顺序：build_recommendation_report() -> _recommend_threshold()。
    """
    if current is None:
        # 原因：当前值缺失时无法计算差异，置信度无从谈起，只能交人工复核。
        return {
            "parameter": parameter,
            "action": "review_manually",
            "current": None,
            "candidate": candidate,
            "recommended": None,
            "confidence": "low",
            "config_target": config_target,
            "reason": "current_value_missing; " + reason,
        }
    if candidate is None:
        # 原因：没有候选值说明校准报告缺数据，维持现状最稳妥。
        return {
            "parameter": parameter,
            "action": "keep_current",
            "current": current,
            "candidate": None,
            "recommended": current,
            "confidence": "low",
            "config_target": config_target,
            "reason": "candidate_missing; " + reason,
        }
    if blockers:
        # 原因：存在阻断项（评测失败/样本不足/采集失败）时禁止采纳候选，保住线上稳定性。
        return {
            "parameter": parameter,
            "action": "keep_current",
            "current": current,
            "candidate": candidate,
            "recommended": current,
            "confidence": "low",
            "config_target": config_target,
            "reason": "; ".join(blockers) + "; " + reason,
        }
    if abs(candidate - current) < min_delta:
        # 原因：候选与当前值差异小于最小变化量，改动没有实质收益，保持现状并给高置信度。
        return {
            "parameter": parameter,
            "action": "keep_current",
            "current": current,
            "candidate": candidate,
            "recommended": current,
            "confidence": "high",
            "config_target": config_target,
            "reason": "candidate_close_to_current; " + reason,
        }
    # 原因：四道前置检查全部通过（值齐全、无阻断、差异足够），
    # 这是唯一输出 apply_candidate_after_review 的分支。
    return {
        "parameter": parameter,
        "action": "apply_candidate_after_review",
        "current": current,
        "candidate": candidate,
        "recommended": candidate,
        "confidence": "medium",
        "config_target": config_target,
        "reason": reason,
    }


def _faq_direct_blockers(
    calibration_report: dict[str, Any] | None,
    *,
    min_positive: int,
    min_negative: int,
) -> list[str]:
    """判断 FAQ 直出阈值候选是否具备采纳条件。

    检查候选本身约束是否满足、正负样本数量是否达标、采集是否出现失败。

    参数：
        calibration_report: 阈值校准报告；缺失时按阻断处理。
        min_positive: FAQ 直出正样本最低数量。
        min_negative: FAQ 直出负样本最低数量。

    返回：
        阻断原因字符串列表；无阻断时为空列表。

    调用顺序：build_recommendation_report() -> _faq_direct_blockers()。
    """
    if not calibration_report:
        return ["missing_threshold_calibration_report"]
    faq = dict(calibration_report.get("faq_direct") or {})
    sample_counts = dict(faq.get("sample_counts") or {})
    positive = int(sample_counts.get("positive") or 0)
    negative = int(sample_counts.get("negative") or 0)
    failures = int(sample_counts.get("collection_failures") or 0)
    blockers: list[str] = []
    if not faq.get("ok"):
        blockers.append("faq_threshold_constraints_failed")
    if positive < min_positive:
        blockers.append(f"positive_samples={positive}<required={min_positive}")
    if negative < min_negative:
        blockers.append(f"negative_samples={negative}<required={min_negative}")
    if failures > 0:
        blockers.append(f"collection_failures={failures}")
    return blockers


def _intent_model_blockers(
    calibration_report: dict[str, Any] | None,
    *,
    min_intent_cases: int,
) -> list[str]:
    """判断意图模型接管阈值候选是否具备采纳条件。

    参数：
        calibration_report: 阈值校准报告；缺失时按阻断处理。
        min_intent_cases: 意图模型校准样本最低数量。

    返回：
        阻断原因字符串列表；无阻断时为空列表。

    调用顺序：build_recommendation_report() -> _intent_model_blockers()。
    """
    if not calibration_report:
        return ["missing_threshold_calibration_report"]
    intent_model = dict(calibration_report.get("intent_model") or {})
    selected = dict(intent_model.get("selected") or {})
    total = int(selected.get("total") or 0)
    blockers: list[str] = []
    if not intent_model.get("ok"):
        blockers.append("intent_model_calibration_failed")
    if total < min_intent_cases:
        blockers.append(f"intent_cases={total}<required={min_intent_cases}")
    return blockers


def _most_common_plan_value(rows: list[dict[str, Any]], key: str, default: int) -> int:
    """从评测行里的 RetrievalPlan 估算当前常用参数值。

    取所有行中出现频率最高的整数值作为“当前常用值”，没有记录时退回默认值。

    参数：
        rows: 评测行列表，含 retrieval.plan 字段。
        key: 计划参数字段名（如 doc_top_k）。
        default: 无记录时的兜底默认值。

    返回：
        出现频率最高的整数值；无记录时返回 default。

    调用顺序：build_core_chain_advice() -> _most_common_plan_value()。
    """
    values: list[int] = []
    for row in rows:
        retrieval = dict(row.get("retrieval") or {})
        plan = dict(retrieval.get("plan") or {})
        value = plan.get(key)
        if isinstance(value, int):
            values.append(value)
    if not values:
        return default
    return Counter(values).most_common(1)[0][0]


def build_core_chain_advice(evaluation_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    """从主链路评测报告生成 top_k、上下文和直出阈值的方向性建议。

    识别三类证据：非 FAQ 直出被误直出（调高阈值）、应直出未直出（谨慎调低）、
    预期来源未召回（扩大候选池与最终上下文）。

    参数：
        evaluation_report: 主链路评测报告；缺失时返回空列表。

    返回：
        方向性建议列表，每条含参数名、动作、推荐值和证据样本。

    调用顺序：build_recommendation_report() -> build_core_chain_advice() ->
        _case_ids() / _most_common_plan_value()。
    """
    if not evaluation_report:
        return []
    rows = [row for row in evaluation_report.get("rows") or [] if isinstance(row, dict)]
    false_direct_rows = [
        row
        for row in rows
        if row.get("hit_type") == "faq_direct"
        and row.get("expected_hit_type")
        and row.get("expected_hit_type") != "faq_direct"
    ]
    missed_direct_rows = [
        row
        for row in rows
        if row.get("expected_hit_type") == "faq_direct" and row.get("hit_type") != "faq_direct"
    ]
    recall_miss_rows = [row for row in rows if row.get("source_recall_hit") is False]
    advice: list[dict[str, Any]] = []
    if false_direct_rows:
        advice.append(
            {
                "parameter": "FAQ_DIRECT_SCORE_THRESHOLD / retrieval_strategy.*_direct_threshold",
                "action": "raise_threshold_or_add_exact_guard",
                "recommended": "increase by 0.03-0.05, then rerun calibration",
                "confidence": "medium",
                "evidence_cases": _case_ids(false_direct_rows),
                "reason": "评测出现非 FAQ 直出问题被 FAQ 直出的样本，优先压低误直出风险。",
            }
        )
    if missed_direct_rows and not false_direct_rows:
        advice.append(
            {
                "parameter": "FAQ_DIRECT_SCORE_THRESHOLD",
                "action": "lower_threshold_carefully",
                "recommended": "decrease by 0.02-0.03 only if false_direct_rate remains within gate",
                "confidence": "low",
                "evidence_cases": _case_ids(missed_direct_rows),
                "reason": "评测出现应 FAQ 直出但没有直出的样本，且当前没有误直出样本。",
            }
        )
    if recall_miss_rows:
        doc_top_k = _most_common_plan_value(rows, "doc_top_k", 20)
        final_context_top_n = _most_common_plan_value(rows, "final_context_top_n", 4)
        advice.append(
            {
                "parameter": "DOC_TOP_K / FINAL_CONTEXT_TOP_N",
                "action": "increase_evidence_pool",
                "recommended": {
                    "DOC_TOP_K": max(doc_top_k + 4, round(doc_top_k * 1.2)),
                    "FINAL_CONTEXT_TOP_N": final_context_top_n + 1,
                },
                "confidence": "medium",
                "evidence_cases": _case_ids(recall_miss_rows),
                "reason": "评测出现预期来源未召回，先扩大文档候选池和最终上下文，再看延迟门禁。",
            }
        )
    return advice


def build_recommendation_report(
    *,
    calibration_report: dict[str, Any] | None,
    evaluation_report: dict[str, Any] | None,
    intent_policy_report: dict[str, Any] | None,
    min_faq_positive: int,
    min_faq_negative: int,
    min_intent_cases: int,
    min_delta: float,
) -> dict[str, Any]:
    """生成运行时参数建议报告。（★★★ 核心）

    汇总全局与分项阻断后，为每个候选阈值生成单条建议，再附上主链路方向性建议；
    只有 action=apply_candidate_after_review 的条目计入 safe_to_apply_count。

    参数：
        calibration_report: 阈值校准报告。
        evaluation_report: 主链路评测报告。
        intent_policy_report: 意图策略评测报告。
        min_faq_positive: FAQ 直出正样本最低数量。
        min_faq_negative: FAQ 直出负样本最低数量。
        min_intent_cases: 意图模型校准样本最低数量。
        min_delta: 采纳候选所需的最小变化量。

    返回：
        参数建议报告字典，含 global_blockers / recommendations / safe_to_apply_count 等。

    调用顺序：main() -> build_recommendation_report() -> _core_eval_blockers() /
        _intent_policy_blockers() / _faq_direct_blockers() / _intent_model_blockers() /
        _recommend_threshold() / build_core_chain_advice()。
    """
    calibration_report = calibration_report or {}
    current_policy = dict(calibration_report.get("current_policy") or {})
    candidate_policy = dict(calibration_report.get("candidate_policy") or {})
    global_blockers = _core_eval_blockers(evaluation_report) + _intent_policy_blockers(intent_policy_report)
    faq_blockers = global_blockers + _faq_direct_blockers(
        calibration_report,
        min_positive=min_faq_positive,
        min_negative=min_faq_negative,
    )
    intent_blockers = global_blockers + _intent_model_blockers(
        calibration_report,
        min_intent_cases=min_intent_cases,
    )
    recommendations = [
        _recommend_threshold(
            parameter="FAQ_DIRECT_SCORE_THRESHOLD",
            current=_number(current_policy.get("faq_direct_score_threshold")),
            candidate=_number(candidate_policy.get("faq_direct_score_threshold")),
            config_target=".env / qa_core.config.settings.Settings.faq_direct_score_threshold",
            blockers=faq_blockers,
            min_delta=min_delta,
            reason="来自 threshold_calibration.faq_direct，目标是控制 FAQ 误直出率。",
        ),
        _recommend_threshold(
            parameter="IntentDecisionPolicy.model_min_score",
            current=_number(current_policy.get("intent_model_min_score")),
            candidate=_number(candidate_policy.get("intent_model_min_score")),
            config_target="qa_core/intent/decision.py",
            blockers=intent_blockers,
            min_delta=min_delta,
            reason="来自 threshold_calibration.intent_model，目标是控制模型接管规则候选的最低置信度。",
        ),
    ]
    recommendations.extend(build_core_chain_advice(evaluation_report))
    actionable = [item for item in recommendations if item.get("action") == "apply_candidate_after_review"]
    return {
        "report_type": "runtime_parameter_recommendation",
        "created_at": utc_now(),
        "ok": not global_blockers,
        "safe_to_apply_count": len(actionable),
        "approval_required": True,
        "global_blockers": global_blockers,
        "recommendations": recommendations,
        "manual_next_step": (
            "只把 action=apply_candidate_after_review 的参数作为候选变更；"
            "修改配置后重新运行 evaluate_intent_policy、evaluate_core_chain 和对应 gate。"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。

    参数：
        无。

    返回：
        配置好各报告路径与最小样本/变化量阈值的 ArgumentParser。

    调用顺序：main() -> build_parser()。
    """
    parser = argparse.ArgumentParser(description="Recommend runtime parameter changes from evaluation reports.")
    parser.add_argument("--calibration-report", default=str(DEFAULT_CALIBRATION_REPORT.relative_to(PROJECT_ROOT)))
    parser.add_argument("--evaluation-report", default=str(DEFAULT_EVALUATION_REPORT.relative_to(PROJECT_ROOT)))
    parser.add_argument("--intent-policy-report", default=str(DEFAULT_INTENT_POLICY_REPORT.relative_to(PROJECT_ROOT)))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)))
    parser.add_argument("--min-faq-positive", type=int, default=4)
    parser.add_argument("--min-faq-negative", type=int, default=4)
    parser.add_argument("--min-intent-cases", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.01)
    return parser


def main() -> int:
    """命令行入口。

    读取三份可选报告，生成参数建议报告并写入 JSON，同时打印到 stdout。

    参数：
        无（参数由 build_parser() 解析）。

    返回：
        恒为 0；报告写盘与打印均在 main 内完成。

    调用顺序：命令行入口 -> main() -> build_parser() -> load_optional_report() ->
        build_recommendation_report() -> write_json_file() / print_json()。
    """
    configure_utf8_stdio()
    os.chdir(PROJECT_ROOT)
    args = build_parser().parse_args()
    payload = build_recommendation_report(
        calibration_report=load_optional_report(args.calibration_report),
        evaluation_report=load_optional_report(args.evaluation_report),
        intent_policy_report=load_optional_report(args.intent_policy_report),
        min_faq_positive=args.min_faq_positive,
        min_faq_negative=args.min_faq_negative,
        min_intent_cases=args.min_intent_cases,
        min_delta=args.min_delta,
    )
    write_json_file(project_path(args.output), payload)
    print_json(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
