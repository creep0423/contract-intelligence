# -*- coding: utf-8 -*-
"""意图策略校准评测。

该脚本只评测“意图分数和阈值是否驱动了正确业务行为”，不连接 Milvus，不调用 LLM。
它覆盖：

- direct route 是否提前收口；
- 检索类 intent / source / rewrite 是否符合预期；
- 规则候选分、模型网关分和 policy 是否符合预期；
- RetrievalPlan 是否触发正确的 FAQ 直出阈值、文档召回和保守保护；
- Prompt Profile 是否被路由到正确模板。

用法：
    python scripts/evaluate_intent_policy.py
    python scripts/evaluate_intent_policy.py --dataset eval_sets/intent_policy_cases.jsonl --fail-on-critical
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa_core.config.rules import get_rule_config
from qa_core.intent.classifier import classify_direct_intent, classify_intent
from qa_core.intent.decision import POLICY
from qa_core.pipeline.context import effective_source_filter
from qa_core.pipeline.query_input import normalize_user_query
from qa_core.prompts.selector import build_answer_prompt_profile
from qa_core.retrieval.strategy import build_retrieval_plan
from qa_core.scenarios.registry import get_scenario_registry
from scripts.common import PROJECT_ROOT, configure_utf8_stdio, print_json, utc_now, write_json_file
from scripts.eval_common import resolve_eval_dataset_path


DEFAULT_DATASET = PROJECT_ROOT / "eval_sets" / "intent_policy_cases.jsonl"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "intent_policy"


@dataclass(frozen=True)
class CheckResult:
    """记录单个意图策略断言的名称、结果和错误原因。

    调用顺序：业务模块 -> CheckResult。
    """
    name: str
    ok: bool
    expected: Any
    actual: Any

    def as_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的诊断数据。

        调用顺序：测试或业务入口 -> CheckResult.as_dict()。
        """
        return {
            "name": self.name,
            "ok": self.ok,
            "expected": self.expected,
            "actual": self.actual,
        }


def load_policy_cases(dataset: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 或 JSON list 格式的意图策略评测集。"""

    path = resolve_eval_dataset_path(dataset)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        cases: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            compact = line.strip()
            if not compact or compact.startswith("#"):
                continue
            payload = json.loads(compact)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON object")
            cases.append(payload)
        return cases
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError(f"意图策略评测集必须是 JSON list 或 JSONL：{path}")
    return [dict(item) for item in payload]


def _history_messages(case: dict[str, Any]) -> list[BaseMessage]:
    """把评测样本中的 history 转成 LangChain message 列表。"""

    messages: list[BaseMessage] = []
    for item in case.get("history") or []:
        if isinstance(item, str):
            messages.append(HumanMessage(content=item))
            continue
        role = str(item.get("role") or "user").lower()
        content = str(item.get("content") or "")
        if role in {"assistant", "ai"}:
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


def _expected_options(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _match_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) < 1e-6
        except (TypeError, ValueError):
            return False
    return actual == expected


def _add_expected_check(checks: list[CheckResult], name: str, actual: Any, expected: Any) -> None:
    if expected is None:
        return
    options = _expected_options(expected)
    checks.append(CheckResult(name=name, ok=any(_match_value(actual, option) for option in options), expected=expected, actual=actual))


def _add_min_check(checks: list[CheckResult], name: str, actual: Any, expected_min: Any) -> None:
    if expected_min is None:
        return
    actual_float = float(actual or 0.0)
    checks.append(CheckResult(name=name, ok=actual_float >= float(expected_min), expected=f">= {expected_min}", actual=actual_float))


def _add_max_check(checks: list[CheckResult], name: str, actual: Any, expected_max: Any) -> None:
    if expected_max is None:
        return
    actual_float = float(actual or 0.0)
    checks.append(CheckResult(name=name, ok=actual_float <= float(expected_max), expected=f"<= {expected_max}", actual=actual_float))


def _add_contains_checks(checks: list[CheckResult], name: str, actual_values: list[Any] | tuple[Any, ...], expected_values: Any) -> None:
    if not expected_values:
        return
    actual_set = {str(value) for value in actual_values}
    expected_list = [str(value) for value in expected_values]
    missing = [value for value in expected_list if value not in actual_set]
    checks.append(CheckResult(name=name, ok=not missing, expected=expected_list, actual=list(actual_set)))


def _check_plan_fields(plan_payload: dict[str, Any], expected: dict[str, Any] | None) -> list[CheckResult]:
    checks: list[CheckResult] = []
    if not expected:
        return checks
    for key, expected_value in expected.items():
        checks.append(
            CheckResult(
                name=f"plan.{key}",
                ok=_match_value(plan_payload.get(key), expected_value),
                expected=expected_value,
                actual=plan_payload.get(key),
            )
        )
    return checks


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """执行单条意图策略样本，返回明细结果。"""

    scenario = get_scenario_registry().resolve(str(case.get("scenario_id") or "enterprise_knowledge"))
    raw_query = str(case.get("question") or case.get("query") or "").strip()
    query = normalize_user_query(raw_query)
    history = _history_messages(case)
    expected_route = str(case.get("expected_route") or "retrieval")
    checks: list[CheckResult] = []

    direct_intent = classify_direct_intent(query, scenario)
    if direct_intent is not None:
        route = "direct_answer"
        intent_payload = direct_intent.as_dict()
        plan_payload: dict[str, Any] = {}
        prompt_profile = ""
        source_filter = None
        question_category = ""
    else:
        route = "retrieval"
        intent = classify_intent(query, history, scenario)
        source_filter = effective_source_filter(case.get("source_filter"), intent.suggested_source, scenario)
        plan_query = str(case.get("plan_query") or query)
        plan = build_retrieval_plan(plan_query, intent)
        profile = build_answer_prompt_profile(intent.intent, scenario, plan_query)
        intent_payload = intent.as_dict()
        plan_payload = plan.as_dict()
        prompt_profile = profile.name
        question_category = plan.question_category

    _add_expected_check(checks, "route", route, case.get("expected_route"))
    _add_expected_check(checks, "intent", intent_payload.get("intent"), case.get("expected_intent"))
    _add_expected_check(checks, "reason", intent_payload.get("reason"), case.get("expected_reason"))
    _add_expected_check(checks, "effective_source", source_filter, case.get("expected_source"))
    _add_expected_check(checks, "requires_rewrite", intent_payload.get("requires_rewrite"), case.get("expected_requires_rewrite"))
    _add_expected_check(checks, "decision_policy", intent_payload.get("decision_policy"), case.get("expected_decision_policy"))
    _add_expected_check(checks, "rule_score", intent_payload.get("rule_score"), case.get("expected_rule_score"))
    _add_expected_check(checks, "question_category", question_category, case.get("expected_question_category"))
    _add_expected_check(checks, "prompt_profile", prompt_profile, case.get("expected_prompt_profile"))
    _add_min_check(checks, "confidence_min", intent_payload.get("confidence"), case.get("expected_confidence_min"))
    _add_max_check(checks, "confidence_max", intent_payload.get("confidence"), case.get("expected_confidence_max"))
    _add_contains_checks(checks, "risk_tags", intent_payload.get("risk_tags") or [], case.get("expected_risk_tags_contains"))
    checks.extend(_check_plan_fields(plan_payload, case.get("expected_plan_contains")))

    failed_checks = [check for check in checks if not check.ok]
    return {
        "case_id": case.get("case_id") or raw_query,
        "critical": bool(case.get("critical", False)),
        "ok": not failed_checks,
        "query": raw_query,
        "normalized_query": query,
        "route": route,
        "intent": intent_payload,
        "effective_source": source_filter,
        "question_category": question_category,
        "prompt_profile": prompt_profile,
        "plan": plan_payload,
        "checks": [check.as_dict() for check in checks],
        "failed_checks": [check.as_dict() for check in failed_checks],
    }


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _metric_from_check(rows: list[dict[str, Any]], check_name: str) -> float:
    selected = []
    for row in rows:
        for check in row["checks"]:
            if check["name"] == check_name:
                selected.append(bool(check["ok"]))
    return _ratio(sum(1 for item in selected if item), len(selected))


def _confidence_band(value: float, low_threshold: float, medium_threshold: float) -> str:
    if value < low_threshold:
        return "low"
    if value < medium_threshold:
        return "medium"
    return "high"


def _group_pass_rates(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field) or "unknown")].append(row)
    return {
        key: {
            "total": len(items),
            "passed": sum(1 for item in items if item["ok"]),
            "pass_rate": _ratio(sum(1 for item in items if item["ok"]), len(items)),
        }
        for key, items in sorted(grouped.items())
    }


def build_report(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """运行全部样本并生成校准报告。"""

    rows = [evaluate_case(case) for case in cases]
    rules = get_rule_config()
    low_threshold = rules.retrieval_strategy.low_rule_score_threshold
    medium_threshold = rules.retrieval_strategy.medium_rule_score_threshold
    for row in rows:
        confidence = float((row.get("intent") or {}).get("confidence") or 0.0)
        row["confidence_band"] = _confidence_band(confidence, low_threshold, medium_threshold)

    total = len(rows)
    passed = sum(1 for row in rows if row["ok"])
    critical_failures = [row for row in rows if row["critical"] and not row["ok"]]
    policy_counter = Counter(str((row["intent"] or {}).get("decision_policy") or "none") for row in rows)
    intent_counter = Counter(str((row["intent"] or {}).get("intent") or "none") for row in rows)
    reason_counter = Counter(str((row["intent"] or {}).get("reason") or "none") for row in rows)

    return {
        "report_type": "intent_policy_calibration",
        "created_at": utc_now(),
        "ok": not critical_failures,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": _ratio(passed, total),
        "critical_failure_count": len(critical_failures),
        "metrics": {
            "route_accuracy": _metric_from_check(rows, "route"),
            "intent_accuracy": _metric_from_check(rows, "intent"),
            "source_accuracy": _metric_from_check(rows, "effective_source"),
            "rewrite_accuracy": _metric_from_check(rows, "requires_rewrite"),
            "policy_accuracy": _metric_from_check(rows, "decision_policy"),
            "rule_score_accuracy": _metric_from_check(rows, "rule_score"),
            "question_category_accuracy": _metric_from_check(rows, "question_category"),
            "prompt_profile_accuracy": _metric_from_check(rows, "prompt_profile"),
        },
        "distributions": {
            "intent": dict(intent_counter),
            "decision_policy": dict(policy_counter),
            "reason": dict(reason_counter),
        },
        "confidence_bands": _group_pass_rates(rows, "confidence_band"),
        "threshold_snapshot": {
            "intent_rule_scores": rules.intent_rule_scores.__dict__,
            "retrieval_strategy": rules.retrieval_strategy.__dict__,
            "decision_policy": POLICY.__dict__,
        },
        "calibration_guidance": [
            "如果低分样本没有触发 faq_direct_exact_only，应提高 low_rule_score_threshold 或检查 default_knowledge 分数。",
            "如果 FAQ 样本误进入 KNOWLEDGE_QUERY，应检查 FAQ 关键词、source_patterns 或 model_min_score。",
            "如果规则和模型冲突样本过多，应扩充 BERT 训练集，并保持 conflict_final_score 偏保守。",
            "如果高风险 pricing/compliance 样本未提高 FAQ 直出阈值，应优先调整 retrieval_strategy 中的风险阈值。",
        ],
        "critical_failures": [
            {
                "case_id": row["case_id"],
                "failed_checks": row["failed_checks"],
            }
            for row in critical_failures
        ],
        "details": rows,
    }


def default_output_path() -> Path:
    """生成带时间戳的意图策略评测报告路径。

    调用顺序：业务模块或命令行入口 -> default_output_path()。
    """
    timestamp = utc_now().replace(":", "").replace("-", "").split(".")[0]
    return DEFAULT_REPORT_DIR / f"{timestamp}_intent_policy_calibration.json"


def parse_args() -> argparse.Namespace:
    """解析当前命令行工具的运行参数。

    调用顺序：业务模块或命令行入口 -> parse_args()。
    """
    parser = argparse.ArgumentParser(description="评测意图分数、网关策略和检索计划阈值是否形成闭环。")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="JSONL 或 JSON list 评测集路径。")
    parser.add_argument("--output", default="", help="报告输出路径，默认写入 reports/intent_policy。")
    parser.add_argument("--limit", type=int, default=0, help="只评测前 N 条，0 表示全部。")
    parser.add_argument("--fail-on-critical", action="store_true", help="存在 critical 样本失败时返回非 0。")
    return parser.parse_args()


def main() -> None:
    """执行当前脚本的完整命令行流程。

    调用顺序：业务模块或命令行入口 -> main()。
    """
    configure_utf8_stdio()
    args = parse_args()
    cases = load_policy_cases(args.dataset)
    if args.limit and args.limit > 0:
        cases = cases[: args.limit]
    report = build_report(cases)
    output_path = Path(args.output) if args.output else default_output_path()
    report["output_path"] = write_json_file(output_path, report)
    print_json(
        {
            "ok": report["ok"],
            "total": report["total"],
            "pass_rate": report["pass_rate"],
            "critical_failure_count": report["critical_failure_count"],
            "output_path": report["output_path"],
            "metrics": report["metrics"],
        }
    )
    if args.fail_on_critical and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
