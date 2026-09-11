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
    python scripts/intent/evaluate_intent_policy.py
    python scripts/intent/evaluate_intent_policy.py --dataset eval_sets/intent_policy_cases.jsonl --fail-on-critical
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    """读取 JSONL 或 JSON list 格式的意图策略评测集。

    支持两种格式：
    - .jsonl：每行一个 JSON 对象，跳过空行和 # 开头的注释行。
    - .json：顶层必须是 JSON 对象列表。

    参数：
        dataset: 评测集文件路径（字符串或 Path 对象）。

    返回：
        解析后的评测样本字典列表。

    异常：
        ValueError: 格式不符合预期（JSONL 行不是对象、JSON 顶层不是列表）。

    调用顺序：命令行入口 -> load_policy_cases()。
    """

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
    """把评测样本中的 history 转成 LangChain message 列表。

    支持的 history 条目格式：
    - 字符串：直接作为 HumanMessage。
    - 字典：通过 role（user/assistant/ai）和 content 字段区分消息类型。

    参数：
        case: 评测样本字典，history 字段为消息列表。

    返回：
        LangChain BaseMessage 列表。

    调用顺序：evaluate_case() -> _history_messages()。
    """

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
    """将期望值统一为列表形式，支持单个值和列表两种写法。

    评测用例中 expected_intent 可以写 "FAQ_QUERY" 或 ["FAQ_QUERY", "KNOWLEDGE_QUERY"]，
    前者表示精确匹配，后者表示命中任一即可。

    参数：
        value: 期望值（单值或列表）。

    返回：
        包装后的列表。

    调用顺序：_add_expected_check() -> _expected_options()。
    """
    if isinstance(value, list):
        return value
    return [value]


def _match_value(actual: Any, expected: Any) -> bool:
    """比较实际值与期望值是否匹配。

    浮点数使用 1e-6 容差比较，其他类型使用 == 精确比较。

    参数：
        actual: 实际值。
        expected: 期望值。

    返回：
        是否匹配。

    调用顺序：_add_expected_check() -> _match_value()。
    """
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) < 1e-6
        except (TypeError, ValueError):
            return False
    return actual == expected


def _add_expected_check(checks: list[CheckResult], name: str, actual: Any, expected: Any) -> None:
    """添加一条精确匹配断言到 checks 列表。

    如果 expected 为 None 则跳过（表示该维度不检查）。
    如果 expected 是列表，命中任一值即通过。

    参数：
        checks: 断言结果列表（原地追加）。
        name: 断言名称（如 "intent" / "route"）。
        actual: 实际值。
        expected: 期望值（单值、列表或 None）。

    调用顺序：evaluate_case() -> _add_expected_check()。
    """
    if expected is None:
        return
    options = _expected_options(expected)
    checks.append(CheckResult(name=name, ok=any(_match_value(actual, option) for option in options), expected=expected, actual=actual))


def _add_min_check(checks: list[CheckResult], name: str, actual: Any, expected_min: Any) -> None:
    """添加一条最小值断言（actual >= expected_min）。

    用于 confidence 等连续值的下限检查。expected_min 为 None 时跳过。

    参数：
        checks: 断言结果列表（原地追加）。
        name: 断言名称（如 "confidence_min"）。
        actual: 实际值。
        expected_min: 期望的最小值。

    调用顺序：evaluate_case() -> _add_min_check()。
    """
    if expected_min is None:
        return
    actual_float = float(actual or 0.0)
    checks.append(CheckResult(name=name, ok=actual_float >= float(expected_min), expected=f">= {expected_min}", actual=actual_float))


def _add_max_check(checks: list[CheckResult], name: str, actual: Any, expected_max: Any) -> None:
    """添加一条最大值断言（actual <= expected_max）。

    用于 confidence 等连续值的上限检查。expected_max 为 None 时跳过。

    参数：
        checks: 断言结果列表（原地追加）。
        name: 断言名称（如 "confidence_max"）。
        actual: 实际值。
        expected_max: 期望的最大值。

    调用顺序：evaluate_case() -> _add_max_check()。
    """
    if expected_max is None:
        return
    actual_float = float(actual or 0.0)
    checks.append(CheckResult(name=name, ok=actual_float <= float(expected_max), expected=f"<= {expected_max}", actual=actual_float))


def _add_contains_checks(checks: list[CheckResult], name: str, actual_values: list[Any] | tuple[Any, ...], expected_values: Any) -> None:
    """添加一条集合包含断言（expected ⊆ actual）。

    用于 risk_tags 等集合字段检查：实际标签必须包含所有期望标签。
    expected_values 为空时跳过。

    参数：
        checks: 断言结果列表（原地追加）。
        name: 断言名称（如 "risk_tags"）。
        actual_values: 实际值集合。
        expected_values: 期望包含的值列表。

    调用顺序：evaluate_case() -> _add_contains_checks()。
    """
    if not expected_values:
        return
    actual_set = {str(value) for value in actual_values}
    expected_list = [str(value) for value in expected_values]
    missing = [value for value in expected_list if value not in actual_set]
    checks.append(CheckResult(name=name, ok=not missing, expected=expected_list, actual=list(actual_set)))


def _check_plan_fields(plan_payload: dict[str, Any], expected: dict[str, Any] | None) -> list[CheckResult]:
    """逐字段检查检索计划是否符合预期值。

    对 expected 中的每个 key，使用 _match_value 比较 plan_payload 对应值。

    参数：
        plan_payload: build_retrieval_plan() 返回的字典。
        expected: 期望的字段值字典（key 为 plan 字段名如 "faq_direct_exact_only"）。

    返回：
        CheckResult 列表。

    调用顺序：evaluate_case() -> _check_plan_fields()。
    """
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
    """执行单条意图策略样本，返回明细结果。

    评测流程分两条路径：
    - direct_answer 路径：classify_direct_intent 提前收口（问候/越界/转人工），
      此时无检索计划和 prompt profile。
    - retrieval 路径：经 classify_intent -> build_retrieval_plan ->
      build_answer_prompt_profile 完整链路，检查所有维度。

    检查维度（根据评测用例中提供的 expected_* 字段选择性检查）：
    - route / intent / reason / effective_source / requires_rewrite
    - decision_policy / rule_score / question_category / prompt_profile
    - confidence_min / confidence_max / risk_tags_contains
    - plan 字段（faq_direct_exact_only 等）

    参数：
        case: 评测样本字典，需包含 question/query 字段，
              可选 expected_* 系列断言字段。

    返回：
        包含 case_id / ok / checks / failed_checks 等字段的评测结果字典。

    调用顺序：build_report() -> evaluate_case()。
    """

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
    """计算百分比比例，保留 4 位小数。分母为 0 时返回 0.0。

    调用顺序：_metric_from_check() / _group_pass_rates() / build_report() -> _ratio()。
    """
    return round(numerator / denominator, 4) if denominator else 0.0


def _metric_from_check(rows: list[dict[str, Any]], check_name: str) -> float:
    """计算指定断言名称的通过率。

    遍历所有样本的 checks 列表，收集所有同名的 CheckResult，
    返回 passed / total 的比例。

    参数：
        rows: evaluate_case 返回的评测结果列表。
        check_name: 断言名称（如 "intent" / "route"）。

    返回：
        通过率（0.0 ~ 1.0）。

    调用顺序：build_report() -> _metric_from_check()。
    """
    selected = []
    for row in rows:
        for check in row["checks"]:
            if check["name"] == check_name:
                selected.append(bool(check["ok"]))
    return _ratio(sum(1 for item in selected if item), len(selected))


def _confidence_band(value: float, low_threshold: float, medium_threshold: float) -> str:
    """将 confidence 分数映射到 low / medium / high 三档。

    阈值来自 retrieval_strategy 规则的 low_rule_score_threshold
    和 medium_rule_score_threshold，用于分析不同分数段样本的通过率分布。

    参数：
        value: confidence 分数（0.0 ~ 1.0）。
        low_threshold: 低分段上限。
        medium_threshold: 中分段上限。

    返回：
        "low" / "medium" / "high" 档位标签。

    调用顺序：build_report() -> _confidence_band()。
    """
    if value < low_threshold:
        return "low"
    if value < medium_threshold:
        return "medium"
    return "high"


def _group_pass_rates(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    """按指定字段分组计算各组的通过率。

    用于分析不同 confidence_band / intent / reason 下的通过率分布，
    帮助发现特定条件下策略失效的模式。

    参数：
        rows: 评测结果列表。
        field: 分组字段名（如 "confidence_band" / "intent"）。

    返回：
        {分组值: {total, passed, pass_rate}} 的字典，按分组值排序。

    调用顺序：build_report() -> _group_pass_rates()。
    """
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
    """运行全部样本并生成校准报告。

    报告包含以下维度：
    - 整体统计：total / passed / failed / pass_rate / critical_failure_count。
    - 各维度通过率：route / intent / source / rewrite / policy / rule_score /
      question_category / prompt_profile。
    - 分布统计：intent / decision_policy / reason 的频次分布。
    - 分数段分析：按 confidence_band（low/medium/high）分组通过率。
    - 阈值快照：当前使用的 intent_rule_scores / retrieval_strategy / decision_policy 配置。
    - 校准建议：基于常见问题的自动诊断建议。
    - 严重失败明细：critical=True 且未通过的样本。

    参数：
        cases: 评测样本列表。

    返回：
        完整的校准报告字典。

    调用顺序：命令行入口 -> build_report()。
    """

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
