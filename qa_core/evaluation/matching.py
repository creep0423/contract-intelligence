"""不依赖 LLM 的合同 Benchmark matcher 与规范化函数。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Iterable

from qa_core.evaluation.schemas import (
    BenchmarkCase,
    GoldField,
    GoldObligation,
    GoldRisk,
    GoldSourceReference,
    MatchType,
    PredictedEvidence,
    PredictedField,
    PredictedObligation,
    PredictedRisk,
)


_PUNCTUATION = "，。；：、！？（）()【】[]《》<>‘’“”\"'`·"
_CURRENCY = {"人民币": "CNY", "rmb": "CNY", "cny": "CNY", "￥": "CNY", "¥": "CNY", "美元": "USD", "usd": "USD", "$": "USD"}


@dataclass(frozen=True)
class PairMatch:
    gold_index: int
    predicted_index: int
    score: float


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = re.sub(r"\s+", "", text)
    return text.translate(str.maketrans("", "", _PUNCTUATION))


def normalize_number(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    text = unicodedata.normalize("NFKC", str(value)).replace(",", "").replace(" ", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        number = Decimal(match.group())
    except InvalidOperation:
        return None
    if "亿" in text:
        number *= Decimal("100000000")
    elif "万" in text:
        number *= Decimal("10000")
    return number.normalize()


def normalize_percentage(value: Any) -> Decimal | None:
    number = normalize_number(value)
    if number is None:
        return None
    text = str(value)
    if "%" in text or "百分之" in text:
        number /= Decimal("100")
    elif abs(number) > 1:
        number /= Decimal("100")
    return number.normalize()


def normalize_date(value: Any) -> str | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    match = re.search(r"(\d{4})[年\-/\.](\d{1,2})[月\-/\.](\d{1,2})日?", text)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
    except ValueError:
        return None


def normalize_currency(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    for token, currency in _CURRENCY.items():
        if token in text:
            return currency
    return text.upper()


def normalize_set(value: Any) -> frozenset[str]:
    values = value if isinstance(value, list) else re.split(r"[,，;；、|]", str(value or ""))
    return frozenset(item for raw in values if (item := normalize_text(raw)))


def values_match(expected: Any, predicted: Any, match_type: MatchType) -> tuple[bool, bool]:
    """返回 (exact_match, normalized_match)。"""
    exact = str(expected).strip() == str(predicted).strip()
    if match_type == MatchType.EXACT:
        return exact, exact
    if match_type == MatchType.NUMERIC:
        return exact, normalize_number(expected) is not None and normalize_number(expected) == normalize_number(predicted)
    if match_type == MatchType.DATE:
        return exact, normalize_date(expected) is not None and normalize_date(expected) == normalize_date(predicted)
    if match_type == MatchType.PERCENTAGE:
        return exact, normalize_percentage(expected) is not None and normalize_percentage(expected) == normalize_percentage(predicted)
    if match_type == MatchType.CURRENCY:
        return exact, normalize_currency(expected) == normalize_currency(predicted)
    if match_type == MatchType.SET:
        return exact, normalize_set(expected) == normalize_set(predicted)
    return exact, normalize_text(expected) == normalize_text(predicted)


def match_field(gold: GoldField, predicted: PredictedField | None) -> tuple[bool, bool]:
    if predicted is None or predicted.value is None or str(predicted.value).strip() == "":
        missing = True
    else:
        missing = False
    if gold.status.value in {"NOT_PRESENT", "NOT_APPLICABLE"}:
        return missing, missing
    if gold.status.value == "AMBIGUOUS":
        return True, True
    if missing:
        return False, False
    return values_match(gold.value, predicted.value, gold.match_type)


def evidence_matches_gold(predicted: PredictedEvidence, gold: GoldSourceReference) -> bool:
    if gold.document_id:
        if normalize_text(predicted.document_id) != normalize_text(gold.document_id):
            return False
    elif normalize_text(predicted.document_name) != normalize_text(gold.document_name):
        return False
    if gold.document_version and normalize_text(predicted.document_version) != normalize_text(gold.document_version):
        return False
    if gold.page is not None and predicted.page != gold.page:
        return False
    predicted_clause = predicted.clause_no or predicted.clause
    gold_clause = gold.clause_no or gold.clause
    if gold_clause and normalize_text(predicted_clause) != normalize_text(gold_clause):
        return False
    gold_quote = gold.text_snippet or gold.quote
    if gold_quote and normalize_text(gold_quote) not in normalize_text(predicted.quote):
        return False
    return True


def any_evidence_match(
    evidence: Iterable[PredictedEvidence], reference_ids: Iterable[str], source_map: dict[str, GoldSourceReference]
) -> bool:
    references = [source_map[item] for item in reference_ids if item in source_map]
    return any(evidence_matches_gold(predicted, gold) for predicted in evidence for gold in references)


def time_rule_component_matches(gold: Any, predicted: Any) -> dict[str, bool]:
    fields = ("rule_type", "base_event_type", "offset_value", "offset_unit", "business_day", "direction", "fixed_date", "resolved_date")
    result: dict[str, bool] = {}
    if gold is None:
        return result
    if getattr(gold, "ambiguous", False):
        result["ambiguous"] = predicted is None or getattr(predicted, "ambiguous", False) or getattr(predicted, "rule_type", None) == "no_explicit_date"
        return result
    for field in fields:
        expected = getattr(gold, field, None)
        if expected is None:
            continue
        actual = getattr(predicted, field, None) if predicted is not None else None
        if field in {"fixed_date", "resolved_date"}:
            result[field] = normalize_date(expected) == normalize_date(actual)
        else:
            result[field] = expected == actual
    return result


def _obligation_score(gold: GoldObligation, predicted: PredictedObligation, source_map: dict[str, GoldSourceReference]) -> float:
    if normalize_text(gold.obligation_type) != normalize_text(predicted.obligation_type):
        return 0.0
    weights: list[tuple[float, bool]] = [(0.35, True)]
    if gold.responsible_party:
        weights.append((0.15, normalize_text(gold.responsible_party) == normalize_text(predicted.responsible_party)))
    if gold.beneficiary_party:
        weights.append((0.05, normalize_text(gold.beneficiary_party) == normalize_text(predicted.beneficiary_party)))
    if gold.trigger_type:
        weights.append((0.10, normalize_text(gold.trigger_type) == normalize_text(predicted.trigger_type)))
    if gold.percentage is not None:
        weights.append((0.10, normalize_percentage(gold.percentage) == normalize_percentage(predicted.percentage)))
    if gold.amount is not None:
        weights.append((0.10, normalize_number(gold.amount) == normalize_number(predicted.amount)))
    components = time_rule_component_matches(gold.time_rule, predicted.time_rule)
    if components:
        weights.append((0.10, all(components.values())))
    if gold.source_references:
        weights.append((0.15, any_evidence_match(predicted.evidence, gold.source_references, source_map)))
    total = sum(weight for weight, _ in weights)
    return sum(weight for weight, matched in weights if matched) / total if total else 0.0


def _risk_score(gold: GoldRisk, predicted: PredictedRisk, source_map: dict[str, GoldSourceReference]) -> float:
    accepted_types = {normalize_text(gold.risk_type), *(normalize_text(item) for item in gold.acceptable_variants)}
    if normalize_text(predicted.risk_type) not in accepted_types:
        return 0.0
    score = 0.55
    denominator = 0.55
    if gold.related_obligation_gold_id:
        denominator += 0.15
        score += 0.15 if gold.related_obligation_gold_id == predicted.related_obligation_gold_id else 0
    if gold.normalized_business_key:
        denominator += 0.10
        score += 0.10 if normalize_text(gold.normalized_business_key) == normalize_text(predicted.normalized_business_key) else 0
    if gold.source_references:
        denominator += 0.20
        score += 0.20 if any_evidence_match(predicted.evidence, gold.source_references, source_map) else 0
    return score / denominator


def greedy_match(
    gold_items: list[Any], predicted_items: list[Any], scorer: Callable[[Any, Any], float], threshold: float
) -> tuple[list[PairMatch], list[int], list[int]]:
    candidates = sorted(
        (
            PairMatch(gold_index, predicted_index, scorer(gold, predicted))
            for gold_index, gold in enumerate(gold_items)
            for predicted_index, predicted in enumerate(predicted_items)
        ),
        key=lambda item: (-item.score, item.gold_index, item.predicted_index),
    )
    matches: list[PairMatch] = []
    used_gold: set[int] = set()
    used_predicted: set[int] = set()
    for candidate in candidates:
        if candidate.score < threshold or candidate.gold_index in used_gold or candidate.predicted_index in used_predicted:
            continue
        matches.append(candidate)
        used_gold.add(candidate.gold_index)
        used_predicted.add(candidate.predicted_index)
    return matches, sorted(set(range(len(gold_items))) - used_gold), sorted(set(range(len(predicted_items))) - used_predicted)


def match_obligations(case: BenchmarkCase, predicted: list[PredictedObligation]):
    source_map = {item.source_id: item for item in case.sources}
    gold = [item for item in case.gold_obligations if item.required]
    return greedy_match(gold, predicted, lambda left, right: _obligation_score(left, right, source_map), 0.65), gold


def match_risks(case: BenchmarkCase, predicted: list[PredictedRisk]):
    source_map = {item.source_id: item for item in case.sources}
    gold = [item for item in case.gold_risks if item.required]
    return greedy_match(gold, predicted, lambda left, right: _risk_score(left, right, source_map), 0.70), gold
