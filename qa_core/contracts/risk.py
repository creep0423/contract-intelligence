"""可测试的合同确定性风险规则、统一评分和 RULE/LLM 去重。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Callable

from qa_core.config.settings import get_settings
from qa_core.contracts.schemas import (
    ContractExtractionResult,
    ContractObligation,
    ContractRisk,
    ObligationStatus,
    ObligationType,
    RiskLevel,
    RiskSource,
    RiskStatus,
)
from qa_core.contracts.time_rules import parse_date


RISK_LEVEL_SCORE = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3, RiskLevel.CRITICAL: 4}


@dataclass(frozen=True)
class RiskRuleContext:
    contract_id: str
    extraction: ContractExtractionResult
    obligations: list[ContractObligation]
    today: date


RiskRule = Callable[[RiskRuleContext], list[ContractRisk]]


def normalize_risk_level(severity: int, likelihood: int, urgency: int) -> RiskLevel:
    score = max(1, min(severity, 4)) + max(1, min(likelihood, 4)) + max(1, min(urgency, 4))
    if score >= 11:
        return RiskLevel.CRITICAL
    if score >= 8:
        return RiskLevel.HIGH
    if score >= 5:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _rule_risk(
    *,
    risk_type: str,
    name: str,
    description: str,
    reason: str,
    suggestion: str,
    severity: int,
    likelihood: int,
    urgency: int,
    obligation: ContractObligation | None = None,
) -> ContractRisk:
    return ContractRisk(
        risk_type=risk_type,
        risk_name=name,
        risk_level=normalize_risk_level(severity, likelihood, urgency),
        description=description,
        reason=reason,
        suggestion=suggestion,
        severity=severity,
        likelihood=likelihood,
        urgency=urgency,
        risk_source=RiskSource.RULE,
        source_evidence=obligation.source_evidence if obligation else None,
        related_obligation_id=obligation.id if obligation else None,
        confidence=1.0,
    )


def overdue_obligation_rule(context: RiskRuleContext) -> list[ContractRisk]:
    results: list[ContractRisk] = []
    for item in context.obligations:
        planned = parse_date(item.planned_date)
        if item.status in {ObligationStatus.COMPLETED, ObligationStatus.WAIVED, ObligationStatus.CANCELLED}:
            continue
        if planned is None:
            continue
        if item.status != ObligationStatus.OVERDUE and planned >= context.today:
            continue
        overdue_days = (context.today - planned).days
        if item.obligation_type in {ObligationType.PAYMENT, ObligationType.RECEIVABLE}:
            risk_type, name = "payment", "付款事项已逾期"
        elif item.obligation_type == ObligationType.DEPOSIT_REFUND:
            risk_type, name = "deposit", "保证金退还逾期"
        else:
            risk_type, name = "time", "履约节点已逾期"
        results.append(
            _rule_risk(
                risk_type=risk_type,
                name=name,
                description=f"“{item.title}”已逾期 {overdue_days} 天。",
                reason=item.time_rule.original_text or str(item.planned_date or "计划日期已过"),
                suggestion="立即核实实际完成情况、保留催告或履约证据并评估违约处置。",
                severity=4 if overdue_days >= 30 else 3,
                likelihood=4,
                urgency=4,
                obligation=item,
            )
        )
    return results


def upcoming_deadline_rule(context: RiskRuleContext) -> list[ContractRisk]:
    warning_days = get_settings().contract_upcoming_days
    results: list[ContractRisk] = []
    for item in context.obligations:
        if item.status in {ObligationStatus.COMPLETED, ObligationStatus.WAIVED, ObligationStatus.CANCELLED, ObligationStatus.OVERDUE}:
            continue
        planned = parse_date(item.planned_date)
        days = (planned - context.today).days if planned else -1
        if not 0 <= days <= warning_days:
            continue
        if item.obligation_type == ObligationType.WARRANTY:
            name, risk_type = "质保期即将结束", "warranty"
        elif item.obligation_type == ObligationType.RENEWAL:
            name, risk_type = "自动续约通知窗口即将错过", "termination"
        elif item.obligation_type == ObligationType.NOTICE:
            name, risk_type = "通知期限即将到期", "notice"
        elif item.obligation_type == ObligationType.PAYMENT:
            name, risk_type = "付款义务即将到期", "payment"
        else:
            name, risk_type = "履约节点即将到期", "time"
        results.append(
            _rule_risk(
                risk_type=risk_type,
                name=name,
                description=f"“{item.title}”将在 {days} 天内到期。",
                reason=item.time_rule.original_text or str(item.planned_date),
                suggestion="确认责任人、完成材料和内部审批，避免临期遗漏。",
                severity=2,
                likelihood=3,
                urgency=4 if days <= 2 else 3,
                obligation=item,
            )
        )
    return results


def contract_expiry_rule(context: RiskRuleContext) -> list[ContractRisk]:
    field = context.extraction.basic_info.termination_date
    expiry = parse_date(field.extracted_value)
    if expiry is None:
        return []
    days = (expiry - context.today).days
    warning_days = get_settings().contract_expiry_warning_days
    if not 0 <= days <= warning_days:
        return []
    return [
        ContractRisk(
            risk_type="contract_expiry",
            risk_name="合同即将到期",
            risk_level=RiskLevel.HIGH if days <= 7 else RiskLevel.MEDIUM,
            description=f"合同将在 {days} 天内到期。",
            reason=field.extracted_value or "",
            suggestion="确认续签、终止、交接和未结履约事项。",
            risk_source=RiskSource.RULE,
            source_evidence=field.source_evidence,
            severity=3,
            likelihood=4,
            urgency=4 if days <= 7 else 2,
            confidence=1.0,
        )
    ]


DETERMINISTIC_RISK_RULES: tuple[RiskRule, ...] = (
    overdue_obligation_rule,
    upcoming_deadline_rule,
    contract_expiry_rule,
)


def evaluate_deterministic_risks(
    contract_id: str,
    extraction: ContractExtractionResult,
    obligations: list[ContractObligation],
    *,
    today: date | None = None,
) -> list[ContractRisk]:
    context = RiskRuleContext(contract_id, extraction, obligations, today or date.today())
    return [risk for rule in DETERMINISTIC_RISK_RULES for risk in rule(context)]


def _normalized_risk_key(risk: ContractRisk) -> str:
    aliases = (
        (r"自动续约|续约通知", "auto_renewal"),
        (r"逾期.*付款|付款.*逾期", "payment_overdue"),
        (r"保证金.*逾期|逾期.*保证金", "deposit_overdue"),
        (r"验收.*主观|主观.*验收", "subjective_acceptance"),
        (r"责任.*不对等|不对等.*责任", "unbalanced_liability"),
    )
    text = f"{risk.risk_type}:{risk.risk_name}".lower()
    for pattern, key in aliases:
        if re.search(pattern, text):
            return key
    stable_type = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(risk.risk_type).lower())
    return stable_type or re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(risk.risk_name).lower())


RISK_STATUS_TRANSITIONS: dict[RiskStatus, set[RiskStatus]] = {
    RiskStatus.OPEN: {RiskStatus.OPEN, RiskStatus.ACKNOWLEDGED, RiskStatus.RESOLVED, RiskStatus.IGNORED},
    RiskStatus.ACKNOWLEDGED: {RiskStatus.ACKNOWLEDGED, RiskStatus.OPEN, RiskStatus.RESOLVED, RiskStatus.IGNORED},
    RiskStatus.RESOLVED: {RiskStatus.RESOLVED},
    RiskStatus.IGNORED: {RiskStatus.IGNORED},
}


def transition_risk_status(current: RiskStatus, target: RiskStatus) -> RiskStatus:
    if target not in RISK_STATUS_TRANSITIONS[current]:
        raise ValueError(f"不允许将风险状态从 {current.value} 修改为 {target.value}")
    return target


def risk_fingerprint(contract_id: str, risk: ContractRisk) -> tuple[str, str, str, str]:
    evidence_chunk = risk.source_evidence.chunk_id if risk.source_evidence else ""
    relation = risk.related_obligation_id
    if not relation and risk.related_obligation_index is not None:
        relation = str(risk.related_obligation_index)
    return (
        contract_id,
        relation or "",
        "" if relation else evidence_chunk or risk.risk_type,
        _normalized_risk_key(risk),
    )


def normalize_semantic_risk(risk: ContractRisk) -> ContractRisk:
    level = normalize_risk_level(risk.severity, risk.likelihood, risk.urgency)
    if risk.source_evidence is None and level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
        level = RiskLevel.MEDIUM
    return risk.model_copy(
        update={
            "risk_level": level,
            "risk_source": RiskSource.LLM,
            "needs_review": risk.needs_review or risk.source_evidence is None,
        }
    )


def deduplicate_risks(
    contract_id: str,
    risks: list[ContractRisk],
    obligations: list[ContractObligation] | None = None,
) -> list[ContractRisk]:
    merged: dict[tuple[str, str, str, str], ContractRisk] = {}
    for raw in risks:
        if (
            not raw.related_obligation_id
            and raw.related_obligation_index is not None
            and obligations
            and 0 <= raw.related_obligation_index < len(obligations)
        ):
            raw = raw.model_copy(update={"related_obligation_id": obligations[raw.related_obligation_index].id})
        risk = normalize_semantic_risk(raw) if raw.risk_source == RiskSource.LLM else raw
        key = risk_fingerprint(contract_id, risk)
        current = merged.get(key)
        if current is None:
            merged[key] = risk
            continue
        source = RiskSource.HYBRID if current.risk_source != risk.risk_source else current.risk_source
        higher = risk if RISK_LEVEL_SCORE[risk.risk_level] > RISK_LEVEL_SCORE[current.risk_level] else current
        evidence = higher.source_evidence or current.source_evidence or risk.source_evidence
        merged[key] = higher.model_copy(
            update={
                "risk_source": source,
                "source_evidence": evidence,
                "confidence": max(current.confidence, risk.confidence),
                "needs_review": current.needs_review and risk.needs_review,
            }
        )
    return sorted(merged.values(), key=lambda item: RISK_LEVEL_SCORE[item.risk_level], reverse=True)


def apply_obligation_risk_levels(
    obligations: list[ContractObligation],
    risks: list[ContractRisk],
) -> list[ContractObligation]:
    """把关联风险的最高等级同步到履约事项，Dashboard 与明细保持同一口径。"""
    levels: dict[str, RiskLevel] = {}
    for risk in risks:
        obligation_id = risk.related_obligation_id
        if not obligation_id:
            continue
        current = levels.get(obligation_id, RiskLevel.LOW)
        if RISK_LEVEL_SCORE[risk.risk_level] > RISK_LEVEL_SCORE[current]:
            levels[obligation_id] = risk.risk_level
    return [
        item.model_copy(update={"risk_level": levels.get(str(item.id), item.risk_level)})
        for item in obligations
    ]


def overall_risk_level(risks: list[dict] | list[ContractRisk]) -> str:
    levels: list[RiskLevel] = []
    for item in risks:
        raw = item.risk_level if isinstance(item, ContractRisk) else item.get("risk_level", "low")
        levels.append(RiskLevel(raw))
    return max(levels, key=lambda item: RISK_LEVEL_SCORE[item], default=RiskLevel.LOW).value
