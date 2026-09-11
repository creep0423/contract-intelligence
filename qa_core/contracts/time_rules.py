"""合同时间规则解析与日期计算；不调用 LLM。"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta
from typing import Mapping

from qa_core.contracts.schemas import (
    ContractEventType,
    ContractObligation,
    ObligationStatus,
    TimeDirection,
    TimeOffsetUnit,
    TimeRule,
    TimeRuleType,
    TriggerType,
)


DATE_PATTERNS = (
    re.compile(r"(?P<year>20\d{2})\s*[年\-/\.]\s*(?P<month>\d{1,2})\s*[月\-/\.]\s*(?P<day>\d{1,2})\s*日?"),
    re.compile(r"(?P<year>20\d{2})(?P<month>\d{2})(?P<day>\d{2})"),
)
OFFSET_PATTERN = re.compile(
    r"(?P<offset>\d+)\s*(?:个)?(?P<business>工作)?(?P<unit>日|天|月|年)(?:内|前)?"
)
EVENT_KEYWORDS: tuple[tuple[str, ContractEventType], ...] = (
    ("验收合格|验收通过|验收后|验收完成", ContractEventType.ACCEPTANCE_PASSED),
    ("验收不合格|验收失败", ContractEventType.ACCEPTANCE_FAILED),
    ("开始验收|验收开始", ContractEventType.ACCEPTANCE_STARTED),
    ("交付完成|完成交付|交付后|交货后", ContractEventType.DELIVERY_COMPLETED),
    ("收到发票|发票收到|收票", ContractEventType.INVOICE_RECEIVED),
    ("里程碑完成", ContractEventType.MILESTONE_COMPLETED),
    ("通知送达|发出通知|书面通知", ContractEventType.NOTICE_SENT),
    ("质保期结束|质保结束", ContractEventType.WARRANTY_ENDED),
)


def parse_date(value: str | None) -> date | None:
    text = str(value or "").strip()
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        try:
            return date(int(match["year"]), int(match["month"]), int(match["day"]))
        except ValueError:
            return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def add_business_days(start: date, days: int) -> date:
    """仅跳过周六、周日；未来可在这里注入企业/法定节假日日历。"""
    current = start
    direction = 1 if days >= 0 else -1
    remaining = abs(days)
    while remaining:
        current += timedelta(days=direction)
        if current.weekday() < 5:
            remaining -= 1
    return current


def _add_months(start: date, months: int) -> date:
    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    day = min(start.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def add_time_offset(
    start: date,
    value: int,
    unit: TimeOffsetUnit,
    *,
    business_day: bool = False,
) -> date:
    if unit == TimeOffsetUnit.DAY:
        return add_business_days(start, value) if business_day else start + timedelta(days=value)
    if unit == TimeOffsetUnit.MONTH:
        return _add_months(start, value)
    return _add_months(start, value * 12)


def parse_time_rule(text: str | None) -> TimeRule:
    """覆盖合同常见固定日期、签署/生效相对日期及事件相对日期。"""
    original = str(text or "").strip()
    fixed = parse_date(original)
    if fixed:
        return TimeRule(
            rule_type=TimeRuleType.FIXED_DATE,
            fixed_date=fixed.isoformat(),
            direction=TimeDirection.ON,
            original_text=original,
        )
    if re.search(r"每(?:日|周|月|季度|年)", original):
        return TimeRule(
            rule_type=TimeRuleType.PERIODIC,
            periodic_expression=original,
            original_text=original,
        )
    offset_match = OFFSET_PATTERN.search(original)
    offset = int(offset_match["offset"]) if offset_match else None
    unit_text = offset_match["unit"] if offset_match else "日"
    unit = TimeOffsetUnit.MONTH if unit_text == "月" else TimeOffsetUnit.YEAR if unit_text == "年" else TimeOffsetUnit.DAY
    business_day = bool(offset_match and offset_match["business"])
    direction = TimeDirection.BEFORE if "前" in original and "后" not in original else TimeDirection.AFTER
    if offset_match is None and ("后" in original or "前" in original):
        return TimeRule(rule_type=TimeRuleType.NO_EXPLICIT_DATE, original_text=original)
    if "合同生效" in original or "生效之日" in original:
        return TimeRule(
            rule_type=TimeRuleType.RELATIVE_TO_CONTRACT_EFFECTIVE,
            base_event_type=ContractEventType.CONTRACT_EFFECTIVE,
            offset_value=offset if offset is not None else 0,
            offset_unit=unit,
            business_day=business_day,
            direction=direction if offset is not None else TimeDirection.ON,
            original_text=original,
        )
    if "合同签署" in original or "签约" in original or "签订" in original:
        return TimeRule(
            rule_type=TimeRuleType.RELATIVE_TO_CONTRACT_SIGNING,
            base_event_type=ContractEventType.CONTRACT_SIGNED,
            offset_value=offset if offset is not None else 0,
            offset_unit=unit,
            business_day=business_day,
            direction=direction if offset is not None else TimeDirection.ON,
            original_text=original,
        )
    for keyword, event_type in EVENT_KEYWORDS:
        if re.search(keyword, original):
            return TimeRule(
                rule_type=TimeRuleType.RELATIVE_TO_EVENT,
                base_event_type=event_type,
                offset_value=offset if offset is not None else 0,
                offset_unit=unit,
                business_day=business_day,
                direction=direction if offset is not None else TimeDirection.ON,
                original_text=original,
            )
    return TimeRule(rule_type=TimeRuleType.NO_EXPLICIT_DATE, original_text=original)


def normalize_time_rule(rule: TimeRule, original_text: str | None = None) -> TimeRule:
    """以 Python 解析结果修正 LLM 遗漏字段，但保留有效的模型结构化结果。"""
    text = str(original_text or rule.original_text or "").strip()
    parsed = parse_time_rule(text)
    if parsed.rule_type != TimeRuleType.NO_EXPLICIT_DATE:
        return parsed.model_copy(
            update={
                "base_event_id": rule.base_event_id,
                "offset_value": rule.offset_value if rule.offset_value is not None else parsed.offset_value,
                "business_day": rule.business_day or parsed.business_day,
            }
        )
    return rule.model_copy(update={"original_text": text})


def resolve_due_date(
    rule: TimeRule,
    event_dates: Mapping[ContractEventType | str, str | date],
) -> date | None:
    """根据结构化 TimeRule 和已发生事件确定性计算计划日期。"""
    if rule.rule_type == TimeRuleType.FIXED_DATE:
        return parse_date(rule.fixed_date)
    if rule.rule_type in {TimeRuleType.PERIODIC, TimeRuleType.NO_EXPLICIT_DATE}:
        return None
    base_type = rule.base_event_type
    if base_type is None:
        return None
    raw = event_dates.get(base_type) or event_dates.get(base_type.value)
    base_date = raw if isinstance(raw, date) else parse_date(str(raw or ""))
    if base_date is None:
        return None
    if rule.offset_value is None and rule.direction != TimeDirection.ON:
        return None
    offset = int(rule.offset_value or 0)
    if rule.direction == TimeDirection.BEFORE:
        offset = -offset
    return add_time_offset(base_date, offset, rule.offset_unit, business_day=rule.business_day)


def resolve_obligation(
    obligation: ContractObligation,
    event_dates: Mapping[ContractEventType | str, str | date],
    *,
    today: date | None = None,
) -> ContractObligation:
    planned = resolve_due_date(obligation.time_rule, event_dates)
    protected = {ObligationStatus.COMPLETED, ObligationStatus.WAIVED, ObligationStatus.CANCELLED, ObligationStatus.IN_PROGRESS}
    if obligation.status in protected:
        status = obligation.status
    elif planned and planned < (today or date.today()):
        status = ObligationStatus.OVERDUE
    elif obligation.time_rule.rule_type in {
        TimeRuleType.RELATIVE_TO_CONTRACT_EFFECTIVE,
        TimeRuleType.RELATIVE_TO_CONTRACT_SIGNING,
        TimeRuleType.RELATIVE_TO_EVENT,
    } and planned is None:
        status = ObligationStatus.WAITING_TRIGGER
    else:
        status = ObligationStatus.PENDING
    return obligation.model_copy(update={"planned_date": planned.isoformat() if planned else None, "status": status})


def derive_trigger_type(rule: TimeRule) -> TriggerType:
    if rule.rule_type == TimeRuleType.FIXED_DATE:
        return TriggerType.FIXED
    if rule.rule_type in {TimeRuleType.RELATIVE_TO_CONTRACT_EFFECTIVE, TimeRuleType.RELATIVE_TO_CONTRACT_SIGNING}:
        return TriggerType.CONTRACT
    if rule.rule_type == TimeRuleType.RELATIVE_TO_EVENT:
        return TriggerType.EVENT
    if rule.rule_type == TimeRuleType.PERIODIC:
        return TriggerType.PERIODIC
    return TriggerType.NONE
