"""履约义务抽取、TimeRule 规范化及事件驱动重算。"""

from __future__ import annotations

import uuid
from datetime import date

from qa_core.contracts.evidence import evidence_blocks, ground_model_evidence
from qa_core.contracts.prompts import OBLIGATION_EXTRACTION_SYSTEM_PROMPT, task_user_prompt
from qa_core.contracts.retrieval import ContractChunkRetriever, ContractRetrievalContext
from qa_core.contracts.schemas import (
    ContractEventType,
    ContractExtractionResult,
    ContractObligation,
    ObligationExtractionOutput,
    ObligationStatus,
)
from qa_core.contracts.structured_tasks import StructuredTaskRunner
from qa_core.contracts.time_rules import derive_trigger_type, normalize_time_rule, resolve_obligation


class ContractObligationService:
    def __init__(self, retriever: ContractChunkRetriever, runner: StructuredTaskRunner) -> None:
        self.retriever = retriever
        self.runner = runner

    def extract(
        self,
        context: ContractRetrievalContext,
        extraction: ContractExtractionResult,
        *,
        today: date | None = None,
    ) -> list[ContractObligation]:
        chunks = self.retriever.retrieve(context, "obligation")
        if not chunks:
            return []
        output = self.runner.invoke(
            ObligationExtractionOutput,
            task_name="obligation",
            system_prompt=OBLIGATION_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=task_user_prompt(
                contract_id=context.contract_id,
                evidence_blocks=evidence_blocks(chunks),
                confirmed_facts=extraction.model_dump_json(exclude_none=True),
            ),
        )
        grounded = ground_model_evidence(output, chunks)
        initial_events = extraction_event_dates(extraction)
        result: list[ContractObligation] = []
        seen: set[tuple[str, str, str]] = set()
        for item in grounded.obligations:
            original_text = item.time_rule.original_text or item.trigger_description or item.description
            rule = normalize_time_rule(item.time_rule, original_text)
            normalized = item.model_copy(
                update={
                    "id": str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            "|".join(
                                (
                                    context.contract_id,
                                    item.obligation_type.value,
                                    item.title.strip(),
                                    str(item.source_evidence.chunk_id if item.source_evidence else ""),
                                )
                            ),
                        )
                    ),
                    "contract_id": context.contract_id,
                    "time_rule": rule.model_copy(update={"base_event_id": None}),
                    "trigger_type": derive_trigger_type(rule),
                    "trigger_event_type": rule.base_event_type,
                    "trigger_event_id": None,
                }
            )
            normalized = resolve_obligation(normalized, initial_events, today=today)
            evidence_id = normalized.source_evidence.chunk_id if normalized.source_evidence else ""
            key = (normalized.obligation_type.value, normalized.title.strip(), str(evidence_id))
            if key not in seen:
                seen.add(key)
                result.append(normalized)
        return result


OBLIGATION_STATUS_TRANSITIONS: dict[ObligationStatus, set[ObligationStatus]] = {
    ObligationStatus.PENDING: {ObligationStatus.PENDING, ObligationStatus.IN_PROGRESS, ObligationStatus.COMPLETED, ObligationStatus.WAIVED, ObligationStatus.CANCELLED},
    ObligationStatus.WAITING_TRIGGER: {ObligationStatus.WAITING_TRIGGER, ObligationStatus.PENDING, ObligationStatus.IN_PROGRESS, ObligationStatus.COMPLETED, ObligationStatus.WAIVED, ObligationStatus.CANCELLED},
    ObligationStatus.IN_PROGRESS: {ObligationStatus.IN_PROGRESS, ObligationStatus.PENDING, ObligationStatus.COMPLETED, ObligationStatus.WAIVED, ObligationStatus.CANCELLED},
    ObligationStatus.OVERDUE: {ObligationStatus.OVERDUE, ObligationStatus.IN_PROGRESS, ObligationStatus.COMPLETED, ObligationStatus.WAIVED, ObligationStatus.CANCELLED},
    ObligationStatus.COMPLETED: {ObligationStatus.COMPLETED},
    ObligationStatus.WAIVED: {ObligationStatus.WAIVED},
    ObligationStatus.CANCELLED: {ObligationStatus.CANCELLED},
}


def transition_obligation_status(current: ObligationStatus, target: ObligationStatus) -> ObligationStatus:
    if target not in OBLIGATION_STATUS_TRANSITIONS[current]:
        raise ValueError(f"不允许将履约状态从 {current.value} 修改为 {target.value}")
    return target


def extraction_event_dates(extraction: ContractExtractionResult) -> dict[ContractEventType, str]:
    result: dict[ContractEventType, str] = {}
    signed = extraction.basic_info.signing_date.extracted_value
    effective = extraction.basic_info.effective_date.extracted_value
    if signed:
        result[ContractEventType.CONTRACT_SIGNED] = signed
    if effective:
        result[ContractEventType.CONTRACT_EFFECTIVE] = effective
    return result


def recalculate_obligations(
    obligations: list[ContractObligation],
    extraction: ContractExtractionResult,
    events: list[dict],
    *,
    today: date | None = None,
) -> list[ContractObligation]:
    """新增/修改事件后统一重算所有依赖义务；已完成/豁免/取消状态受保护。"""
    base_dates: dict[ContractEventType | str, str] = dict(extraction_event_dates(extraction))
    events_by_id = {str(event.get("id")): event for event in events if event.get("id")}
    latest_by_type: dict[str, dict] = {}
    for event in sorted(events, key=lambda item: (str(item.get("event_date") or ""), str(item.get("created_at") or ""), str(item.get("id") or ""))):
        event_type = str(event.get("event_type") or "")
        if event_type:
            latest_by_type[event_type] = event
    result: list[ContractObligation] = []
    for item in obligations:
        event_type = item.time_rule.base_event_type.value if item.time_rule.base_event_type else ""
        requested_event_id = item.time_rule.base_event_id or item.trigger_event_id
        selected_event = events_by_id.get(str(requested_event_id or "")) or latest_by_type.get(event_type)
        event_dates = dict(base_dates)
        if selected_event and event_type:
            event_dates[event_type] = str(selected_event.get("event_date") or "")
        resolved = resolve_obligation(item, event_dates, today=today)
        if event_type and selected_event:
            selected_id = str(selected_event.get("id") or "")
            resolved = resolved.model_copy(
                update={
                    "trigger_event_id": selected_id,
                    "time_rule": resolved.time_rule.model_copy(update={"base_event_id": selected_id}),
                }
            )
        result.append(resolved)
    return result
