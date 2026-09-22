"""Synthetic tests for the obligation structured-output contract boundary."""

import pytest
from pydantic import ValidationError

from qa_core.contracts.prompts import OBLIGATION_EXTRACTION_SYSTEM_PROMPT
from qa_core.contracts.schemas import (
    ContractObligation,
    ContractRisk,
    ObligationExtractionOutput,
    ObligationType,
    SourceEvidence,
    TimeOffsetUnit,
    TimeRule,
)


def _evidence() -> SourceEvidence:
    return SourceEvidence(
        chunk_id="synthetic-chunk-1",
        page_number=1,
        quote="Supplier shall deliver the synthetic equipment.",
    )


def _obligation(**updates) -> ContractObligation:
    payload = {
        "obligation_type": "delivery",
        "title": "Deliver synthetic equipment",
        "description": "Supplier shall deliver the synthetic equipment.",
        "source_evidence": _evidence(),
        "time_rule": {"offset_value": 1, "offset_unit": "day"},
    }
    payload.update(updates)
    return ContractObligation(**payload)


def test_canonical_title_is_unchanged_and_provider_schema_requires_it():
    item = _obligation(title="Deliver synthetic equipment")
    schema = ObligationExtractionOutput.model_json_schema()
    defs = next(value for key, value in schema.items() if key.endswith("defs"))

    assert item.title == "Deliver synthetic equipment"
    assert "title" in defs["ContractObligation"]["required"]
    assert defs["ContractObligation"]["properties"]["title"]["minLength"] == 1
    assert "必须输出非空 title" in OBLIGATION_EXTRACTION_SYSTEM_PROMPT


def test_missing_title_is_rejected_without_fallback():
    payload = _obligation().model_dump()
    payload.pop("title")
    with pytest.raises(ValidationError, match="title"):
        ContractObligation(**payload)


@pytest.mark.parametrize("title", ["", "   "])
def test_empty_or_blank_title_is_rejected(title):
    with pytest.raises(ValidationError, match="title"):
        _obligation(title=title)


def test_arbitrary_missing_semantic_field_is_still_rejected():
    with pytest.raises(ValidationError, match="risk_name"):
        ContractRisk(
            risk_type="synthetic",
            risk_level="low",
            description="Synthetic description",
            reason="Synthetic reason",
            suggestion="Synthetic suggestion",
        )


def test_title_validation_never_changes_evidence_metadata():
    evidence = _evidence()
    item = _obligation(title="  Deliver synthetic equipment  ", source_evidence=evidence)

    assert item.title == "Deliver synthetic equipment"
    assert item.source_evidence == evidence


def test_identical_validated_input_has_identical_title():
    first = _obligation()
    second = _obligation()

    assert first.title == second.title


def test_proven_penalty_alias_is_payment():
    assert _obligation(obligation_type="penalty").obligation_type is ObligationType.PAYMENT


@pytest.mark.parametrize(("unit", "value", "expected"), [("week", 1, 7), ("weeks", 2, 14)])
def test_proven_week_alias_is_converted_exactly_to_days(unit, value, expected):
    rule = TimeRule(offset_value=value, offset_unit=unit)

    assert rule.offset_unit is TimeOffsetUnit.DAY
    assert rule.offset_value == expected


@pytest.mark.parametrize("value", ["fine", "penalty_or_service", "unknown_kind"])
def test_unknown_or_ambiguous_obligation_type_is_rejected(value):
    with pytest.raises(ValidationError, match="obligation_type"):
        _obligation(obligation_type=value)


@pytest.mark.parametrize("value", ["fortnight", "business_week", "unknown_unit"])
def test_unknown_or_ambiguous_offset_unit_is_rejected(value):
    with pytest.raises(ValidationError, match="offset_unit"):
        TimeRule(offset_value=1, offset_unit=value)


def test_non_integral_week_is_rejected_instead_of_rounded():
    with pytest.raises(ValidationError, match="offset_value|offset_unit"):
        TimeRule(offset_value=1.5, offset_unit="week")
