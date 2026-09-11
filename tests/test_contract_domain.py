from datetime import date

from langchain_core.documents import Document

from qa_core.contracts.evidence import ground_model_evidence, validate_source_evidence
from qa_core.contracts.analysis import ContractAnalysisService
from qa_core.contracts.extraction import ContractExtractionService
from qa_core.contracts.obligations import ContractObligationService
from qa_core.contracts.obligations import recalculate_obligations
from qa_core.contracts.prompts import no_contract_evidence_answer
from qa_core.contracts.risk import deduplicate_risks, evaluate_deterministic_risks, normalize_risk_level
from qa_core.contracts.schemas import (
    ContractAccessContext,
    ContractBasicInfo,
    ContractExtractionResult,
    ContractField,
    ContractObligation,
    ContractRisk,
    BasicExtractionOutput,
    TermsExtractionOutput,
    ObligationExtractionOutput,
    ObligationStatus,
    ObligationType,
    RiskLevel,
    RiskSource,
    SourceEvidence,
)
from qa_core.contracts.store import can_access_contract
from qa_core.contracts.retrieval import ContractRetrievalContext
from qa_core.contracts.time_rules import add_business_days, parse_time_rule, resolve_due_date, resolve_obligation
from qa_core.indexing.chunking import split_contract_documents
from qa_core.scenarios.registry import get_scenario_registry


def evidence() -> SourceEvidence:
    return SourceEvidence(document_id="doc-1", chunk_id="chunk-1", page_number=2, quote="验收合格后30日内支付尾款")


def obligation(**updates) -> ContractObligation:
    payload = {
        "id": "ob-1",
        "contract_id": "contract-1",
        "obligation_type": ObligationType.PAYMENT,
        "title": "支付尾款",
        "time_rule": parse_time_rule("验收合格后30日内支付尾款"),
        "source_evidence": evidence(),
        "status": ObligationStatus.WAITING_TRIGGER,
    }
    payload.update(updates)
    return ContractObligation(**payload)


def retrieval_context() -> ContractRetrievalContext:
    return ContractRetrievalContext(
        contract_id="contract-1", dataset_id="contract:contract-1", tenant_id="tenant-a",
        visibility="private", allowed_roles=["legal"], scenario_id="tender_contract_risk",
    )


class FixedRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, context, task):
        return self.chunks


def test_time_rule_parses_fixed_effective_signing_and_event_dates():
    assert parse_time_rule("应于2026年5月1日前交付").fixed_date == "2026-05-01"
    assert parse_time_rule("合同生效后10日").rule_type.value == "relative_to_contract_effective"
    assert parse_time_rule("合同签署后2个月").rule_type.value == "relative_to_contract_signing"
    event_rule = parse_time_rule("验收合格后30日内支付尾款")
    assert event_rule.rule_type.value == "relative_to_event"
    assert resolve_due_date(event_rule, {"acceptance_passed": "2026-01-01"}).isoformat() == "2026-01-31"


def test_workday_offset_skips_weekends():
    assert add_business_days(date(2026, 9, 4), 3).isoformat() == "2026-09-09"


def test_event_recalculates_all_dependent_obligations_and_preserves_completed():
    items = [obligation(), obligation(id="ob-2", title="已支付尾款", status=ObligationStatus.COMPLETED)]
    result = recalculate_obligations(
        items,
        ContractExtractionResult(),
        [{"id": "event-1", "event_type": "acceptance_passed", "event_date": "2026-01-01"}],
        today=date(2026, 1, 2),
    )
    assert [item.planned_date for item in result] == ["2026-01-31", "2026-01-31"]
    assert result[0].status == ObligationStatus.PENDING
    assert result[1].status == ObligationStatus.COMPLETED
    assert all(item.trigger_event_id == "event-1" for item in result)


def test_relative_event_without_event_waits_for_trigger():
    resolved = resolve_obligation(obligation(), {}, today=date(2026, 1, 2))
    assert resolved.planned_date is None
    assert resolved.status == ObligationStatus.WAITING_TRIGGER


def test_overdue_and_upcoming_risks_are_deterministic_but_completed_is_excluded():
    items = [
        obligation(planned_date="2026-01-01", status=ObligationStatus.OVERDUE),
        obligation(id="ob-2", planned_date="2026-01-05", status=ObligationStatus.COMPLETED),
    ]
    risks = evaluate_deterministic_risks("contract-1", ContractExtractionResult(), items, today=date(2026, 1, 10))
    assert len(risks) == 1
    assert risks[0].risk_source == RiskSource.RULE
    assert risks[0].risk_name == "付款事项已逾期"


def test_risk_score_and_rule_llm_deduplication():
    assert normalize_risk_level(4, 4, 4) == RiskLevel.CRITICAL
    rule = ContractRisk(
        risk_type="payment", risk_name="付款事项已逾期", risk_level=RiskLevel.HIGH,
        description="已逾期", reason="日期已过", suggestion="催办", risk_source=RiskSource.RULE,
        related_obligation_id="ob-1", source_evidence=evidence(), severity=3, likelihood=4, urgency=4,
    )
    llm = rule.model_copy(update={"risk_name": "逾期付款风险", "risk_source": RiskSource.LLM, "risk_level": RiskLevel.MEDIUM})
    merged = deduplicate_risks("contract-1", [rule, llm])
    assert len(merged) == 1
    assert merged[0].risk_source == RiskSource.HYBRID
    assert merged[0].risk_level == RiskLevel.CRITICAL


def test_evidence_must_reference_input_chunk_and_quote_original_text():
    chunk = Document(
        page_content="甲方应在验收合格后30日内支付尾款。",
        metadata={"chunk_id": "chunk-1", "contract_document_id": "doc-1", "page_number": 2, "section": "付款", "clause_no": "4.2"},
    )
    valid = validate_source_evidence(evidence(), {"chunk-1": chunk})
    assert valid and valid.page_number == 2 and valid.clause_no == "4.2"
    invalid = validate_source_evidence(SourceEvidence(chunk_id="forged", quote="支付尾款"), {"chunk-1": chunk})
    assert invalid is None


def test_low_confidence_and_invalid_evidence_are_marked_for_review():
    field = ContractField(extracted_value="100万元", source_evidence=SourceEvidence(chunk_id="forged", quote="100万元"), confidence=0.4)
    grounded = ground_model_evidence(field, [])
    assert grounded.source_evidence is None
    assert grounded.needs_review is True


def test_fact_extraction_is_structured_grounded_and_deduplicated():
    chunk = Document(
        page_content="合同总金额为人民币100万元。",
        metadata={"chunk_id": "chunk-1", "contract_document_id": "doc-1", "page_number": 1},
    )

    class Runner:
        def invoke(self, schema, **kwargs):
            if schema is BasicExtractionOutput:
                return BasicExtractionOutput(basic_info=ContractBasicInfo(contract_amount=ContractField(extracted_value="100万元", source_evidence=SourceEvidence(chunk_id="chunk-1", quote="合同总金额为人民币100万元"), confidence=0.96)))
            return TermsExtractionOutput()

    result = ContractExtractionService(FixedRetriever([chunk]), Runner()).extract(retrieval_context())
    assert result.basic_info.contract_amount.extracted_value == "100万元"
    assert result.basic_info.contract_amount.source_evidence.page_number == 1
    assert result.basic_info.contract_amount.needs_review is False


def test_obligation_extraction_normalizes_time_rule_without_llm_date_calculation():
    chunk = Document(
        page_content="甲方应在验收合格后30日内支付尾款。",
        metadata={"chunk_id": "chunk-1", "contract_document_id": "doc-1", "page_number": 2},
    )

    class Runner:
        def invoke(self, schema, **kwargs):
            assert schema is ObligationExtractionOutput
            return ObligationExtractionOutput(obligations=[ContractObligation(
                obligation_type="payment", title="支付尾款", responsible_party="甲方",
                trigger_description="验收合格后30日内", time_rule={"original_text": "验收合格后30日内"},
                source_evidence=SourceEvidence(chunk_id="chunk-1", quote="验收合格后30日内支付尾款"), confidence=0.92,
            )])

    items = ContractObligationService(FixedRetriever([chunk]), Runner()).extract(retrieval_context(), ContractExtractionResult(), today=date(2026, 1, 1))
    assert len(items) == 1
    assert items[0].time_rule.rule_type.value == "relative_to_event"
    assert items[0].planned_date is None
    assert items[0].status == ObligationStatus.WAITING_TRIGGER


def test_contract_chunking_keeps_clause_metadata_and_parent_trace():
    source = Document(
        page_content="第一章 商务条款\n1.1 合同金额\n合同总价为100万元。\n1.2 付款\n验收后30日支付尾款。",
        metadata={"is_contract_document": True, "page_index": 0, "doc_id": "doc-1", "scenario_id": "tender_contract_risk"},
    )
    chunks, ids = split_contract_documents([source])
    assert len(chunks) == len(ids) >= 2
    assert all(item.metadata["page_number"] == 1 for item in chunks)
    assert all(item.metadata["parent_chunk_id"] for item in chunks)
    assert {item.metadata["clause_no"] for item in chunks} >= {"1.1", "1.2"}


def test_contract_access_is_tenant_and_role_scoped():
    record = {"tenant_id": "tenant-a", "visibility": "private", "allowed_roles": ["legal"]}
    assert can_access_contract(record, ContractAccessContext(tenant_id="tenant-a", visibility="private", user_roles=["legal"]))
    assert not can_access_contract(record, ContractAccessContext(tenant_id="tenant-b", user_roles=["legal"]))
    assert not can_access_contract(record, ContractAccessContext(tenant_id="tenant-a", user_roles=["sales"]))
    assert not can_access_contract(record, ContractAccessContext(tenant_id="tenant-a", visibility="public", user_roles=["legal"]))


def test_only_contract_scenario_is_exposed():
    registry = get_scenario_registry()
    assert [item.scenario_id for item in registry.list_runtime_scenarios()] == ["tender_contract_risk"]


def test_no_context_contract_answer_uses_required_wording():
    assert no_contract_evidence_answer() == "根据当前合同内容，未发现明确约定。"


def test_repeated_analysis_keeps_run_history_and_each_run_has_one_result():
    class FakeStore:
        def __init__(self):
            self.runs = []
            self.results = []

        def create_analysis_run(self, contract_id, *, model_name):
            run_id = f"run-{len(self.runs) + 1}"
            self.runs.append(run_id)
            return run_id

        def save_analysis_extraction(self, run_id, extraction):
            pass

        def save_analysis_result(self, run_id, contract_id, result, *, warning_summary=""):
            self.results.append((run_id, result))

        def fail_analysis(self, run_id, contract_id, error_summary):
            raise AssertionError(error_summary)

    class ExtractionService:
        def extract(self, context):
            return ContractExtractionResult()

    class ObligationService:
        def extract(self, context, extraction):
            return []

    class SemanticService:
        def analyze(self, context, extraction, obligations):
            return []

    class SummaryService:
        def generate(self, context, result):
            return []

    store = FakeStore()
    service = ContractAnalysisService(store, retriever=object(), runner=object())
    service.extraction_service = ExtractionService()
    service.obligation_service = ObligationService()
    service.semantic_risk_service = SemanticService()
    service.summary_service = SummaryService()
    contract = {
        "id": "contract-1", "dataset_id": "contract:contract-1", "tenant_id": "tenant-a",
        "visibility": "private", "allowed_roles": ["legal"], "scenario_id": "tender_contract_risk",
    }
    service.analyze(contract)
    service.analyze(contract)
    assert store.runs == ["run-1", "run-2"]
    assert [run_id for run_id, _ in store.results] == store.runs
