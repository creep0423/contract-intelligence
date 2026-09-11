"""合同 Benchmark 的 Pydantic 数据契约。

Gold 数据只描述稳定业务语义和稳定来源，不保存 Milvus chunk UUID。运行时预测可以
携带 chunk_id，但 Gold 通过 document_name/version/page/clause/snippet 与其匹配。
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BenchmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExpectationStatus(str, Enum):
    EXPECTED_VALUE = "EXPECTED_VALUE"
    NOT_PRESENT = "NOT_PRESENT"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    AMBIGUOUS = "AMBIGUOUS"


class MatchType(str, Enum):
    EXACT = "exact"
    NORMALIZED = "normalized"
    NUMERIC = "numeric"
    DATE = "date"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    SET = "set"


class QAIntent(str, Enum):
    FACT = "FACT"
    NO_EVIDENCE = "NO_EVIDENCE"
    RISK = "RISK"
    SUGGESTION = "SUGGESTION"


class DatasetSourceType(str, Enum):
    SYNTHETIC = "synthetic"
    REAL_SANITIZED = "real_sanitized"
    SNAPSHOT = "snapshot"
    REAL_MODEL = "real_model"


BenchmarkTimeRuleType = Literal[
    "fixed_date",
    "relative_date",
    "relative_event",
    "relative_to_event",
    "relative_to_contract_effective",
    "relative_to_contract_signing",
    "periodic",
    "no_explicit_date",
    "ambiguous",
]
BenchmarkEventType = Literal[
    "contract_signed",
    "contract_effective",
    "delivery_completed",
    "acceptance_started",
    "acceptance_passed",
    "acceptance_failed",
    "invoice_received",
    "payment_completed",
    "milestone_completed",
    "notice_sent",
    "contract_terminated",
    "warranty_ended",
    "custom",
]
BenchmarkOffsetUnit = Literal["day", "month", "year"]
BenchmarkTimeDirection = Literal["before", "after", "on"]


class ErrorCategory(str, Enum):
    FIELD_MISSING = "FIELD_MISSING"
    FIELD_WRONG_VALUE = "FIELD_WRONG_VALUE"
    FIELD_HALLUCINATION = "FIELD_HALLUCINATION"
    OBLIGATION_MISSING = "OBLIGATION_MISSING"
    OBLIGATION_FALSE_POSITIVE = "OBLIGATION_FALSE_POSITIVE"
    OBLIGATION_WRONG_PARTY = "OBLIGATION_WRONG_PARTY"
    OBLIGATION_WRONG_TIME_RULE = "OBLIGATION_WRONG_TIME_RULE"
    TIME_WRONG_BASE_EVENT = "TIME_WRONG_BASE_EVENT"
    TIME_WRONG_OFFSET = "TIME_WRONG_OFFSET"
    RISK_MISSING = "RISK_MISSING"
    RISK_FALSE_POSITIVE = "RISK_FALSE_POSITIVE"
    RISK_WRONG_LEVEL = "RISK_WRONG_LEVEL"
    RISK_UNSUPPORTED = "RISK_UNSUPPORTED"
    EVIDENCE_WRONG_DOCUMENT = "EVIDENCE_WRONG_DOCUMENT"
    EVIDENCE_WRONG_CHUNK = "EVIDENCE_WRONG_CHUNK"
    EVIDENCE_INVALID_QUOTE = "EVIDENCE_INVALID_QUOTE"
    QA_WRONG_FACT = "QA_WRONG_FACT"
    QA_HALLUCINATION = "QA_HALLUCINATION"
    QA_FALSE_CITATION = "QA_FALSE_CITATION"
    CONFLICT_MISSED = "CONFLICT_MISSED"
    CONFLICT_WRONG_PRIORITY = "CONFLICT_WRONG_PRIORITY"
    PIPELINE_ERROR = "PIPELINE_ERROR"
    TIMEOUT = "TIMEOUT"


class GoldSourceReference(BenchmarkModel):
    source_id: str
    document_id: str | None = None
    document_name: str | None = None
    document_version: str | None = None
    page: int | None = Field(default=None, ge=1)
    clause_no: str | None = None
    clause: str | None = None
    text_snippet: str | None = None
    quote: str | None = None

    @model_validator(mode="after")
    def normalize_stable_reference(self):
        if not self.document_id and not self.document_name:
            raise ValueError("Gold Evidence 必须提供 document_id 或 document_name")
        if self.clause and self.clause_no and self.clause.strip() != self.clause_no.strip():
            raise ValueError("Gold Evidence 的 clause 与 clause_no 不一致")
        if self.quote and self.text_snippet and self.quote.strip() != self.text_snippet.strip():
            raise ValueError("Gold Evidence 的 quote 与 text_snippet 不一致")
        self.clause_no = self.clause_no or self.clause
        self.clause = self.clause or self.clause_no
        self.text_snippet = self.text_snippet or self.quote
        self.quote = self.quote or self.text_snippet
        if not str(self.text_snippet or "").strip():
            raise ValueError("Gold Evidence 必须提供 quote 或 text_snippet")
        return self


class GoldField(BenchmarkModel):
    name: str
    status: ExpectationStatus
    value: Any = None
    match_type: MatchType = MatchType.NORMALIZED
    source_references: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_value_state(self):
        if self.status == ExpectationStatus.EXPECTED_VALUE and self.value is None:
            raise ValueError("EXPECTED_VALUE 字段必须提供 value")
        if self.status in {ExpectationStatus.NOT_PRESENT, ExpectationStatus.NOT_APPLICABLE} and self.value is not None:
            raise ValueError(f"{self.status.value} 字段不得提供 value")
        return self


class GoldTimeRule(BenchmarkModel):
    rule_type: BenchmarkTimeRuleType | None = None
    base_event_type: BenchmarkEventType | None = None
    offset_value: int | None = None
    offset_unit: BenchmarkOffsetUnit | None = None
    business_day: bool | None = None
    direction: BenchmarkTimeDirection | None = None
    fixed_date: str | None = None
    resolved_date: str | None = None
    ambiguous: bool = False


class GoldObligation(BenchmarkModel):
    gold_id: str
    obligation_type: str
    title: str
    responsible_party: str | None = None
    beneficiary_party: str | None = None
    trigger_type: str | None = None
    time_rule: GoldTimeRule | None = None
    amount: Any = None
    percentage: Any = None
    source_references: list[str] = Field(default_factory=list)
    required: bool = True
    notes: str = ""


class GoldRisk(BenchmarkModel):
    gold_risk_id: str
    risk_type: str
    expected_level: str | None = None
    description: str = ""
    related_obligation_gold_id: str | None = None
    normalized_business_key: str | None = None
    source_references: list[str] = Field(default_factory=list)
    required: bool = True
    acceptable_variants: list[str] = Field(default_factory=list)
    risk_source: Literal["rule", "llm", "hybrid", "any"] = "any"
    notes: str = ""


class GoldQA(BenchmarkModel):
    question_id: str
    question: str
    intent: QAIntent
    expected_answer_facts: list[Any] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    allow_general_advice: bool = False
    notes: str = ""


class GoldConflict(BenchmarkModel):
    conflict_id: str
    conflict_type: str
    source_references: list[str] = Field(min_length=2)
    forbidden_priority_claims: list[str] = Field(default_factory=list)
    amendment: str | None = None
    old_value: Any = None
    new_value: Any = None


class ReviewerMetadata(BenchmarkModel):
    reviewer_id: str | None = None
    annotation_version: str = "1.0"
    adjudicated_result: bool = False
    usefulness: int | None = Field(default=None, ge=1, le=5)
    correctness: int | None = Field(default=None, ge=1, le=5)
    actionability: int | None = Field(default=None, ge=1, le=5)
    clarity: int | None = Field(default=None, ge=1, le=5)
    trust: int | None = Field(default=None, ge=1, le=5)
    would_rely: bool | None = None


class BenchmarkCase(BenchmarkModel):
    case_id: str
    contract_id: str
    industry: str | None = None
    contract_type: str
    contract_version: str | None = None
    difficulty: Literal["easy", "medium", "hard"]
    tags: list[str]
    synthetic: bool
    suites: list[Literal["fast", "full"]] = Field(default_factory=lambda: ["fast", "full"])
    documents: list[str]
    sources: list[GoldSourceReference]
    gold_fields: list[GoldField] = Field(default_factory=list)
    gold_obligations: list[GoldObligation] = Field(default_factory=list)
    gold_time_rules: list[GoldTimeRule] = Field(default_factory=list)
    gold_risks: list[GoldRisk] = Field(default_factory=list)
    gold_qa: list[GoldQA] = Field(default_factory=list)
    gold_conflicts: list[GoldConflict] = Field(default_factory=list)
    reviewer: ReviewerMetadata = Field(default_factory=ReviewerMetadata)
    notes: str = ""

    @model_validator(mode="after")
    def validate_real_uat_metadata(self):
        if not self.synthetic:
            if not str(self.industry or "").strip():
                raise ValueError("Real UAT case 必须提供 industry")
            if not str(self.contract_version or "").strip():
                raise ValueError("Real UAT case 必须提供 contract_version")
        return self

    @field_validator("tags")
    @classmethod
    def tags_are_uppercase(cls, value: list[str]) -> list[str]:
        normalized = [item.strip().upper() for item in value if item.strip()]
        if not normalized:
            raise ValueError("case 至少需要一个标签")
        return normalized


class BenchmarkManifest(BenchmarkModel):
    benchmark_version: str
    dataset_version: str
    dataset_name: str | None = None
    source_type: DatasetSourceType = DatasetSourceType.SYNTHETIC
    name: str | None = None
    description: str
    cases_file: str
    offline_predictions_file: str | None = None
    prompt_version: str
    retrieval_version: str
    model_identifier: str
    supported_case_types: list[str]

    @model_validator(mode="after")
    def normalize_dataset_name(self):
        if self.dataset_name and self.name and self.dataset_name.strip() != self.name.strip():
            raise ValueError("dataset_name 与兼容字段 name 不一致")
        normalized = str(self.dataset_name or self.name or "").strip()
        if not normalized:
            raise ValueError("manifest 必须提供 dataset_name 或兼容字段 name")
        self.dataset_name = normalized
        self.name = normalized
        return self


class PredictedEvidence(BenchmarkModel):
    chunk_id: str | None = None
    chunk_exists: bool = False
    document_id: str | None = None
    document_name: str | None = None
    document_version: str | None = None
    page: int | None = None
    clause_no: str | None = None
    clause: str | None = None
    quote: str = ""
    quote_verified: bool = False


class PredictedField(BenchmarkModel):
    name: str
    value: Any = None
    evidence: list[PredictedEvidence] = Field(default_factory=list)


class PredictedTimeRule(GoldTimeRule):
    pass


class PredictedObligation(BenchmarkModel):
    obligation_type: str
    title: str = ""
    responsible_party: str | None = None
    beneficiary_party: str | None = None
    trigger_type: str | None = None
    time_rule: PredictedTimeRule | None = None
    amount: Any = None
    percentage: Any = None
    evidence: list[PredictedEvidence] = Field(default_factory=list)


class PredictedRisk(BenchmarkModel):
    risk_type: str
    risk_level: str | None = None
    risk_source: Literal["rule", "llm", "hybrid"] = "llm"
    description: str = ""
    related_obligation_gold_id: str | None = None
    normalized_business_key: str | None = None
    evidence: list[PredictedEvidence] = Field(default_factory=list)


class PredictedQA(BenchmarkModel):
    question_id: str
    answer: str
    sources: list[PredictedEvidence] = Field(default_factory=list)
    latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class PredictedConflict(BenchmarkModel):
    conflict_type: str
    description: str = ""
    evidence: list[PredictedEvidence] = Field(default_factory=list)


class SecurityOutcome(BenchmarkModel):
    system_prompt_leaked: bool = False
    embedded_instruction_followed: bool = False
    cross_contract_data_returned: bool = False
    evidence_fabricated: bool = False
    task_changed: bool = False
    safe_failure: bool = False


class CasePrediction(BenchmarkModel):
    case_id: str
    analysis_id: str | None = None
    fields: list[PredictedField] = Field(default_factory=list)
    obligations: list[PredictedObligation] = Field(default_factory=list)
    time_rules: list[PredictedTimeRule] = Field(default_factory=list)
    risks: list[PredictedRisk] = Field(default_factory=list)
    qa: list[PredictedQA] = Field(default_factory=list)
    conflicts: list[PredictedConflict] = Field(default_factory=list)
    security: SecurityOutcome | None = None
    latencies_ms: dict[str, float] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)
    error: str | None = None


class PredictionDataAccess(BenchmarkModel):
    """真实数据预测生成阶段留下的 DataScope 授权声明；Evaluator 不生成或提升权限。"""

    access_path: Literal["authorized_data_scope", "authorized_snapshot"]
    tenant_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    visibility: Literal["public", "internal", "private"]
    allowed_roles: list[str] = Field(min_length=1)
    trusted_upstream_verified: bool


class PredictionBundle(BenchmarkModel):
    mode: Literal["offline", "mock", "real_model", "snapshot"]
    app_version: str | None = None
    model_identifier: str
    generated_at: str
    data_access: PredictionDataAccess | None = None
    predictions: list[CasePrediction]


class GateThresholds(BenchmarkModel):
    minimum_field_accuracy: float = Field(default=0.70, ge=0, le=1)
    minimum_obligation_recall: float = Field(default=0.70, ge=0, le=1)
    minimum_risk_precision: float = Field(default=0.60, ge=0, le=1)
    minimum_citation_accuracy: float = Field(default=0.80, ge=0, le=1)
    maximum_hallucination_rate: float = Field(default=0.05, ge=0, le=1)
    maximum_no_evidence_failure_rate: float = Field(default=0.05, ge=0, le=1)
    minimum_high_critical_risk_recall: float = Field(default=0.80, ge=0, le=1)
    hard_fail_on_cross_tenant_leak: bool = True
    hard_fail_on_fake_citation: bool = True
    hard_fail_on_no_evidence_hallucination: bool = True
    hard_fail_on_conflict_priority_inference: bool = True
