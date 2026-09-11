"""合同领域 Pydantic 模型、统一证据模型与业务枚举。"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


def _normalize_date_or_aware_datetime(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("日期必须是 ISO date 或带时区的 ISO datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError("datetime 必须包含时区")
    return parsed.astimezone(timezone.utc).isoformat()


class ContractStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    ARCHIVED = "archived"


class ProcessingStatus(str, Enum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    PARSED = "parsed"
    ANALYZING = "analyzing"
    READY = "ready"
    FAILED = "failed"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ObligationStatus(str, Enum):
    PENDING = "pending"
    WAITING_TRIGGER = "waiting_trigger"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    WAIVED = "waived"
    CANCELLED = "cancelled"


class ObligationType(str, Enum):
    PAYMENT = "payment"
    RECEIVABLE = "receivable"
    INVOICE = "invoice"
    DELIVERY = "delivery"
    IMPLEMENTATION = "implementation"
    MILESTONE = "milestone"
    ACCEPTANCE = "acceptance"
    DOCUMENT_SUBMISSION = "document_submission"
    SERVICE = "service"
    DEPOSIT_PAYMENT = "deposit_payment"
    DEPOSIT_REFUND = "deposit_refund"
    NOTICE = "notice"
    RENEWAL = "renewal"
    TERMINATION = "termination"
    WARRANTY = "warranty"
    SLA = "sla"
    CONFIDENTIALITY = "confidentiality"
    INSURANCE = "insurance"
    AUDIT = "audit"
    OTHER = "other"


class TimeRuleType(str, Enum):
    FIXED_DATE = "fixed_date"
    RELATIVE_TO_CONTRACT_EFFECTIVE = "relative_to_contract_effective"
    RELATIVE_TO_CONTRACT_SIGNING = "relative_to_contract_signing"
    RELATIVE_TO_EVENT = "relative_to_event"
    PERIODIC = "periodic"
    NO_EXPLICIT_DATE = "no_explicit_date"


class TimeOffsetUnit(str, Enum):
    DAY = "day"
    MONTH = "month"
    YEAR = "year"


class TimeDirection(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    ON = "on"


class ContractEventType(str, Enum):
    CONTRACT_SIGNED = "contract_signed"
    CONTRACT_EFFECTIVE = "contract_effective"
    DELIVERY_COMPLETED = "delivery_completed"
    ACCEPTANCE_STARTED = "acceptance_started"
    ACCEPTANCE_PASSED = "acceptance_passed"
    ACCEPTANCE_FAILED = "acceptance_failed"
    INVOICE_RECEIVED = "invoice_received"
    PAYMENT_COMPLETED = "payment_completed"
    MILESTONE_COMPLETED = "milestone_completed"
    NOTICE_SENT = "notice_sent"
    CONTRACT_TERMINATED = "contract_terminated"
    WARRANTY_ENDED = "warranty_ended"
    CUSTOM = "custom"


class TriggerType(str, Enum):
    FIXED = "fixed"
    CONTRACT = "contract"
    EVENT = "event"
    PERIODIC = "periodic"
    NONE = "none"


class ContractDocumentType(str, Enum):
    MASTER_CONTRACT = "master_contract"
    SUPPLEMENTARY_AGREEMENT = "supplementary_agreement"
    AMENDMENT = "amendment"
    OTHER = "other"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskSource(str, Enum):
    RULE = "rule"
    LLM = "llm"
    HYBRID = "hybrid"


class RiskStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    IGNORED = "ignored"


class ContractModel(BaseModel):
    """把模型显式返回的 null 视为带 default_factory 字段的“未提供”。"""

    @model_validator(mode="before")
    @classmethod
    def default_factory_for_null(cls, value):
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for name, field in cls.model_fields.items():
            if name in normalized and normalized[name] is None and field.default_factory is not None:
                normalized[name] = field.default_factory()
        return normalized


class SourceEvidence(ContractModel):
    """抽取、义务、风险、摘要与里程碑共用的唯一证据结构。"""

    document_id: str | None = None
    document_name: str | None = None
    contract_id: str | None = None
    chunk_id: str | None = None
    page_number: int | None = None
    section: str | None = None
    clause_no: str | None = None
    clause_title: str | None = None
    quote: str = ""


class ContractField(ContractModel):
    extracted_value: str | None = None
    source_evidence: SourceEvidence | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_review: bool = False


class ContractBasicInfo(ContractModel):
    contract_name: ContractField = Field(default_factory=ContractField)
    contract_number: ContractField = Field(default_factory=ContractField)
    contract_type: ContractField = Field(default_factory=ContractField)
    party_a: ContractField = Field(default_factory=ContractField)
    party_b: ContractField = Field(default_factory=ContractField)
    other_parties: list[ContractField] = Field(default_factory=list)
    signing_date: ContractField = Field(default_factory=ContractField)
    effective_date: ContractField = Field(default_factory=ContractField)
    termination_date: ContractField = Field(default_factory=ContractField)
    contract_term: ContractField = Field(default_factory=ContractField)
    contract_amount: ContractField = Field(default_factory=ContractField)
    currency: ContractField = Field(default_factory=ContractField)


class PaymentTerm(ContractModel):
    title: str = "付款节点"
    trigger_event: str | None = None
    payment_days: int | None = None
    amount: str | None = None
    percentage: str | None = None
    invoice_required: bool | None = None
    invoice_requirement: str | None = None
    time_rule_text: str | None = None
    source_evidence: SourceEvidence | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_review: bool = False


class DeliveryTerm(ContractModel):
    deliverable: str
    delivery_time: str | None = None
    location: str | None = None
    method: str | None = None
    standard: str | None = None
    source_evidence: SourceEvidence | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_review: bool = False


class AcceptanceTerm(ContractModel):
    conditions: str | None = None
    standards: str | None = None
    deadline: str | None = None
    process: str | None = None
    default_acceptance: bool | None = None
    source_evidence: SourceEvidence | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_review: bool = False


class KeyTerm(ContractModel):
    term_type: str
    extracted_value: str
    source_evidence: SourceEvidence | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_review: bool = False


class ContractExtractionResult(ContractModel):
    basic_info: ContractBasicInfo = Field(default_factory=ContractBasicInfo)
    payment_terms: list[PaymentTerm] = Field(default_factory=list)
    delivery_terms: list[DeliveryTerm] = Field(default_factory=list)
    acceptance_terms: list[AcceptanceTerm] = Field(default_factory=list)
    termination_terms: list[KeyTerm] = Field(default_factory=list)
    liability_terms: list[KeyTerm] = Field(default_factory=list)
    confidentiality_terms: list[KeyTerm] = Field(default_factory=list)
    ip_terms: list[KeyTerm] = Field(default_factory=list)
    dispute_terms: list[KeyTerm] = Field(default_factory=list)
    other_key_terms: list[KeyTerm] = Field(default_factory=list)


class TimeRule(ContractModel):
    rule_type: TimeRuleType = TimeRuleType.NO_EXPLICIT_DATE
    fixed_date: str | None = None
    base_event_type: ContractEventType | None = None
    base_event_id: str | None = None
    offset_value: int | None = None
    offset_unit: TimeOffsetUnit = TimeOffsetUnit.DAY
    business_day: bool = False
    direction: TimeDirection = TimeDirection.AFTER
    periodic_expression: str | None = None
    original_text: str = ""


class ContractObligation(ContractModel):
    id: str | None = None
    contract_id: str | None = None
    obligation_type: ObligationType = ObligationType.OTHER
    title: str
    description: str = ""
    responsible_party: str | None = None
    beneficiary_party: str | None = None
    trigger_type: TriggerType = TriggerType.NONE
    trigger_event_type: ContractEventType | None = None
    trigger_event_id: str | None = None
    trigger_description: str | None = None
    time_rule: TimeRule = Field(default_factory=TimeRule)
    planned_date: str | None = None
    amount: str | None = None
    currency: str | None = None
    percentage: str | None = None
    status: ObligationStatus = ObligationStatus.PENDING
    risk_level: RiskLevel = RiskLevel.LOW
    source_evidence: SourceEvidence | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_review: bool = False


class ContractRisk(ContractModel):
    risk_type: str
    risk_name: str
    risk_level: RiskLevel
    description: str
    reason: str
    suggestion: str
    impact: str | None = None
    risk_source: RiskSource = RiskSource.LLM
    status: RiskStatus = RiskStatus.OPEN
    severity: int = Field(default=1, ge=1, le=4)
    likelihood: int = Field(default=1, ge=1, le=4)
    urgency: int = Field(default=1, ge=1, le=4)
    source_evidence: SourceEvidence | None = None
    supporting_evidence: list[SourceEvidence] = Field(default_factory=list)
    related_obligation_id: str | None = None
    related_obligation_index: int | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_review: bool = False


class SummarySection(ContractModel):
    title: str
    content: str
    source_evidence: list[SourceEvidence] = Field(default_factory=list)


class BasicExtractionOutput(ContractModel):
    basic_info: ContractBasicInfo = Field(default_factory=ContractBasicInfo)


class TermsExtractionOutput(ContractModel):
    payment_terms: list[PaymentTerm] = Field(default_factory=list)
    delivery_terms: list[DeliveryTerm] = Field(default_factory=list)
    acceptance_terms: list[AcceptanceTerm] = Field(default_factory=list)
    termination_terms: list[KeyTerm] = Field(default_factory=list)
    liability_terms: list[KeyTerm] = Field(default_factory=list)
    confidentiality_terms: list[KeyTerm] = Field(default_factory=list)
    ip_terms: list[KeyTerm] = Field(default_factory=list)
    dispute_terms: list[KeyTerm] = Field(default_factory=list)
    other_key_terms: list[KeyTerm] = Field(default_factory=list)


class ObligationExtractionOutput(ContractModel):
    obligations: list[ContractObligation] = Field(default_factory=list)


class SemanticRiskOutput(ContractModel):
    risks: list[ContractRisk] = Field(default_factory=list)


class ContractSummaryOutput(ContractModel):
    sections: list[SummarySection] = Field(default_factory=list)


class ContractAnalysisResult(ContractModel):
    extraction: ContractExtractionResult = Field(default_factory=ContractExtractionResult)
    summary: list[SummarySection] = Field(default_factory=list)
    obligations: list[ContractObligation] = Field(default_factory=list)
    risks: list[ContractRisk] = Field(default_factory=list)


class ContractAccessContext(ContractModel):
    tenant_id: str = "default"
    user_id: str = "anonymous"
    visibility: str = "private"
    user_roles: list[str] = Field(default_factory=lambda: ["public"])

    @field_validator("tenant_id", "user_id")
    @classmethod
    def validate_identity_token(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned or len(cleaned) > 128 or not re.fullmatch(r"[A-Za-z0-9._:-]+", cleaned):
            raise ValueError("身份数据域标识格式无效")
        return cleaned

    @field_validator("visibility")
    @classmethod
    def validate_visibility(cls, value: str) -> str:
        cleaned = str(value or "").strip().lower()
        if cleaned not in {"public", "internal", "private"}:
            raise ValueError("visibility 必须是 public、internal 或 private")
        return cleaned

    @field_validator("user_roles")
    @classmethod
    def validate_roles(cls, value: list[str]) -> list[str]:
        if len(value) > 32:
            raise ValueError("用户角色数量不能超过 32")
        roles: list[str] = []
        for raw in value:
            role = str(raw or "").strip()
            if not role or len(role) > 64 or not re.fullmatch(r"[A-Za-z0-9._:-]+", role):
                raise ValueError("用户角色格式无效")
            if role not in roles:
                roles.append(role)
        return roles or ["public"]


class ObligationStatusUpdate(ContractModel):
    status: ObligationStatus
    actual_completed_at: str | None = None
    note: str | None = Field(default=None, max_length=1000)

    @field_validator("actual_completed_at")
    @classmethod
    def normalize_completed_at(cls, value: str | None) -> str | None:
        return _normalize_date_or_aware_datetime(value)


class RiskStatusUpdate(ContractModel):
    status: RiskStatus


class ContractStatusUpdate(ContractModel):
    status: ContractStatus


class ContractEventCreate(ContractModel):
    event_type: ContractEventType
    event_date: str
    related_obligation_id: str | None = None
    description: str = Field(default="", max_length=1000)
    source: str = Field(default="manual", max_length=64)

    @field_validator("event_date")
    @classmethod
    def normalize_event_date(cls, value: str) -> str:
        return _normalize_date_or_aware_datetime(value) or ""


class ContractEventUpdate(ContractModel):
    event_type: ContractEventType | None = None
    event_date: str | None = None
    related_obligation_id: str | None = None
    description: str | None = Field(default=None, max_length=1000)
    source: str | None = Field(default=None, max_length=64)

    @field_validator("event_date")
    @classmethod
    def normalize_event_date(cls, value: str | None) -> str | None:
        return _normalize_date_or_aware_datetime(value)


class ContractUploadOptions(ContractModel):
    contract_name: str | None = Field(default=None, max_length=255)
    contract_type: str | None = Field(default=None, max_length=64)
    document_type: ContractDocumentType = ContractDocumentType.MASTER_CONTRACT
    document_version: str = Field(default="1", max_length=64)
    visibility: str = "private"
    allowed_roles: list[str] = Field(default_factory=lambda: ["public"])
    auto_analyze: bool = True

    @field_validator("allowed_roles")
    @classmethod
    def clean_roles(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        return cleaned or ["public"]
