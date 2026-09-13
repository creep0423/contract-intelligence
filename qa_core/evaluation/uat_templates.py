"""Industrial Equipment Real UAT 安全模板 schema 与轻量校验。"""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from qa_core.evaluation.schemas import DatasetSourceType


TEMPLATE_FILE_MODELS: dict[str, type[BaseModel]] = {}
CONTRACT_ID_PATTERN = re.compile(r"^CI-CONTRACT-\d{6}$")
DOCUMENT_ID_PATTERN = re.compile(r"^CI-DOC-\d{6}$")
ANNOTATION_ID_PATTERN = re.compile(r"^CI-ANNOTATION-\d{6}$")
DATASET_VERSION_PATTERN = re.compile(r"^industrial-equipment-uat-v\d+\.\d+\.\d+$")


class UATTemplateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContractType(str, Enum):
    EQUIPMENT_PURCHASE = "equipment_purchase"
    EQUIPMENT_SALES = "equipment_sales"


class CompanyRole(str, Enum):
    BUYER = "buyer"
    SELLER = "seller"


class RelatedDocumentType(str, Enum):
    TECHNICAL_AGREEMENT = "technical_agreement"
    QUALITY_AGREEMENT = "quality_agreement"
    ACCEPTANCE_AGREEMENT = "acceptance_agreement"
    AMENDMENT = "amendment"
    CHANGE_ORDER = "change_order"


class Complexity(str, Enum):
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"
    HIGH = "HIGH"


class SanitizationStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"


class AnnotationStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    REVIEW = "REVIEW"
    FROZEN = "FROZEN"


class ReviewStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_REVIEW = "IN_REVIEW"
    APPROVED = "APPROVED"
    NEEDS_REVISION = "NEEDS_REVISION"


def _validate_identifier(value: str, pattern: re.Pattern[str], field_name: str) -> str:
    normalized = value.strip()
    if not pattern.fullmatch(normalized):
        raise ValueError(f"{field_name} 不符合匿名稳定 ID 格式")
    return normalized


def _validate_dataset_version(value: str) -> str:
    normalized = value.strip()
    if not DATASET_VERSION_PATTERN.fullmatch(normalized):
        raise ValueError("dataset_version 必须符合 industrial-equipment-uat-vMAJOR.MINOR.PATCH")
    return normalized


class RelatedDocumentTemplate(UATTemplateModel):
    document_id: str
    related_document_type: RelatedDocumentType

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        return _validate_identifier(value, DOCUMENT_ID_PATTERN, "document_id")


class ContractMetadataTemplate(UATTemplateModel):
    contract_id: str
    dataset_version: str
    source_type: DatasetSourceType
    parent_domain: str
    industry_segment: str
    contract_type: ContractType
    company_role: CompanyRole
    contract_version: str = Field(min_length=1)
    language: str = Field(min_length=1)
    document_count: int = Field(ge=1)
    related_documents: list[RelatedDocumentTemplate] = Field(default_factory=list)
    complexity: Complexity
    has_amendment: bool
    has_tables: bool
    sanitization_status: SanitizationStatus
    annotation_status: AnnotationStatus
    review_status: ReviewStatus

    @field_validator("contract_id")
    @classmethod
    def validate_contract_id(cls, value: str) -> str:
        return _validate_identifier(value, CONTRACT_ID_PATTERN, "contract_id")

    @field_validator("dataset_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _validate_dataset_version(value)

    @model_validator(mode="after")
    def validate_uat_boundary(self):
        if self.source_type is not DatasetSourceType.REAL_SANITIZED:
            raise ValueError("Real UAT metadata source_type 必须为 real_sanitized")
        if self.parent_domain != "manufacturing_supply_chain":
            raise ValueError("parent_domain 必须为 manufacturing_supply_chain")
        if self.industry_segment != "industrial_equipment_manufacturing":
            raise ValueError("industry_segment 必须为 industrial_equipment_manufacturing")
        if self.document_count != 1 + len(self.related_documents):
            raise ValueError("document_count 必须等于主合同 1 加 related_documents 数量")
        if self.has_amendment != any(
            item.related_document_type in {RelatedDocumentType.AMENDMENT, RelatedDocumentType.CHANGE_ORDER}
            for item in self.related_documents
        ):
            raise ValueError("has_amendment 必须与 amendment/change_order 文档保持一致")
        return self


class DatasetManifestTemplate(UATTemplateModel):
    dataset_name: str = Field(min_length=1)
    dataset_version: str
    source_type: DatasetSourceType
    parent_domain: str
    phase_1_icp: str
    planned_case_count: int = Field(ge=1)
    contract_type_minimums: dict[ContractType, int]
    flexible_case_count: int = Field(ge=0)
    amendment_coverage_minimum: int = Field(ge=0)
    complexity_distribution: dict[Complexity, int]
    case_ids: list[str]

    @field_validator("dataset_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _validate_dataset_version(value)

    @field_validator("case_ids")
    @classmethod
    def validate_case_ids(cls, value: list[str]) -> list[str]:
        normalized = [_validate_identifier(item, CONTRACT_ID_PATTERN, "case_id") for item in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("case_ids 不能重复")
        return normalized

    @model_validator(mode="after")
    def validate_plan(self):
        if self.source_type is not DatasetSourceType.REAL_SANITIZED:
            raise ValueError("Real UAT manifest source_type 必须为 real_sanitized")
        if self.parent_domain != "manufacturing_supply_chain":
            raise ValueError("parent_domain 必须为 manufacturing_supply_chain")
        if self.phase_1_icp != "industrial_equipment_manufacturers_50_300_employees":
            raise ValueError("phase_1_icp 不符合首批 ICP")
        if set(self.contract_type_minimums) != set(ContractType):
            raise ValueError("contract_type_minimums 必须覆盖 equipment_purchase 和 equipment_sales")
        minimum_total = sum(self.contract_type_minimums.values())
        if minimum_total + self.flexible_case_count != self.planned_case_count:
            raise ValueError("合同类型最低数量与 flexible_case_count 之和必须等于 planned_case_count")
        if set(self.complexity_distribution) != set(Complexity):
            raise ValueError("complexity_distribution 必须覆盖全部 complexity 枚举")
        if sum(self.complexity_distribution.values()) != self.planned_case_count:
            raise ValueError("complexity_distribution 数量之和必须等于 planned_case_count")
        if len(self.case_ids) > self.planned_case_count:
            raise ValueError("case_ids 数量不能超过 planned_case_count")
        return self


class AnnotationStatusTemplate(UATTemplateModel):
    annotation_id: str
    contract_id: str
    dataset_version: str
    annotation_status: AnnotationStatus
    review_status: ReviewStatus
    primary_reviewer_id: str = Field(min_length=1)
    secondary_reviewer_id: str | None = None
    adjudicator_id: str | None = None
    adjudication_result: str | None = None
    gold_frozen_at: str | None = None
    notes: str = ""

    @field_validator("annotation_id")
    @classmethod
    def validate_annotation_id(cls, value: str) -> str:
        return _validate_identifier(value, ANNOTATION_ID_PATTERN, "annotation_id")

    @field_validator("contract_id")
    @classmethod
    def validate_contract_id(cls, value: str) -> str:
        return _validate_identifier(value, CONTRACT_ID_PATTERN, "contract_id")

    @field_validator("dataset_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _validate_dataset_version(value)

    @model_validator(mode="after")
    def validate_freeze_state(self):
        if self.annotation_status is AnnotationStatus.FROZEN:
            if self.review_status is not ReviewStatus.APPROVED or not self.gold_frozen_at:
                raise ValueError("FROZEN annotation 必须已 APPROVED 并提供 gold_frozen_at")
        return self


TEMPLATE_FILE_MODELS.update(
    {
        "contract_metadata.template.json": ContractMetadataTemplate,
        "dataset_manifest.template.json": DatasetManifestTemplate,
        "annotation_status.template.json": AnnotationStatusTemplate,
    }
)


def validate_template_directory(root: Path) -> list[str]:
    """校验模板文件存在、JSON 可解析且满足匿名 ID 和枚举约束。"""

    errors: list[str] = []
    for filename, model in TEMPLATE_FILE_MODELS.items():
        path = root / filename
        if not path.is_file():
            errors.append(f"缺少模板文件：{filename}")
            continue
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
            model.model_validate(payload)
        except json.JSONDecodeError as exc:
            errors.append(f"{filename}: JSON 格式错误：{exc.msg}")
        except ValidationError as exc:
            errors.append(f"{filename}: {exc}")
    return errors
