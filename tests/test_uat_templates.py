from __future__ import annotations

import json
from pathlib import Path

import pytest

from qa_core.evaluation.uat_templates import (
    ContractMetadataTemplate,
    DatasetManifestTemplate,
    RelatedDocumentType,
    validate_template_directory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = PROJECT_ROOT / "data_packs" / "contract_uat_templates"


def _load(name: str) -> dict:
    return json.loads((TEMPLATE_ROOT / name).read_text(encoding="utf-8"))


def test_versioned_uat_templates_are_valid_and_contain_no_real_cases():
    assert validate_template_directory(TEMPLATE_ROOT) == []
    manifest = DatasetManifestTemplate.model_validate(_load("dataset_manifest.template.json"))
    assert manifest.planned_case_count == 20
    assert manifest.case_ids == []
    assert manifest.source_type.value == "real_sanitized"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("contract_id", "CUSTOMER-A-001"),
        ("dataset_version", "latest"),
        ("source_type", "synthetic"),
        ("contract_type", "service"),
        ("company_role", "customer"),
        ("complexity", "NORMAL"),
        ("sanitization_status", "DONE"),
        ("annotation_status", "COMPLETE"),
        ("review_status", "PASSED"),
    ],
)
def test_contract_metadata_rejects_identifier_and_enum_drift(field: str, invalid_value: str):
    payload = _load("contract_metadata.template.json")
    payload[field] = invalid_value
    with pytest.raises(ValueError):
        ContractMetadataTemplate.model_validate(payload)


def test_related_document_requires_stable_id_and_supported_type():
    payload = _load("contract_metadata.template.json")
    payload.update(
        {
            "document_count": 2,
            "has_amendment": True,
            "related_documents": [
                {
                    "document_id": "CI-DOC-000002",
                    "related_document_type": RelatedDocumentType.AMENDMENT.value,
                }
            ],
        }
    )
    assert ContractMetadataTemplate.model_validate(payload).has_amendment is True
    payload["related_documents"][0]["related_document_type"] = "other_attachment"
    with pytest.raises(ValueError):
        ContractMetadataTemplate.model_validate(payload)


def test_dataset_manifest_distribution_and_case_ids_are_consistent():
    payload = _load("dataset_manifest.template.json")
    payload["complexity_distribution"]["HIGH"] = 3
    with pytest.raises(ValueError):
        DatasetManifestTemplate.model_validate(payload)
    payload = _load("dataset_manifest.template.json")
    payload["case_ids"] = ["CI-CONTRACT-000001", "CI-CONTRACT-000001"]
    with pytest.raises(ValueError):
        DatasetManifestTemplate.model_validate(payload)


def test_private_real_uat_boundaries_remain_git_ignored():
    ignored = {
        line.strip()
        for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert {
        "data/private_uat/",
        "data/private_annotations/",
        "artifacts/private/",
        "reports/private/",
        "models/",
    }.issubset(ignored)


def test_template_pack_contains_only_safe_json_templates():
    files = [path for path in TEMPLATE_ROOT.rglob("*") if path.is_file()]
    assert {path.name for path in files} == {
        "contract_metadata.template.json",
        "dataset_manifest.template.json",
        "annotation_status.template.json",
    }
    assert all(path.suffix == ".json" for path in files)
