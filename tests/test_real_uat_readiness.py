from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from qa_core.api import dependencies as api_dependencies
from qa_core.evaluation.loader import load_benchmark, validate_prediction_access
from qa_core.evaluation.matching import evidence_matches_gold
from qa_core.evaluation.runner import DEFAULT_MANIFEST, run_benchmark
from qa_core.evaluation.schemas import (
    BenchmarkCase,
    BenchmarkManifest,
    CasePrediction,
    GoldSourceReference,
    GoldTimeRule,
    PredictedEvidence,
    PredictionBundle,
    PredictionDataAccess,
)


def _real_manifest_payload() -> dict:
    return {
        "benchmark_version": "contract-benchmark-v1.0.0",
        "dataset_name": "Contract Intelligence Real UAT",
        "dataset_version": "contract-uat-v1.0.0",
        "source_type": "real_sanitized",
        "description": "脱敏真实合同",
        "cases_file": "annotations/cases.json",
        "prompt_version": "frozen",
        "retrieval_version": "frozen",
        "model_identifier": "external-authorized-generator",
        "supported_case_types": ["STANDARD"],
    }


def _real_case_payload() -> dict:
    return {
        "case_id": "UAT-001",
        "contract_id": "contract-stable-001",
        "industry": "manufacturing",
        "contract_type": "procurement",
        "contract_version": "1.0",
        "difficulty": "medium",
        "tags": ["STANDARD"],
        "synthetic": False,
        "suites": ["full"],
        "documents": ["contracts/private-contract.pdf"],
        "sources": [
            {
                "source_id": "payment-1",
                "document_id": "UAT-001-MASTER",
                "document_name": "private-contract.pdf",
                "document_version": "1.0",
                "page": 3,
                "clause": "4.1",
                "quote": "验收合格后30日内支付",
            }
        ],
        "gold_fields": [],
        "gold_obligations": [],
        "gold_time_rules": [],
        "gold_risks": [],
        "gold_qa": [],
        "gold_conflicts": [],
    }


def _authorized_bundle() -> PredictionBundle:
    return PredictionBundle(
        mode="snapshot",
        model_identifier="authorized-snapshot-v1",
        generated_at="2026-09-10T00:00:00Z",
        data_access=PredictionDataAccess(
            access_path="authorized_snapshot",
            tenant_id="tenant-uat",
            dataset_id="contract-uat-v1",
            visibility="private",
            allowed_roles=["legal"],
            trusted_upstream_verified=True,
        ),
        predictions=[CasePrediction(case_id="UAT-001")],
    )


def test_legacy_synthetic_manifest_remains_compatible():
    loaded = load_benchmark(DEFAULT_MANIFEST, suite="full")
    assert loaded.manifest.dataset_name == loaded.manifest.name
    assert loaded.manifest.source_type.value == "synthetic"
    assert len(loaded.cases) == 10


def test_real_uat_manifest_does_not_open_registered_contract(tmp_path):
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps(_real_manifest_payload()), encoding="utf-8")
    (annotation_dir / "cases.json").write_text(json.dumps([_real_case_payload()]), encoding="utf-8")

    loaded = load_benchmark(tmp_path / "manifest.json", suite="full")

    assert loaded.manifest.dataset_name == "Contract Intelligence Real UAT"
    assert loaded.manifest.source_type.value == "real_sanitized"
    assert loaded.cases[0].sources[0].document_id == "UAT-001-MASTER"
    assert not (tmp_path / "contracts" / "private-contract.pdf").exists()


def test_real_uat_prediction_requires_authorized_datascope_attestation(tmp_path):
    manifest = BenchmarkManifest.model_validate(_real_manifest_payload())
    case = _real_case_payload()
    parsed_case = BenchmarkCase.model_validate(case)
    unauthorized = PredictionBundle(
        mode="snapshot",
        model_identifier="untrusted",
        generated_at="2026-09-10T00:00:00Z",
        predictions=[CasePrediction(case_id=case["case_id"])],
    )
    assert validate_prediction_access(manifest, [parsed_case], unauthorized)
    assert validate_prediction_access(manifest, [parsed_case], _authorized_bundle()) == []


def test_real_uat_report_records_dataset_source_type(tmp_path):
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    predictions_path = tmp_path / "predictions.json"
    manifest_path.write_text(json.dumps(_real_manifest_payload()), encoding="utf-8")
    (annotation_dir / "cases.json").write_text(json.dumps([_real_case_payload()]), encoding="utf-8")
    predictions_path.write_text(_authorized_bundle().model_dump_json(), encoding="utf-8")

    report, _, markdown_path = run_benchmark(
        manifest_path=manifest_path,
        suite="full",
        predictions_path=predictions_path,
        output_root=tmp_path / "report",
    )

    assert report["metadata"]["dataset_source_type"] == "real_sanitized"
    assert "Dataset source: `real_sanitized`" in markdown_path.read_text(encoding="utf-8")


def test_gold_evidence_prefers_stable_document_id_and_accepts_new_names():
    gold = GoldSourceReference(
        source_id="s1",
        document_id="DOC-STABLE-1",
        document_name="display.pdf",
        page=2,
        clause="3.1",
        quote="支付30%",
    )
    predicted = PredictedEvidence(
        chunk_id="runtime-random",
        document_id="DOC-STABLE-1",
        document_name="renamed.pdf",
        page=2,
        clause="3.1",
        quote="甲方支付30%",
    )
    assert evidence_matches_gold(predicted, gold)
    assert not evidence_matches_gold(predicted.model_copy(update={"document_id": "DOC-OTHER"}), gold)


def test_time_rule_rejects_annotation_drift():
    with pytest.raises(ValueError):
        GoldTimeRule(rule_type="whenever_model_thinks")
    with pytest.raises(ValueError):
        GoldTimeRule(rule_type="relative_to_event", direction="later-ish")


def test_uat_identity_requires_trusted_upstream_token(monkeypatch):
    headers = {
        "x-tenant-id": "tenant-a",
        "x-user-id": "reviewer",
        "x-visibility": "private",
        "x-user-roles": "legal",
    }
    monkeypatch.setattr(
        api_dependencies,
        "settings",
        SimpleNamespace(contract_trusted_upstream_token="", env="uat"),
    )
    with pytest.raises(Exception) as missing:
        api_dependencies._resolve_contract_access_headers(headers, None)
    assert getattr(missing.value, "status_code", None) == 500

    monkeypatch.setattr(
        api_dependencies,
        "settings",
        SimpleNamespace(contract_trusted_upstream_token="uat-secret", env="uat"),
    )
    with pytest.raises(Exception) as rejected:
        api_dependencies._resolve_contract_access_headers(headers, "wrong")
    assert getattr(rejected.value, "status_code", None) == 401
    access = api_dependencies._resolve_contract_access_headers(headers, "uat-secret")
    assert access.tenant_id == "tenant-a"
    assert access.user_roles == ["legal"]
