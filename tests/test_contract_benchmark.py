from copy import deepcopy
from pathlib import Path
import re

from qa_core.evaluation.compare_runs import compare_reports
from qa_core.evaluation.gate import evaluate_gate
from qa_core.evaluation.loader import load_benchmark, load_predictions, validate_annotations
from qa_core.evaluation.matching import (
    evidence_matches_gold,
    match_field,
    match_obligations,
    match_risks,
    normalize_date,
    normalize_number,
    normalize_percentage,
    values_match,
)
from qa_core.evaluation.metrics import evaluate_benchmark
from qa_core.evaluation.runner import DEFAULT_MANIFEST, run_benchmark
from qa_core.evaluation.schemas import (
    BenchmarkCase,
    ExpectationStatus,
    GateThresholds,
    GoldField,
    GoldObligation,
    GoldRisk,
    GoldSourceReference,
    MatchType,
    PredictedEvidence,
    PredictedField,
    PredictedObligation,
    PredictedRisk,
)


def test_numeric_date_percentage_and_currency_normalization():
    assert normalize_number("100万元") == normalize_number("1,000,000 CNY")
    assert normalize_date("2026年3月1日") == "2026-03-01"
    assert normalize_percentage("30%") == normalize_percentage("0.3")
    assert values_match("人民币", "CNY", MatchType.CURRENCY)[1]
    assert values_match(["20%", "30%"], ["30%", "20%"], MatchType.SET)[1]


def test_field_match_distinguishes_missing_hallucination_and_normalized_value():
    absent = GoldField(name="x", status=ExpectationStatus.NOT_PRESENT)
    assert match_field(absent, PredictedField(name="x", value=None)) == (True, True)
    assert match_field(absent, PredictedField(name="x", value="invented")) == (False, False)
    amount = GoldField(name="amount", status=ExpectationStatus.EXPECTED_VALUE, value="100万元", match_type=MatchType.NUMERIC)
    assert match_field(amount, PredictedField(name="amount", value="1000000")) == (False, True)


def test_evidence_match_uses_stable_document_clause_and_snippet_not_chunk_uuid():
    gold = GoldSourceReference(source_id="s1", document_name="master.md", document_version="1.0", page=2, clause_no="3.1", text_snippet="支付30%")
    predicted = PredictedEvidence(chunk_id="runtime-random", chunk_exists=True, document_name="master.md", document_version="1.0", page=2, clause_no="3.1", quote="甲方支付 30%", quote_verified=True)
    assert evidence_matches_gold(predicted, gold)
    assert not evidence_matches_gold(predicted.model_copy(update={"document_name": "other.md"}), gold)


def test_obligation_matching_does_not_depend_on_runtime_id():
    case = _case(
        obligations=[GoldObligation(gold_id="gold-1", obligation_type="payment", title="付款", responsible_party="甲方", percentage="30%", source_references=["s1"])]
    )
    predicted = PredictedObligation(obligation_type="payment", title="预付款", responsible_party="甲方", percentage="0.3", evidence=[_evidence()])
    (matches, missing, extra), _ = match_obligations(case, [predicted])
    assert len(matches) == 1 and missing == [] and extra == []


def test_risk_matching_ignores_description_wording_but_keeps_source_boundary():
    case = _case(risks=[GoldRisk(gold_risk_id="r1", risk_type="acceptance_subjective", expected_level="medium", description="验收标准较为主观", source_references=["s1"])])
    predicted = PredictedRisk(risk_type="acceptance_subjective", risk_level="medium", description="验收条件存在主观性", evidence=[_evidence()])
    (matches, missing, extra), _ = match_risks(case, [predicted])
    assert len(matches) == 1 and not missing and not extra


def test_no_evidence_scoring_requires_refusal_and_zero_sources():
    loaded = load_benchmark(DEFAULT_MANIFEST, suite="fast")
    bundle = load_predictions(loaded.root / loaded.manifest.offline_predictions_file)
    scored = evaluate_benchmark(loaded.cases, bundle.predictions)
    assert scored["metrics"]["qa"]["no_evidence_accuracy"] == 1.0
    assert scored["metrics"]["qa"]["fact_accuracy"] <= 1.0


def test_conflict_scoring_requires_detection_dual_evidence_and_no_priority_claim():
    loaded = load_benchmark(DEFAULT_MANIFEST, suite="fast")
    bundle = load_predictions(loaded.root / loaded.manifest.offline_predictions_file)
    scored = evaluate_benchmark(loaded.cases, bundle.predictions)
    assert scored["metrics"]["conflicts"] == {
        "detection_recall": 1.0,
        "false_positive_rate": 0.0,
        "dual_evidence_accuracy": 1.0,
        "unsupported_priority_inference_rate": 0.0,
        "sample_count": 1,
    }


def test_annotation_validation_detects_duplicate_case_ids():
    loaded = load_benchmark(DEFAULT_MANIFEST, suite="full")
    errors = validate_annotations(loaded.root, loaded.manifest, [loaded.cases[0], loaded.cases[0]])
    assert any("重复 case_id" in item for item in errors)


def test_run_comparison_reports_tradeoff_and_case_regression():
    baseline = {"metadata": {"evaluation_run_id": "b"}, "metrics": {"risk": {"precision": 0.8}, "obligation": {"recall": 0.7}}, "case_results": [{"case_id": "c1", "errors": []}]}
    candidate = {"metadata": {"evaluation_run_id": "c"}, "metrics": {"risk": {"precision": 0.7}, "obligation": {"recall": 0.9}}, "case_results": [{"case_id": "c1", "errors": [{"error_type": "RISK_FALSE_POSITIVE"}]}]}
    result = compare_reports(baseline, candidate)
    assert result["trade_off_warning"] is True
    assert result["regressions"][0]["new_errors"] == ["RISK_FALSE_POSITIVE"]


def test_regression_gate_has_metric_and_hard_fail_paths():
    loaded = load_benchmark(DEFAULT_MANIFEST, suite="fast")
    bundle = load_predictions(loaded.root / loaded.manifest.offline_predictions_file)
    report = evaluate_benchmark(loaded.cases, bundle.predictions)
    result = evaluate_gate(report, GateThresholds())
    assert result["passed"] is False
    assert any(item["metric"] == "high_critical_risk_recall" and not item["passed"] for item in result["threshold_checks"])
    poisoned = deepcopy(report)
    poisoned["metrics"]["qa"]["hallucination_rate"] = 0.2
    assert evaluate_gate(poisoned, GateThresholds())["hard_failures"]


def test_runner_writes_versioned_json_and_markdown(tmp_path):
    report, json_path, markdown_path = run_benchmark(manifest_path=DEFAULT_MANIFEST, suite="fast", output_root=tmp_path)
    assert json_path.is_file() and markdown_path.is_file()
    assert report["metadata"]["run_mode"] == "offline"
    app_version = report["metadata"]["app_version"]
    assert app_version.startswith("source-sha256:") or re.fullmatch(r"[0-9a-f]{40}", app_version)
    assert report["dataset"]["real_uat_count"] == 0


def _case(*, obligations=None, risks=None):
    return BenchmarkCase(
        case_id="case", contract_id="contract", contract_type="service", difficulty="easy", tags=["STANDARD"], synthetic=True,
        documents=["master.md"],
        sources=[GoldSourceReference(source_id="s1", document_name="master.md", document_version="1.0", clause_no="1", text_snippet="支付30%")],
        gold_obligations=obligations or [], gold_risks=risks or [],
    )


def _evidence():
    return PredictedEvidence(chunk_id="runtime", chunk_exists=True, document_name="master.md", document_version="1.0", clause_no="1", quote="甲方支付30%", quote_verified=True)
