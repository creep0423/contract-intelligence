"""合同 Benchmark regression gate；阈值与 hard-fail 条件分开报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qa_core.evaluation.schemas import GateThresholds


def evaluate_gate(report: dict[str, Any], thresholds: GateThresholds) -> dict[str, Any]:
    metrics = report["metrics"]
    checks = [
        _minimum("field_normalized_accuracy", metrics["fields"]["field_normalized_accuracy"], thresholds.minimum_field_accuracy),
        _minimum("obligation_recall", metrics["obligations"]["recall"], thresholds.minimum_obligation_recall),
        _minimum("risk_precision", metrics["risks"]["precision"], thresholds.minimum_risk_precision),
        _minimum("citation_accuracy", metrics["evidence"]["citation_accuracy"], thresholds.minimum_citation_accuracy),
        _maximum("hallucination_rate", metrics["qa"]["hallucination_rate"], thresholds.maximum_hallucination_rate),
        _maximum("no_evidence_failure_rate", 1 - metrics["qa"]["no_evidence_accuracy"], thresholds.maximum_no_evidence_failure_rate),
        _minimum("high_critical_risk_recall", metrics["risks"]["high_critical_recall"], thresholds.minimum_high_critical_risk_recall),
    ]
    hard_failures: list[str] = []
    security = metrics["security"]
    if thresholds.hard_fail_on_cross_tenant_leak and security["cross_contract_leak_rate"] > 0:
        hard_failures.append("security adversarial case returned cross-contract data")
    if thresholds.hard_fail_on_fake_citation and metrics["evidence"]["citation_count"] and metrics["evidence"]["valid_chunk_rate"] < 1:
        hard_failures.append("one or more citations reference an invalid/missing chunk")
    if thresholds.hard_fail_on_no_evidence_hallucination and metrics["qa"]["hallucination_rate"] > 0:
        hard_failures.append("NO_EVIDENCE case produced a hallucinated fact or citation")
    if thresholds.hard_fail_on_conflict_priority_inference and metrics["conflicts"]["unsupported_priority_inference_rate"] > 0:
        hard_failures.append("conflict case inferred unsupported legal priority")
    return {
        "passed": all(item["passed"] for item in checks) and not hard_failures,
        "threshold_checks": checks,
        "hard_failures": hard_failures,
        "thresholds": thresholds.model_dump(),
    }


def _minimum(name: str, actual: float, threshold: float) -> dict[str, Any]:
    return {"metric": name, "operator": ">=", "actual": actual, "threshold": threshold, "passed": actual >= threshold}


def _maximum(name: str, actual: float, threshold: float) -> dict[str, Any]:
    return {"metric": name, "operator": "<=", "actual": actual, "threshold": threshold, "passed": actual <= threshold}


def main() -> int:
    parser = argparse.ArgumentParser(description="检查合同 Benchmark regression gate")
    parser.add_argument("report")
    parser.add_argument("--thresholds")
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    thresholds = GateThresholds.model_validate_json(Path(args.thresholds).read_text(encoding="utf-8")) if args.thresholds else GateThresholds()
    result = evaluate_gate(report, thresholds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
