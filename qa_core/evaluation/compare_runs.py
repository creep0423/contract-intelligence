"""比较两个 Benchmark JSON，展示指标 trade-off 与 case regressions。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare_reports(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    left = _flatten_numbers(baseline.get("metrics") or {})
    right = _flatten_numbers(candidate.get("metrics") or {})
    metric_deltas = [
        {"metric": key, "baseline": left[key], "candidate": right[key], "delta": round(right[key] - left[key], 6)}
        for key in sorted(left.keys() & right.keys())
    ]
    baseline_cases = _case_error_map(baseline)
    candidate_cases = _case_error_map(candidate)
    regressions = []
    improvements = []
    for case_id in sorted(baseline_cases.keys() | candidate_cases.keys()):
        added = sorted(candidate_cases.get(case_id, set()) - baseline_cases.get(case_id, set()))
        removed = sorted(baseline_cases.get(case_id, set()) - candidate_cases.get(case_id, set()))
        if added:
            regressions.append({"case_id": case_id, "new_errors": added})
        if removed:
            improvements.append({"case_id": case_id, "resolved_errors": removed})
    return {
        "baseline_run_id": baseline.get("metadata", {}).get("evaluation_run_id"),
        "candidate_run_id": candidate.get("metadata", {}).get("evaluation_run_id"),
        "metric_deltas": metric_deltas,
        "regressions": regressions,
        "improvements": improvements,
        "trade_off_warning": bool(any(item["delta"] > 0 for item in metric_deltas) and any(item["delta"] < 0 for item in metric_deltas)),
    }


def _flatten_numbers(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(_flatten_numbers(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = float(value)
    return result


def _case_error_map(report: dict[str, Any]) -> dict[str, set[str]]:
    return {
        item["case_id"]: {error["error_type"] for error in item.get("errors") or []}
        for item in report.get("case_results") or []
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="比较两个合同 Benchmark run")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = compare_reports(
        json.loads(Path(args.baseline).read_text(encoding="utf-8")),
        json.loads(Path(args.candidate).read_text(encoding="utf-8")),
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
