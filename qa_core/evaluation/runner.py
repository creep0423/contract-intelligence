"""合同 Benchmark Runner。

默认 offline，绝不隐式调用付费模型。真实模型/系统结果需先导出为 PredictionBundle，
再通过 --predictions 显式评分，报告会保留 bundle.mode，避免与 mock 混淆。
"""

from __future__ import annotations

import argparse
import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from qa_core.evaluation.gate import evaluate_gate
from qa_core.evaluation.loader import (
    load_benchmark,
    load_predictions,
    validate_prediction_access,
    validate_prediction_coverage,
)
from qa_core.evaluation.metrics import evaluate_benchmark
from qa_core.evaluation.report import write_reports
from qa_core.evaluation.schemas import GateThresholds
from qa_core.evaluation.versioning import resolve_app_version


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data_packs" / "contract_benchmark" / "manifest.json"


def run_benchmark(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    suite: str = "fast",
    predictions_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> tuple[dict, Path, Path]:
    started = datetime.now(timezone.utc)
    loaded = load_benchmark(manifest_path, suite=suite)
    if predictions_path is None:
        if not loaded.manifest.offline_predictions_file:
            raise ValueError("manifest 未配置 offline_predictions_file；请显式提供 --predictions")
        predictions_path = loaded.root / loaded.manifest.offline_predictions_file
    bundle = load_predictions(predictions_path)
    coverage_errors = validate_prediction_coverage(loaded.cases, bundle)
    if coverage_errors:
        raise ValueError("\n".join(coverage_errors))
    access_errors = validate_prediction_access(loaded.manifest, loaded.cases, bundle)
    if access_errors:
        raise ValueError("\n".join(access_errors))
    selected_ids = {case.case_id for case in loaded.cases}
    selected_predictions = [item for item in bundle.predictions if item.case_id in selected_ids]
    scored = evaluate_benchmark(loaded.cases, selected_predictions)
    app_version, version_source = resolve_app_version(PROJECT_ROOT)
    if bundle.app_version:
        app_version, version_source = bundle.app_version, "prediction_bundle"
    run_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    tags = Counter(tag for case in loaded.cases for tag in case.tags)
    finished = datetime.now(timezone.utc)
    report = {
        "metadata": {
            "evaluation_run_id": run_id,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_ms": round((finished - started).total_seconds() * 1000, 3),
            "run_mode": bundle.mode,
            "suite": suite,
            "app_version": app_version,
            "app_version_source": version_source,
            "benchmark_version": loaded.manifest.benchmark_version,
            "dataset_version": loaded.manifest.dataset_version,
            "dataset_source_type": loaded.manifest.source_type.value,
            "model_identifier": bundle.model_identifier,
            "prompt_version": loaded.manifest.prompt_version,
            "retrieval_version": loaded.manifest.retrieval_version,
        },
        "dataset": {
            "name": loaded.manifest.dataset_name,
            "case_count": len(loaded.cases),
            "synthetic_count": sum(case.synthetic for case in loaded.cases),
            "real_uat_count": sum(not case.synthetic for case in loaded.cases),
            "case_type_distribution": dict(sorted(tags.items())),
        },
        **scored,
    }
    thresholds_path = loaded.root / "gate.json"
    thresholds = GateThresholds.model_validate_json(thresholds_path.read_text(encoding="utf-8")) if thresholds_path.exists() else GateThresholds()
    report["gate"] = evaluate_gate(report, thresholds)
    destination = Path(output_root).resolve() if output_root else PROJECT_ROOT / "artifacts" / "benchmark" / run_id
    json_path, markdown_path = write_reports(report, destination)
    return report, json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(description="运行合同履约风控 Benchmark（默认 offline，不调用模型）")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--suite", choices=("fast", "full"), default="fast")
    parser.add_argument("--predictions", help="显式 prediction bundle；可为 mock/snapshot/real_model")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    report, json_path, markdown_path = run_benchmark(
        manifest_path=args.manifest,
        suite=args.suite,
        predictions_path=args.predictions,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "run_id": report["metadata"]["evaluation_run_id"],
                "mode": report["metadata"]["run_mode"],
                "cases": report["dataset"]["case_count"],
                "gate_passed": report["gate"]["passed"],
                "json_report": str(json_path),
                "markdown_report": str(markdown_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
