"""合同 Benchmark JSON/Markdown 报告生成。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def write_reports(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_report.json"
    markdown_path = output_dir / "benchmark_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    dataset = report["dataset"]
    metrics = report["metrics"]
    tags = Counter(dataset["case_type_distribution"])
    lines = [
        "# Contract Risk Benchmark Report",
        "",
        "## Executive Summary",
        "",
        f"- Run: `{metadata['evaluation_run_id']}`",
        f"- Mode: `{metadata['run_mode']}`（offline/mock 与 real model 不混淆）",
        f"- Dataset source: `{metadata.get('dataset_source_type', 'unknown')}`",
        f"- App version: `{metadata['app_version']}`",
        f"- Benchmark: `{metadata['benchmark_version']}` / dataset `{metadata['dataset_version']}`",
        f"- Cases: {dataset['case_count']}（synthetic={dataset['synthetic_count']}, real_uat={dataset['real_uat_count']}）",
        f"- Regression gate: **{'PASS' if report['gate']['passed'] else 'FAIL'}**",
        "",
        "## Dataset Composition",
        "",
        *[f"- {name}: {count}" for name, count in sorted(tags.items())],
        "",
        "## Core Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Field exact accuracy | {_pct(metrics['fields']['field_exact_accuracy'])} |",
        f"| Field normalized accuracy | {_pct(metrics['fields']['field_normalized_accuracy'])} |",
        f"| Obligation precision / recall / F1 | {_pct(metrics['obligations']['precision'])} / {_pct(metrics['obligations']['recall'])} / {_pct(metrics['obligations']['f1'])} |",
        f"| Fully correct TimeRule | {_pct(metrics['time_rules'].get('fully_correct', {}).get('accuracy', 0))} |",
        f"| Risk precision / recall / F1 | {_pct(metrics['risks']['precision'])} / {_pct(metrics['risks']['recall'])} / {_pct(metrics['risks']['f1'])} |",
        f"| High/Critical risk recall | {_pct(metrics['risks']['high_critical_recall'])} |",
        f"| Citation accuracy | {_pct(metrics['evidence']['citation_accuracy'])} |",
        f"| Unsupported claim rate | {_pct(metrics['evidence']['unsupported_claim_rate'])} |",
        f"| QA fact accuracy | {_pct(metrics['qa']['fact_accuracy'])} |",
        f"| No-evidence accuracy | {_pct(metrics['qa']['no_evidence_accuracy'])} |",
        f"| Conflict detection / dual evidence | {_pct(metrics['conflicts']['detection_recall'])} / {_pct(metrics['conflicts']['dual_evidence_accuracy'])} |",
        f"| Injection attack success rate | {_pct(metrics['security']['injection_attack_success_rate'])} |",
        "",
        "## Top Failure Categories",
        "",
        *([f"- {name}: {count}" for name, count in sorted(metrics['error_taxonomy'].items(), key=lambda item: (-item[1], item[0]))] or ["- None"]),
        "",
        "## Worst Cases",
        "",
    ]
    if report["top_failures"]:
        for item in report["top_failures"]:
            lines.extend(
                [
                    f"### {item['case_id']} — {item['error_type']}",
                    "",
                    f"- Expected: `{_short(item['expected'])}`",
                    f"- Predicted: `{_short(item['predicted'])}`",
                    f"- Source: `{_short(item.get('source'))}`",
                    "",
                ]
            )
    else:
        lines.extend(["No scored failures.", ""])
    lines.extend(
        [
            "## Latency",
            "",
            "p95 is emitted only when sample_count >= 20; smaller samples report mean/median/max.",
            "",
            *[f"- {name}: {value}" for name, value in metrics["latency_ms"].items()],
            "",
            "## Limitations",
            "",
            "- Synthetic data is a development benchmark, not evidence of real-enterprise UAT quality.",
            "- Risk annotation contains legal and business judgment; ambiguous items remain in human review.",
            "- Deterministic matching deliberately avoids LLM-as-a-Judge and may require adjudication at semantic boundaries.",
            "- Prompt-injection coverage does not prove complete security.",
            "- Benchmark findings must be fixed in a separate benchmark-driven optimization phase.",
            "",
        ]
    )
    return "\n".join(lines)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _short(value: Any, limit: int = 260) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit] + "…"
