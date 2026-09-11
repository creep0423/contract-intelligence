"""RAGAS supplemental quality evaluation.

This script is intentionally a supplement to the local engineering gates:

- `evaluate_core_chain.py` / `check_evaluation_gate.py` remain the main regression
  gate for Recall@K, MRR, hit type, source inference, Prompt Profile routing,
  scenario isolation, DataScope and latency.
- This script adds LLM-as-judge style semantic checks such as faithfulness and
  answer relevancy. It should be used for offline diagnosis or release evidence,
  not as the only CI gate.

Typical workflow:

    python scripts/evaluate_core_chain.py --dataset eval_sets/multi_scenario_smoke.json --limit 20 --output reports/evaluation/core_chain_latest.json
    python scripts/check_evaluation_gate.py --report reports/evaluation/core_chain_latest.json
    python scripts/evaluate_ragas_quality.py --report reports/evaluation/core_chain_latest.json --limit 10
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import PROJECT_ROOT, configure_utf8_stdio, print_json, read_json_file, write_json_file


RAGAS_REPORT_DIR = PROJECT_ROOT / "reports" / "evaluation"


def project_path(path: str | Path) -> Path:
    """Resolve a command-line path relative to the project root.

    调用顺序：命令行入口 -> project_path()。
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_ragas_runtime():
    """Import RAGAS only when this supplemental script is actually executed.

    调用顺序：命令行入口 -> load_ragas_runtime()。
    """
    try:
        ragas_module = importlib.import_module("ragas")
        metrics_module = importlib.import_module("ragas.metrics")
        llms_module = importlib.import_module("ragas.llms.base")
        embeddings_module = importlib.import_module("ragas.embeddings.base")
    except ModuleNotFoundError as exc:
        missing = exc.name or "ragas"
        raise RuntimeError(
            f"RAGAS 补充评测需要安装 {missing}。请先确认 requirements.txt 中的 ragas 已安装。"
        ) from exc
    return ragas_module, metrics_module, llms_module, embeddings_module


def _source_content(source: dict[str, Any]) -> str:
    """Extract a RAGAS context string from a source payload.

    调用顺序：命令行入口 -> _source_content()。
    """
    content = str(source.get("content") or "").strip()
    if content:
        return content
    metadata = source.get("metadata") or {}
    fallback = [
        metadata.get("standard_question"),
        metadata.get("answer"),
        metadata.get("file_name"),
        metadata.get("file_path"),
    ]
    return " ".join(str(item or "") for item in fallback).strip()


def _reference_from_row(row: dict[str, Any]) -> str:
    """Build a lightweight reference text from explicit eval expectations.

    调用顺序：命令行入口 -> _reference_from_row()。
    """
    explicit = row.get("reference") or row.get("ground_truth") or row.get("expected_answer")
    if explicit:
        return str(explicit)
    keywords = [str(item).strip() for item in row.get("expected_keywords") or [] if str(item).strip()]
    if keywords:
        return "；".join(keywords)
    return ""


def build_ragas_rows(report: dict[str, Any], limit: int, max_contexts: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert the local engineering report rows into RAGAS single-turn rows.

    调用顺序：命令行入口 -> build_ragas_rows()。
    """
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in report.get("rows") or []:
        if len(rows) >= limit:
            break
        answer = str(row.get("answer") or row.get("answer_preview") or "").strip()
        sources = row.get("sources") or []
        contexts = [_source_content(source) for source in sources[:max_contexts]]
        contexts = [item for item in contexts if item]
        if not row.get("answer"):
            skipped.append({"case_id": row.get("case_id"), "reason": "missing_full_answer"})
            continue
        if not contexts:
            skipped.append({"case_id": row.get("case_id"), "reason": "missing_contexts"})
            continue
        item = {
            "user_input": str(row.get("question") or ""),
            "response": answer,
            "retrieved_contexts": contexts,
        }
        reference = _reference_from_row(row)
        if reference:
            item["reference"] = reference
        rows.append(item)
    return rows, skipped


def selected_metrics(metrics_module, metric_names: list[str], *, include_reference_metrics: bool):
    """Create RAGAS metric instances by stable project-level names.

    调用顺序：命令行入口 -> selected_metrics()。
    """
    metric_map = {
        "faithfulness": metrics_module.Faithfulness,
        "answer_relevancy": metrics_module.ResponseRelevancy,
        "response_relevancy": metrics_module.ResponseRelevancy,
        "context_relevance": metrics_module.ContextRelevance,
        "response_groundedness": metrics_module.ResponseGroundedness,
    }
    if include_reference_metrics:
        metric_map.update(
            {
                "context_precision": metrics_module.LLMContextPrecisionWithReference,
                "context_recall": metrics_module.LLMContextRecall,
            }
        )
    metrics = []
    unknown = []
    for name in metric_names:
        key = name.strip().lower()
        factory = metric_map.get(key)
        if factory is None:
            unknown.append(name)
            continue
        metrics.append(factory())
    if unknown:
        allowed = ", ".join(sorted(metric_map))
        raise ValueError(f"未知 RAGAS 指标：{unknown}。可选：{allowed}")
    return metrics


def _score_summary(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, float]:
    """Aggregate numeric RAGAS scores from row dicts.

    调用顺序：命令行入口 -> _score_summary()。
    """
    summary: dict[str, float] = {}
    for metric in metric_names:
        values = []
        for row in rows:
            value = row.get(metric)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            summary[f"avg_{metric}"] = round(sum(values) / len(values), 4)
    return summary


def result_rows(result: Any) -> list[dict[str, Any]]:
    """Convert a RAGAS EvaluationResult to JSON rows across supported versions.

    调用顺序：命令行入口 -> result_rows()。
    """
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        return json.loads(frame.to_json(orient="records", force_ascii=False))
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
        if isinstance(payload, dict) and "scores" in payload:
            return list(payload["scores"])
        if isinstance(payload, list):
            return payload
    return []


def evaluate_with_ragas(args: argparse.Namespace) -> dict[str, Any]:
    """Run RAGAS on an existing local engineering evaluation report.

    调用顺序：命令行入口 -> evaluate_with_ragas()。
    """
    ragas_module, metrics_module, llms_module, embeddings_module = load_ragas_runtime()
    from qa_core.llm.client import get_chat_model
    from qa_core.retrieval.models import get_embeddings

    report_path = project_path(args.report)
    report = read_json_file(report_path)
    ragas_rows, skipped = build_ragas_rows(report, args.limit, args.max_contexts)
    if not ragas_rows:
        raise RuntimeError(
            "没有可用于 RAGAS 的样本。请先用新版 evaluate_core_chain.py 生成包含 answer 和 sources 的报告。"
        )

    include_reference_metrics = all("reference" in row for row in ragas_rows)
    metrics = selected_metrics(
        metrics_module,
        [item.strip() for item in args.metrics.split(",") if item.strip()],
        include_reference_metrics=include_reference_metrics,
    )

    llm = llms_module.LangchainLLMWrapper(get_chat_model(streaming=False))
    embeddings = embeddings_module.LangchainEmbeddingsWrapper(get_embeddings())
    dataset = ragas_module.EvaluationDataset.from_list(ragas_rows)
    result = ragas_module.evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        batch_size=args.batch_size or None,
        show_progress=not args.no_progress,
    )
    rows = result_rows(result)
    metric_names = [str(getattr(metric, "name", "")) for metric in metrics if getattr(metric, "name", "")]

    return {
        "report_type": "ragas_supplemental_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str(report_path),
        "source_report_type": report.get("report_type"),
        "dataset": report.get("dataset"),
        "total_source_rows": len(report.get("rows") or []),
        "evaluated_rows": len(ragas_rows),
        "skipped_rows": skipped,
        "metrics": metric_names,
        "summary": _score_summary(rows, metric_names),
        "rows": rows,
        "note": (
            "RAGAS 是补充语义质量评测，不替代 check_evaluation_gate.py。"
            "工程门禁仍以 Recall@K、MRR、hit_type、source 推断、Prompt Profile、场景隔离和错误率为准。"
        ),
    }


def default_output_path(report_path: str | Path) -> Path:
    """Build the default RAGAS report path.

    调用顺序：命令行入口 -> default_output_path()。
    """
    RAGAS_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = Path(report_path).stem or "core_chain"
    return RAGAS_REPORT_DIR / f"{stamp}_{name}_ragas.json"


def main() -> None:
    """命令行入口：解析参数并执行 RAGAS 补充评估流程。

    调用顺序：命令行入口 -> main()。
    """
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Run RAGAS supplemental evaluation on a local core-chain report.")
    parser.add_argument("--report", required=True, help="Existing evaluate_core_chain.py JSON report.")
    parser.add_argument("--limit", type=int, default=10, help="Max rows to evaluate with RAGAS.")
    parser.add_argument("--max-contexts", type=int, default=6, help="Max contexts per case.")
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy",
        help=(
            "Comma-separated RAGAS metrics. Default: faithfulness,answer_relevancy. "
            "Optional non-reference metrics: context_relevance,response_groundedness. "
            "Reference metrics context_precision/context_recall require every row to provide reference/ground_truth/expected_answer."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=0, help="Optional RAGAS batch size.")
    parser.add_argument("--no-progress", action="store_true", help="Disable RAGAS progress bar.")
    parser.add_argument("--output", default="", help="Output JSON path.")
    args = parser.parse_args()

    payload = evaluate_with_ragas(args)
    output = project_path(args.output) if args.output else default_output_path(args.report)
    write_json_file(output, payload)
    print_json(payload)


if __name__ == "__main__":
    main()
