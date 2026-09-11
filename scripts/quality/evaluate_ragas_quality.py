"""RAGAS 补充语义质量评测。

本脚本是本地工程门禁之外的补充：

- `evaluate_core_chain.py` / `check_evaluation_gate.py` 仍是 Recall@K、MRR、hit_type、
  来源推断、Prompt Profile 路由、场景隔离、DataScope 与延迟的主回归门禁。
- 本脚本用 LLM-as-judge 增加 faithfulness（忠实度）、answer relevancy（答案相关性）
  等语义检查，只用于离线诊断或发布证据，不作为唯一 CI 门禁。

典型流程：

    python scripts/evaluate_core_chain.py --dataset eval_sets/multi_scenario_smoke.json --limit 20 --output reports/evaluation/core_chain_latest.json
    python scripts/quality/check_evaluation_gate.py --report reports/evaluation/core_chain_latest.json
    python scripts/quality/evaluate_ragas_quality.py --report reports/evaluation/core_chain_latest.json --limit 10
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common import PROJECT_ROOT, configure_utf8_stdio, print_json, read_json_file, write_json_file


RAGAS_REPORT_DIR = PROJECT_ROOT / "reports" / "evaluation"


def project_path(path: str | Path) -> Path:
    """把命令行路径解析到项目根目录下。

    参数：
        path: 命令行传入的路径；绝对路径原样返回，相对路径按项目根目录拼接。

    返回：
        解析后的 Path 对象。

    调用顺序：evaluate_with_ragas() / main() -> project_path()。
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_ragas_runtime():
    """仅在真正执行本补充脚本时导入 RAGAS。

    原因：RAGAS 是重依赖，按需导入可避免污染其它轻量脚本的模块加载路径。

    参数：
        无。

    返回：
        (ragas 模块, metrics 模块, llms 模块, embeddings 模块) 四元组。

    异常：
        RuntimeError: 缺少 RAGAS 或其子模块。

    调用顺序：evaluate_with_ragas() -> load_ragas_runtime()。
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
    """从来源负载中提取 RAGAS 上下文文本。

    优先取 content；为空时退回 metadata 中的 standard_question / answer / file_name /
    file_path 拼接值，保证上下文非空。

    参数：
        source: 来源负载字典，含 content 与 metadata。

    返回：
        提取出的上下文字符串。

    调用顺序：build_ragas_rows() -> _source_content()。
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
    """从评测样本的显式期望中构建轻量参考文本。

    优先级：reference / ground_truth / expected_answer，再退回 expected_keywords 拼接。

    参数：
        row: 评测行字典。

    返回：
        参考文本；没有可用期望时返回空字符串。

    调用顺序：build_ragas_rows() -> _reference_from_row()。
    """
    explicit = row.get("reference") or row.get("ground_truth") or row.get("expected_answer")
    if explicit:
        return str(explicit)
    keywords = [str(item).strip() for item in row.get("expected_keywords") or [] if str(item).strip()]
    if keywords:
        return "；".join(keywords)
    return ""


def build_ragas_rows(report: dict[str, Any], limit: int, max_contexts: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把本地工程评测报告行转换为 RAGAS 单轮样本。

    每行转成一个含 user_input / response / retrieved_contexts 的样本；缺完整答案
    或缺上下文的行记入 skipped，不参与 RAGAS 评测。

    参数：
        report: 本地评测报告字典，含 rows 列表。
        limit: 最多转换的样本数。
        max_contexts: 每个样本最多携带的上下文条数。

    返回：
        (RAGAS 样本列表, 被跳过行的原因列表)。

    调用顺序：evaluate_with_ragas() -> build_ragas_rows() -> _source_content() / _reference_from_row()。
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
    """按项目稳定名称创建 RAGAS 指标实例。

    无参考类指标始终可选；带参考文本的指标（context_precision / context_recall）仅在
    include_reference_metrics 为真时开放，因为它们要求每个样本都提供 reference。

    参数：
        metrics_module: ragas.metrics 模块。
        metric_names: 指标名称列表，逐个归一化后查表。
        include_reference_metrics: 是否允许带参考文本的指标。

    返回：
        指标实例列表。

    异常：
        ValueError: 包含未知指标名。

    调用顺序：evaluate_with_ragas() -> selected_metrics()。
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
    """从评测行字典汇总各指标的平均分。

    参数：
        rows: RAGAS 结果行列表。
        metric_names: 指标名列表，只汇总存在数值的行。

    返回：
        {avg_{metric}: 平均分} 字典；某指标无任何数值时不出现在结果中。

    调用顺序：evaluate_with_ragas() -> _score_summary()。
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
    """把 RAGAS EvaluationResult 转成跨版本兼容的 JSON 行列表。

    优先走 to_pandas()，退回 to_dict() 的 scores / list 两种形态，均不支持时返回空列表。

    参数：
        result: RAGAS 评测结果对象。

    返回：
        可直接 JSON 序列化的行列表。

    调用顺序：evaluate_with_ragas() -> result_rows()。
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
    """在已有本地评测报告上运行 RAGAS 补充评测。（★★★ 核心）

    流程：读取报告 -> 转换样本 -> 创建指标 -> 包装 LLM/Embedding -> 执行评测 ->
    汇总平均分并输出报告字典。

    参数：
        args: 命令行参数（report / limit / max_contexts / metrics / batch_size / no_progress）。

    返回：
        RAGAS 补充评测报告字典。

    异常：
        RuntimeError: 报告中没有可用于 RAGAS 的样本。

    调用顺序：main() -> evaluate_with_ragas() -> load_ragas_runtime() / build_ragas_rows() /
        selected_metrics() / result_rows() / _score_summary()。
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
    """构建默认的 RAGAS 报告输出路径。

    按 {时间戳}_{源报告名}_ragas.json 命名，放在 reports/evaluation 目录下。

    参数：
        report_path: 源评测报告路径，用于取文件名。

    返回：
        默认输出 Path 对象（目录不存在时会创建）。

    调用顺序：main() -> default_output_path()。
    """
    RAGAS_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = Path(report_path).stem or "core_chain"
    return RAGAS_REPORT_DIR / f"{stamp}_{name}_ragas.json"


def main() -> None:
    """命令行入口：解析参数并执行 RAGAS 补充评估流程。

    参数：
        无（参数在函数内解析）。

    返回：
        无；报告写入输出文件并打印到 stdout。

    调用顺序：命令行入口 -> main() -> evaluate_with_ragas() / default_output_path() / write_json_file() / print_json()。
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
