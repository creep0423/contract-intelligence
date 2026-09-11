"""主问答链路评测门禁。

评测报告负责量化 RAG 主链路表现，评测门禁负责把这些指标变成可执行的通过/失败
标准。它支持两种运行模式：

1. 检查已有报告：只读取 JSON 报告并判断阈值，不加载 QAService、Milvus、Embedding 或 LLM。
2. 现场执行评测：先调用 `scripts.evaluate_core_chain` 跑评测集，再对新报告做门禁判断。

为什么要区分两种模式：
- 线上或 CI 中常见做法是“先产出评测报告，再单独跑门禁”，此时门禁脚本不应该依赖
  LangChain/Milvus/模型包；
- 本地联调时也可以直接让本脚本现场跑评测，但这条路径需要完整 RAG 运行环境。

和入库质量门禁的区别：
- 入库质量门禁关注“资料有没有被正确解析、切分、版本化”；
- 评测门禁关注“真实问题经过意图识别、检索、重排、Prompt 和流式生成后是否稳定”。

用法示例：

    # 第一步：生成评测报告；这条命令需要完整 RAG 运行环境
    python scripts/evaluate_core_chain.py --dataset eval_sets/multi_scenario_smoke.json --limit 20 --output reports/evaluation/core_chain_latest.json

    # 第二步：检查已有评测报告；这条命令不要求本机安装 langchain-milvus
    python scripts/quality/check_evaluation_gate.py --report reports/evaluation/core_chain_latest.json

    # 检查已有报告，并把门禁结果另存为 JSON
    python scripts/quality/check_evaluation_gate.py --report reports/evaluation/core_chain_latest.json --gate-output reports/verification/evaluation_gate_latest.json

    # 现场执行评测集再检查；这条命令需要完整 RAG 依赖、Milvus、MySQL 和模型配置可用
    python scripts/quality/check_evaluation_gate.py --dataset eval_sets/multi_scenario_smoke.json --limit 20

    # 临时调整阈值，常用于定位某一类指标是否刚好卡线
    python scripts/quality/check_evaluation_gate.py --report reports/evaluation/core_chain_latest.json --min-recall-at-k 0.9 --max-error-rate 0
"""

from __future__ import annotations

# argparse: 命令行参数解析
import argparse

# importlib: 按需加载现场评测模块，避免 --report/--help 模式提前加载完整 RAG 运行依赖
import importlib

# os: 切换工作目录到项目根，保证 IDE/命令行从任意目录启动时相对路径稳定
import os

# sys: 系统功能（sys.path + sys.exit）
import sys

# dataclasses: 数据类定义（EvaluationGateThresholds）
from dataclasses import asdict, dataclass

# pathlib.Path: 文件路径操作
from pathlib import Path

# textwrap: argparse 示例文本缩进清理
import textwrap

# typing.Any: 任意类型
from typing import Any

# ── 路径设置 ──
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── 导入公共模块 ──
from scripts.common import PROJECT_ROOT, print_json, read_json_file, write_json_file, write_optional_json
from scripts.gate_utils import add_max_failure, add_min_failure


def project_path(path: str | Path) -> Path:
    """把命令行传入的路径解析为项目内路径。

    评测脚本经常用绝对脚本路径启动，例如：

        python D:/workspace/contract-intelligence/scripts/quality/check_evaluation_gate.py --dataset eval_sets/xxx.json

    这种情况下，`eval_sets/xxx.json` 应该指向项目目录下的文件，而不是 IDE 当前工作目录。
    因此只要用户传的是相对路径，就统一拼到 PROJECT_ROOT 下；绝对路径则保持不变。

    调用顺序：命令行入口 -> project_path()。
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def ensure_project_workdir() -> None:
    """把当前进程工作目录固定到项目根目录。

    QAService、场景配置、评测集和报告目录都采用项目相对路径。显式切到项目根目录后，
    无论脚本从 PowerShell、PyCharm 还是其他目录启动，相对路径含义都一致。

    调用顺序：命令行入口 -> ensure_project_workdir()。
    """
    os.chdir(PROJECT_ROOT)


def load_live_evaluator() -> tuple[Any, Any]:
    """按需加载现场评测函数。

    `check_evaluation_gate.py --report ...` 的职责是检查一个已经存在的评测报告，它只需要
    JSON 读写和阈值判断，不应该因为本机缺少 `langchain_milvus`、Embedding/Reranker 或 LLM
    依赖而无法启动。

    当用户没有传 `--report` 时，脚本才进入“现场执行评测”模式；这时需要加载
    `scripts.evaluate_core_chain`，它会进一步加载 QAService、Milvus 检索、Prompt 和模型调用。
    如果运行环境缺依赖，这里给出明确错误，避免用户误以为门禁逻辑本身坏了。

    调用顺序：命令行入口 -> load_live_evaluator()。
    """
    try:
        module = importlib.import_module("scripts.evaluate_core_chain")
    except ModuleNotFoundError as exc:
        missing_module = exc.name or "unknown"
        raise RuntimeError(
            "现场执行评测需要完整 RAG 运行依赖。"
            "当前环境缺少模块："
            f"{missing_module}。"
            "如果只是检查已有评测报告，请使用："
            "python scripts/quality/check_evaluation_gate.py --report reports/evaluation/你的报告.json"
        ) from exc
    return module.default_output_path, module.evaluate_dataset


@dataclass(frozen=True)
class EvaluationGateThresholds:
    """主链路评测门禁阈值。

    默认值适合 smoke 级别评测：必须无错误，召回、关键词覆盖和场景隔离不能明显退化。
    如果后续扩展成大规模评测集，可以在 CI 中按环境调整这些阈值。

    调用顺序：命令行入口 -> EvaluationGateThresholds。
    """

    # ── 全局指标阈值（跨所有场景/source 的聚合值）──
    min_recall_at_k: float = 0.8               # 预期来源召回率下限（Recall@K）
    min_mrr: float = 0.6                        # 预期来源平均倒数排名下限（MRR）
    min_keyword_coverage: float = 0.7           # 答案关键事实覆盖率下限
    min_hit_type_accuracy: float = 0.7          # 命中路径（FAQ直出/RAG/边界）判断准确率下限
    min_source_inference_accuracy: float = 0.7  # source 自动推断准确率下限
    min_prompt_profile_accuracy: float = 0.7    # Prompt Profile 路由准确率下限
    min_faq_direct_accuracy: float = 0.7        # FAQ 标准问答直出准确率下限
    min_scenario_isolation_accuracy: float = 1.0 # 多场景隔离准确率下限（默认必须 100%）
    min_avg_answer_confidence_score: float = 0.0 # 最终答案置信度均值下限（默认只观察，不阻断）
    max_error_rate: float = 0.0                 # 错误率上限（默认不允许任何样本出错）
    max_avg_elapsed_ms: float = 60000.0         # 平均耗时上限（毫秒）

    # ── 按场景分组的指标阈值（防止局部退化被全局均值掩盖）──
    min_scenario_recall_at_k: float = 0.8       # 单场景召回率下限
    min_scenario_mrr: float = 0.6                # 单场景 MRR 下限
    min_scenario_keyword_coverage: float = 0.7   # 单场景关键事实覆盖率下限
    max_scenario_error_rate: float = 0.0         # 单场景错误率上限

    # ── 按 source 分组的指标阈值（防止某类资料召回退化被掩盖）──
    min_source_recall_at_k: float = 0.8          # 单 source 召回率下限
    min_source_mrr: float = 0.6                   # 单 source MRR 下限
    min_source_keyword_coverage: float = 0.7      # 单 source 关键事实覆盖率下限
    max_source_error_rate: float = 0.0            # 单 source 错误率上限

    # ── 按命中类型分组的指标阈值 ──
    min_hit_type_group_accuracy: float = 0.7     # 单命中路径（FAQ/RAG/边界）准确率下限


def load_evaluation_report(path: str | Path) -> dict[str, Any]:
    """读取评测报告 JSON。

    调用顺序：命令行入口 -> load_evaluation_report()。
    """
    report_path = project_path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"评测报告不存在：{report_path}")
    return read_json_file(report_path)


def recent_evaluation_reports(limit: int = 8) -> list[str]:
    """列出最近生成的评测报告路径，便于 `--report` 路径写错时给出提示。

    调用顺序：命令行入口 -> recent_evaluation_reports()。
    """
    report_dir = PROJECT_ROOT / "reports" / "evaluation"
    if not report_dir.exists():
        return []
    reports = sorted(report_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    return [str(path.relative_to(PROJECT_ROOT)) for path in reports[:limit]]


def available_eval_sets() -> list[str]:
    """返回项目中可用的评测集文件列表。

    调用顺序：命令行入口 -> available_eval_sets()。
    """
    eval_dir = PROJECT_ROOT / "eval_sets"
    if not eval_dir.exists():
        return []
    return [str(path.relative_to(PROJECT_ROOT)) for path in sorted(eval_dir.glob("*.json"))]


def _metric(report: dict[str, Any], key: str, default: float = 0.0) -> float:
    """安全读取数值指标。

    调用顺序：命令行入口 -> _metric()。
    """
    value = report.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _group_key(value: Any) -> str:
    """规范化分组名称，避免空值把分组指标写得不清楚。

    调用顺序：命令行入口 -> _group_key()。
    """
    return str(value or "").strip() or "unknown"


def _truthy(value: Any) -> bool:
    """把评测行里的布尔字段转成明确 bool。

    调用顺序：命令行入口 -> _truthy()。
    """
    return value is True or str(value).lower() == "true"


def _matched_value(row: dict[str, Any], match_field: str, expected_field: str, actual_field: str) -> bool:
    """读取评测脚本写入的明细匹配结果。

    这里要求评测报告显式写出 `*_matched` 字段。原因是门禁脚本负责验收“当前标准”
    下的报告质量，不再根据 expected/actual 临时反推结果，避免同时维护两套报告格式。
    `expected_field` 和 `actual_field` 只保留在函数签名中，用于让调用处能直接看出该匹配项的业务含义。

    调用顺序：命令行入口 -> _matched_value()。
    """
    _ = (expected_field, actual_field)
    return _truthy(row.get(match_field))


def _rows_group_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """从评测明细行计算一个分组的质量指标。

    使用场景：
    - 按场景看某个业务包是否退化；
    - 按 source 看某类资料是否召回变差；
    - 按 expected_hit_type 看 FAQ 直出、RAG、边界识别是否各自稳定。

    为什么不只看全局均值：真实项目里某个小场景坏掉时，全局 Recall@K 可能仍然很好。
    分组门禁可以把这种"局部退化被平均值掩盖"的问题直接暴露出来。

    调用顺序：命令行入口 -> _rows_group_metrics()。
    """
    total = len(rows)
    # 筛选有对应评测标注的样本子集
    source_rows = [row for row in rows if row.get("source_recall_hit") is not None]       # 有预期来源标注的样本
    hit_type_rows = [row for row in rows if row.get("expected_hit_type")]                   # 有预期命中类型标注的样本
    source_inference_rows = [row for row in rows if row.get("expected_effective_source")]   # 有预期 source 标注的样本
    prompt_profile_rows = [row for row in rows if row.get("expected_prompt_profile")]       # 有预期 Prompt Profile 标注的样本
    keyword_rows = [row for row in rows if "keyword_coverage" in row]                       # 有关键词覆盖率的样本
    confidence_rows = [row for row in rows if row.get("answer_confidence_score") is not None] # 有最终答案置信度的样本
    errors = sum(1 for row in rows if row.get("error"))

    return {
        "total": total,
        "errors": errors,
        "error_rate": round(errors / max(total, 1), 4),
        # Recall@K：预期来源是否被检索到（仅统计有预期来源标注的样本）
        "recall_at_k": round(
            sum(1 for row in source_rows if _truthy(row.get("source_recall_hit"))) / max(len(source_rows), 1),
            4,
        ) if source_rows else 1.0,
        # MRR：预期来源排在第几位（排名越靠前越好）
        "mrr": round(
            sum(float(row.get("mrr") or 0.0) for row in source_rows) / max(len(source_rows), 1),
            4,
        ) if source_rows else 1.0,
        # 关键词覆盖率：答案中包含了多少预期关键事实
        "avg_keyword_coverage": round(
            sum(float(row.get("keyword_coverage") or 0.0) for row in keyword_rows) / max(len(keyword_rows), 1),
            4,
        ) if keyword_rows else 1.0,
        # 答案置信度：最终答案层的可解释工程信号，默认只作为观察项
        "avg_answer_confidence_score": round(
            sum(float(row.get("answer_confidence_score") or 0.0) for row in confidence_rows) / max(len(confidence_rows), 1),
            4,
        ) if confidence_rows else 0.0,
        # hit_type 准确率：FAQ 直出 / RAG / 边界识别是否路由正确
        "hit_type_accuracy": round(
            sum(1 for row in hit_type_rows if _matched_value(row, "hit_type_matched", "expected_hit_type", "hit_type")) / max(len(hit_type_rows), 1),
            4,
        ) if hit_type_rows else 1.0,
        # source 推断准确率：无分类条件下的 source 自动推断是否稳定
        "source_inference_accuracy": round(
            sum(
                1
                for row in source_inference_rows
                if _matched_value(row, "source_inference_matched", "expected_effective_source", "effective_source_filter")
            ) / max(len(source_inference_rows), 1),
            4,
        ) if source_inference_rows else 1.0,
        # Prompt Profile 路由准确率：高风险问题是否进入正确模板
        "prompt_profile_accuracy": round(
            sum(
                1
                for row in prompt_profile_rows
                if _matched_value(row, "prompt_profile_matched", "expected_prompt_profile", "prompt_profile")
            ) / max(len(prompt_profile_rows), 1),
            4,
        ) if prompt_profile_rows else 1.0,
    }


def _derive_group_metrics(report: dict[str, Any]) -> dict[str, dict[str, dict[str, float | int]]]:
    """从评测报告中派生场景、source 和 hit_type 分组指标。

    分组逻辑：
      - 优先按评测明细行（rows）的字段做实时分组（scenario_id / expected_effective_source / expected_hit_type）
      - 如果明细行为空（历史报告可能只保存了汇总值），则 fallback 到报告中已保存的 scenario_metrics

    调用顺序：命令行入口 -> _derive_group_metrics()。
    """
    rows = [dict(row) for row in list(report.get("rows") or [])]
    scenario_metrics: dict[str, dict[str, float | int]] = {}
    source_metrics: dict[str, dict[str, float | int]] = {}
    hit_type_metrics: dict[str, dict[str, float | int]] = {}

    if rows:
        # ── 有明细行时：实时计算分组指标 ──
        scenario_groups: dict[str, list[dict[str, Any]]] = {}
        source_groups: dict[str, list[dict[str, Any]]] = {}
        hit_type_groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            # 按场景分组
            scenario_groups.setdefault(_group_key(row.get("scenario_id")), []).append(row)
            # 按 source 分组（优先用 expected_effective_source，fallback 到 source_filter）
            source_name = row.get("expected_effective_source") or row.get("source_filter")
            if source_name:
                source_groups.setdefault(_group_key(source_name), []).append(row)
            # 按命中类型分组
            expected_hit_type = row.get("expected_hit_type")
            if expected_hit_type:
                hit_type_groups.setdefault(_group_key(expected_hit_type), []).append(row)
        scenario_metrics = {name: _rows_group_metrics(items) for name, items in sorted(scenario_groups.items())}
        source_metrics = {name: _rows_group_metrics(items) for name, items in sorted(source_groups.items())}
        hit_type_metrics = {name: _rows_group_metrics(items) for name, items in sorted(hit_type_groups.items())}
    else:
        # ── 无明细行时：fallback 到报告中已保存的汇总值 ──
        for name, metrics in dict(report.get("scenario_metrics") or {}).items():
            total = int(metrics.get("total") or 0)
            errors = int(metrics.get("errors") or 0)
            scenario_metrics[str(name)] = {
                "total": total,
                "errors": errors,
                "error_rate": round(errors / max(total, 1), 4),
                "recall_at_k": _metric(metrics, "recall_at_k", 1.0),
                "mrr": _metric(metrics, "mrr", 1.0),
                "avg_keyword_coverage": _metric(metrics, "avg_keyword_coverage", 1.0),
                "avg_answer_confidence_score": _metric(metrics, "avg_answer_confidence_score", 0.0),
            }
    return {
        "scenario_metrics": scenario_metrics,
        "source_metrics": source_metrics,
        "hit_type_metrics": hit_type_metrics,
    }


def _add_group_min_failure(
    failures: list[dict[str, Any]],
    *,
    group_type: str,
    group_name: str,
    metric: str,
    actual: float,
    minimum: float,
    message: str,
) -> None:
    """追加分组最小值门禁失败。

    调用顺序：命令行入口 -> _add_group_min_failure()。
    """
    add_min_failure(
        failures,
        metric=f"{group_type}.{group_name}.{metric}",
        actual=actual,
        minimum=minimum,
        message=message,
    )


def _add_group_max_failure(
    failures: list[dict[str, Any]],
    *,
    group_type: str,
    group_name: str,
    metric: str,
    actual: float,
    maximum: float,
    message: str,
) -> None:
    """追加分组最大值门禁失败。

    调用顺序：命令行入口 -> _add_group_max_failure()。
    """
    add_max_failure(
        failures,
        metric=f"{group_type}.{group_name}.{metric}",
        actual=actual,
        maximum=maximum,
        message=message,
    )


def add_group_failures(
    failures: list[dict[str, Any]],
    group_metrics: dict[str, dict[str, dict[str, float | int]]],
    thresholds: EvaluationGateThresholds,
) -> None:
    """检查场景、source、hit_type 三类分组门禁。

    调用顺序：命令行入口 -> add_group_failures()。
    """
    for scenario, metrics in group_metrics["scenario_metrics"].items():
        _add_group_max_failure(
            failures,
            group_type="scenario",
            group_name=scenario,
            metric="error_rate",
            actual=float(metrics.get("error_rate") or 0.0),
            maximum=thresholds.max_scenario_error_rate,
            message="该业务场景出现错误，不能被全局错误率掩盖。",
        )
        _add_group_min_failure(
            failures,
            group_type="scenario",
            group_name=scenario,
            metric="recall_at_k",
            actual=float(metrics.get("recall_at_k") or 0.0),
            minimum=thresholds.min_scenario_recall_at_k,
            message="该业务场景召回退化，需要检查该场景资料、source 规则或知识库版本。",
        )
        _add_group_min_failure(
            failures,
            group_type="scenario",
            group_name=scenario,
            metric="mrr",
            actual=float(metrics.get("mrr") or 0.0),
            minimum=thresholds.min_scenario_mrr,
            message="该业务场景预期来源排名靠后，需要检查重排或 chunk 粒度。",
        )
        _add_group_min_failure(
            failures,
            group_type="scenario",
            group_name=scenario,
            metric="avg_keyword_coverage",
            actual=float(metrics.get("avg_keyword_coverage") or 0.0),
            minimum=thresholds.min_scenario_keyword_coverage,
            message="该业务场景答案关键事实覆盖不足。",
        )

    for source, metrics in group_metrics["source_metrics"].items():
        _add_group_max_failure(
            failures,
            group_type="source",
            group_name=source,
            metric="error_rate",
            actual=float(metrics.get("error_rate") or 0.0),
            maximum=thresholds.max_source_error_rate,
            message="该 source 资料链路出现错误，可能是分类过滤或资料版本问题。",
        )
        _add_group_min_failure(
            failures,
            group_type="source",
            group_name=source,
            metric="recall_at_k",
            actual=float(metrics.get("recall_at_k") or 0.0),
            minimum=thresholds.min_source_recall_at_k,
            message="该 source 召回退化，需要检查 source 推断、过滤条件和资料覆盖。",
        )
        _add_group_min_failure(
            failures,
            group_type="source",
            group_name=source,
            metric="mrr",
            actual=float(metrics.get("mrr") or 0.0),
            minimum=thresholds.min_source_mrr,
            message="该 source 的预期来源排名靠后，需要检查 dense/sparse 权重或 rerank。",
        )
        _add_group_min_failure(
            failures,
            group_type="source",
            group_name=source,
            metric="avg_keyword_coverage",
            actual=float(metrics.get("avg_keyword_coverage") or 0.0),
            minimum=thresholds.min_source_keyword_coverage,
            message="该 source 的答案关键事实覆盖不足。",
        )

    for hit_type, metrics in group_metrics["hit_type_metrics"].items():
        _add_group_min_failure(
            failures,
            group_type="hit_type",
            group_name=hit_type,
            metric="hit_type_accuracy",
            actual=float(metrics.get("hit_type_accuracy") or 0.0),
            minimum=thresholds.min_hit_type_group_accuracy,
            message="该命中路径不稳定，FAQ/RAG/边界识别可能被路由到错误链路。",
        )


def evaluate_report_against_gate(
    report: dict[str, Any],
    thresholds: EvaluationGateThresholds,
    *,
    report_path: str = "",
) -> dict[str, Any]:
    """根据评测指标判断主链路是否通过门禁。

    调用顺序：命令行入口 -> evaluate_report_against_gate()。
    """
    total = int(report.get("total") or 0)
    errors = int(report.get("errors") or 0)
    error_rate = round(errors / max(total, 1), 4)
    metrics = {
        "total": total,
        "errors": errors,
        "error_rate": error_rate,
        "recall_at_k": _metric(report, "recall_at_k"),
        "mrr": _metric(report, "mrr"),
        "avg_keyword_coverage": _metric(report, "avg_keyword_coverage"),
        "avg_answer_confidence_score": _metric(report, "avg_answer_confidence_score"),
        "hit_type_accuracy": _metric(report, "hit_type_accuracy"),
        "source_inference_accuracy": _metric(report, "source_inference_accuracy", 1.0),
        "prompt_profile_accuracy": _metric(report, "prompt_profile_accuracy", 1.0),
        "faq_direct_accuracy": _metric(report, "faq_direct_accuracy"),
        "scenario_isolation_accuracy": _metric(report, "scenario_isolation_accuracy"),
        "avg_elapsed_ms": _metric(report, "avg_elapsed_ms"),
    }
    failures: list[dict[str, Any]] = []
    group_metrics = _derive_group_metrics(report)

    add_max_failure(
        failures,
        metric="error_rate",
        actual=metrics["error_rate"],
        maximum=thresholds.max_error_rate,
        message="评测样本出现错误，说明主链路依赖、检索或生成阶段不稳定。",
    )
    add_min_failure(
        failures,
        metric="recall_at_k",
        actual=metrics["recall_at_k"],
        minimum=thresholds.min_recall_at_k,
        message="预期来源召回不足，需要检查入库、query_variants、过滤条件或 top_k。",
    )
    add_min_failure(
        failures,
        metric="mrr",
        actual=metrics["mrr"],
        minimum=thresholds.min_mrr,
        message="预期来源排名靠后，需要检查 dense/sparse 权重、rerank 或 chunk 粒度。",
    )
    add_min_failure(
        failures,
        metric="avg_keyword_coverage",
        actual=metrics["avg_keyword_coverage"],
        minimum=thresholds.min_keyword_coverage,
        message="答案关键事实覆盖不足，需要检查上下文构建、Prompt 或模型输出。",
    )
    add_min_failure(
        failures,
        metric="avg_answer_confidence_score",
        actual=metrics["avg_answer_confidence_score"],
        minimum=thresholds.min_avg_answer_confidence_score,
        message="最终答案置信度均值低于阈值，需要结合 reasons 判断是召回、上下文还是入口规则分导致。",
    )
    add_min_failure(
        failures,
        metric="hit_type_accuracy",
        actual=metrics["hit_type_accuracy"],
        minimum=thresholds.min_hit_type_accuracy,
        message="FAQ 直出/RAG/信息不足路径判断不稳定。",
    )
    add_min_failure(
        failures,
        metric="source_inference_accuracy",
        actual=metrics["source_inference_accuracy"],
        minimum=thresholds.min_source_inference_accuracy,
        message="source 自动推断不稳定，可能影响无分类条件下的检索过滤。",
    )
    add_min_failure(
        failures,
        metric="prompt_profile_accuracy",
        actual=metrics["prompt_profile_accuracy"],
        minimum=thresholds.min_prompt_profile_accuracy,
        message="Prompt Profile 路由不稳定，高风险问题可能没有进入严格模板。",
    )
    add_min_failure(
        failures,
        metric="faq_direct_accuracy",
        actual=metrics["faq_direct_accuracy"],
        minimum=thresholds.min_faq_direct_accuracy,
        message="FAQ 标准问答直出能力退化。",
    )
    add_min_failure(
        failures,
        metric="scenario_isolation_accuracy",
        actual=metrics["scenario_isolation_accuracy"],
        minimum=thresholds.min_scenario_isolation_accuracy,
        message="多场景隔离评测不达标，可能出现跨场景误检索。",
    )
    add_max_failure(
        failures,
        metric="avg_elapsed_ms",
        actual=metrics["avg_elapsed_ms"],
        maximum=thresholds.max_avg_elapsed_ms,
        message="平均耗时超过门禁，需要检查模型、Milvus、rerank 或网络延迟。",
    )
    add_group_failures(failures, group_metrics, thresholds)

    return {
        "ok": not failures,
        "report_type": "core_chain_evaluation_gate",    # 报告类型。这里表示这是“主问答链路评测门禁报告”。
        "report_path": report_path, # 被检查的主问答链路评测报告路径。
        "dataset": report.get("dataset"),
        "created_at": report.get("created_at"),
        "metrics": metrics,
        "group_metrics": group_metrics,
        "thresholds": asdict(thresholds),
        "failures": failures,
    }


def thresholds_from_args(args: argparse.Namespace) -> EvaluationGateThresholds:
    """把命令行参数转换成评测门禁阈值。

    调用顺序：命令行入口 -> thresholds_from_args()。
    """
    return EvaluationGateThresholds(
        min_recall_at_k=args.min_recall_at_k,
        min_mrr=args.min_mrr,
        min_keyword_coverage=args.min_keyword_coverage,
        min_hit_type_accuracy=args.min_hit_type_accuracy,
        min_source_inference_accuracy=args.min_source_inference_accuracy,
        min_prompt_profile_accuracy=args.min_prompt_profile_accuracy,
        min_faq_direct_accuracy=args.min_faq_direct_accuracy,
        min_scenario_isolation_accuracy=args.min_scenario_isolation_accuracy,
        min_avg_answer_confidence_score=args.min_avg_answer_confidence_score,
        max_error_rate=args.max_error_rate,
        max_avg_elapsed_ms=args.max_avg_elapsed_ms,
        min_scenario_recall_at_k=args.min_scenario_recall_at_k,
        min_scenario_mrr=args.min_scenario_mrr,
        min_scenario_keyword_coverage=args.min_scenario_keyword_coverage,
        max_scenario_error_rate=args.max_scenario_error_rate,
        min_source_recall_at_k=args.min_source_recall_at_k,
        min_source_mrr=args.min_source_mrr,
        min_source_keyword_coverage=args.min_source_keyword_coverage,
        max_source_error_rate=args.max_source_error_rate,
        min_hit_type_group_accuracy=args.min_hit_type_group_accuracy,
    )


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    调用顺序：命令行入口 -> build_parser()。
    """
    parser = argparse.ArgumentParser(
        description="Check QA evaluation report against gate thresholds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            examples:
              # 第一步：先生成一份固定路径的评测报告；需要完整 RAG 运行环境
              python scripts/evaluate_core_chain.py --dataset eval_sets/multi_scenario_smoke.json --limit 20 --output reports/evaluation/core_chain_latest.json

              # 第二步：检查已有评测报告；不加载 QAService/Milvus/LLM
              python scripts/quality/check_evaluation_gate.py --report reports/evaluation/core_chain_latest.json

              # 检查已有报告，并输出门禁摘要
              python scripts/quality/check_evaluation_gate.py --report reports/evaluation/core_chain_latest.json --gate-output reports/verification/evaluation_gate_latest.json

              # 现场跑评测集再检查；需要完整 RAG 运行环境
              python scripts/quality/check_evaluation_gate.py --dataset eval_sets/multi_scenario_smoke.json --limit 20

              # 对某次报告临时收紧阈值
              python scripts/quality/check_evaluation_gate.py --report reports/evaluation/core_chain_latest.json --min-recall-at-k 0.9 --max-error-rate 0
            """
        ),
    )
    parser.add_argument("--report", default="", help="已有评测报告路径。未提供时现场执行评测集。")
    parser.add_argument("--dataset", default=str(Path("eval_sets") / "multi_scenario_smoke.json"), help="评测集 JSON。")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="", help="现场评测报告输出路径。")
    parser.add_argument("--gate-output", default="", help="门禁判定摘要输出路径。")
    parser.add_argument("--scenario", default=None, help="默认业务场景 ID。")
    parser.add_argument("--tenant-id", default=None, help="默认租户 ID。")
    parser.add_argument("--dataset-id", default=None, help="默认数据集 ID。")
    parser.add_argument("--visibility", default=None, help="默认可见级别。")
    parser.add_argument("--user-role", default=None, help="默认用户角色。")
    parser.add_argument("--kb-version", default=None, help="可选知识库版本。")
    parser.add_argument("--min-recall-at-k", type=float, default=0.8)
    parser.add_argument("--min-mrr", type=float, default=0.6)
    parser.add_argument("--min-keyword-coverage", type=float, default=0.7)
    parser.add_argument("--min-hit-type-accuracy", type=float, default=0.7)
    parser.add_argument("--min-source-inference-accuracy", type=float, default=0.7)
    parser.add_argument("--min-prompt-profile-accuracy", type=float, default=0.7)
    parser.add_argument("--min-faq-direct-accuracy", type=float, default=0.7)
    parser.add_argument("--min-scenario-isolation-accuracy", type=float, default=1.0)
    parser.add_argument(
        "--min-avg-answer-confidence-score",
        type=float,
        default=0.0,
        help="最终答案置信度均值下限。默认 0 表示只记录观察，不作为阻断门禁。",
    )
    parser.add_argument("--max-error-rate", type=float, default=0.0)
    parser.add_argument("--max-avg-elapsed-ms", type=float, default=60000.0)
    parser.add_argument("--min-scenario-recall-at-k", type=float, default=0.8)
    parser.add_argument("--min-scenario-mrr", type=float, default=0.6)
    parser.add_argument("--min-scenario-keyword-coverage", type=float, default=0.7)
    parser.add_argument("--max-scenario-error-rate", type=float, default=0.0)
    parser.add_argument("--min-source-recall-at-k", type=float, default=0.8)
    parser.add_argument("--min-source-mrr", type=float, default=0.6)
    parser.add_argument("--min-source-keyword-coverage", type=float, default=0.7)
    parser.add_argument("--max-source-error-rate", type=float, default=0.0)
    parser.add_argument("--min-hit-type-group-accuracy", type=float, default=0.7)
    return parser


def main() -> None:
    """执行评测门禁并按结果设置退出码。

    调用顺序：命令行入口 -> main()。
    """
    ensure_project_workdir()
    parser = build_parser()
    args = parser.parse_args()

    if args.report:
        # ===== 模式一：检查已有评测报告 =====
        # 只读 JSON → 做门禁判断，不加载 QAService/Milvus/LLM。
        # 适合 CI 或离线验收已有报告，不要求本机具备完整 RAG 运行环境。
        report_path = args.report
        try:
            report = load_evaluation_report(report_path)
        except FileNotFoundError as exc:
            # 报告不存在时，列出最近可用的报告路径帮助用户定位
            print_json(
                {
                    "ok": False,
                    "report_type": "core_chain_evaluation_gate",
                    "mode": "existing_report",
                    "error": str(exc),
                    "available_reports": recent_evaluation_reports(),
                    "next_step": "先生成评测报告，或把 --report 改成一个已经存在的报告路径。",
                    "generate_example": "python scripts/evaluate_core_chain.py --dataset eval_sets/multi_scenario_smoke.json --limit 20 --output reports/evaluation/core_chain_latest.json",
                    "check_example": "python scripts/quality/check_evaluation_gate.py --report reports/evaluation/core_chain_latest.json",
                }
            )
            sys.exit(1)
    else:
        # ===== 模式二：现场执行评测再判断 =====
        # 需要完整 RAG 运行依赖（Milvus、Embedding、Reranker、LLM）。
        # 通过 importlib 按需加载 evaluate_core_chain.py，避免模式一被依赖缺失阻断。
        dataset_path = project_path(args.dataset)
        if not dataset_path.exists():
            # 评测集不存在时列出可用评测集帮助用户选择
            print_json(
                {
                    "ok": False,
                    "report_type": "core_chain_evaluation_gate",
                    "mode": "live_evaluation",
                    "error": f"评测集不存在：{dataset_path}",
                    "available_eval_sets": available_eval_sets(),
                    "next_step": "把 --dataset 改成 available_eval_sets 中的路径，或先补充对应评测集文件。",
                    "example": "python scripts/quality/check_evaluation_gate.py --dataset eval_sets/business_depth_regression.json --limit 20 --output reports/evaluation/core_chain_latest.json",
                }
            )
            sys.exit(1)
        try:
            # 按需加载评测模块（包含 QAService → Milvus → LLM 依赖链）
            default_output_path, evaluate_dataset = load_live_evaluator()
        except RuntimeError as exc:
            print_json(
                {
                    "ok": False,
                    "report_type": "core_chain_evaluation_gate",
                    "mode": "live_evaluation",
                    "error": str(exc),
                    "next_step": "安装完整 RAG 运行依赖后重试；如果已有评测报告，请改用 --report 检查报告。",
                    "example": "python scripts/quality/check_evaluation_gate.py --report reports/evaluation/core_chain_latest.json",
                }
            )
            sys.exit(1)
        # 执行评测并保存报告
        output_path = Path(args.output) if args.output else default_output_path(args.dataset)
        report = evaluate_dataset(args)
        report_path = str(output_path)
        write_json_file(output_path, report)

    # ===== 统一的门禁判断 =====
    # 两种模式最终都走到同一个 evaluate_report_against_gate，保证口径一致
    result = evaluate_report_against_gate(report, thresholds_from_args(args), report_path=report_path)
    write_optional_json(args.gate_output, result)
    print_json(result)
    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
