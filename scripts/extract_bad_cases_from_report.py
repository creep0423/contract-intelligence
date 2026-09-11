"""从本地评测报告中提取 Bad Case，并沉淀为可复跑的评测集。

本脚本解决的是第 17 讲里的学习路径问题：没有 LangSmith 也能完成
"发现失败样本 -> 人工补充期望字段 -> 进入 eval_sets -> 重新评测 -> Gate 阻断"。

用法示例：

    python scripts/extract_bad_cases_from_report.py \
      --report reports/evaluation/core_chain_latest.json \
      --output eval_sets/local_bad_cases.json

输出文件是普通 `eval_sets/*.json` 数组，可直接传给 `evaluate_core_chain.py`。
"""

from __future__ import annotations

# argparse: 命令行参数解析
import argparse

# json: 读取原始评测集和写出本地 bad case 数据集
import json

# pathlib.Path: 文件路径处理
from pathlib import Path

# sys: 将项目根目录加入导入路径，兼容 python scripts/xxx.py 直接启动
import sys

# typing.Any: 评测报告中的行字段类型不固定，统一使用 Any
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import PROJECT_ROOT, configure_utf8_stdio, print_json
from scripts.eval_common import resolve_eval_dataset_path


MATCH_FIELDS = (
    ("source_recall_hit", "预期来源没有召回"),
    ("hit_type_matched", "命中路径不符合预期"),
    ("source_inference_matched", "source 推断不符合预期"),
    ("prompt_profile_matched", "Prompt Profile 不符合预期"),
    ("scenario_isolation_matched", "场景隔离不符合预期"),
)


def project_path(path: str | Path) -> Path:
    """把命令行路径解析为项目内路径。

    调用顺序：main() -> project_path()。
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_json(path: str | Path) -> Any:
    """读取 JSON 文件。

    调用顺序：main() -> load_report()/load_original_cases() -> load_json()。
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_report(path: str | Path) -> dict[str, Any]:
    """读取评测报告，并校验必须存在 rows 字段。

    调用顺序：main() -> load_report()。
    """
    report_path = project_path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"评测报告不存在：{report_path}")
    report = load_json(report_path)
    if not isinstance(report, dict) or not isinstance(report.get("rows"), list):
        raise ValueError(f"评测报告格式不正确，缺少 rows 列表：{report_path}")
    return report


def load_original_cases(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """按 case_id 读取原始评测集，尽量保留 expected_source_contains 等期望字段。

    调用顺序：main() -> load_original_cases()。
    """
    dataset = str(report.get("dataset") or "").strip()
    if not dataset:
        return {}
    try:
        dataset_path = resolve_eval_dataset_path(dataset)
    except FileNotFoundError:
        return {}
    raw_items = load_json(dataset_path)
    if not isinstance(raw_items, list):
        return {}
    return {
        str(item.get("case_id") or f"case_{index}"): item
        for index, item in enumerate(raw_items, start=1)
        if isinstance(item, dict)
    }


def failure_reasons(row: dict[str, Any], *, min_keyword_coverage: float) -> list[str]:
    """根据评测行判断是否应进入 Bad Case 集。

    调用顺序：main() -> select_bad_cases() -> failure_reasons()。
    """
    reasons: list[str] = []
    if row.get("error"):
        reasons.append(f"运行错误：{row.get('error')}")
    if row.get("debug_error"):
        reasons.append(f"检索诊断错误：{row.get('debug_error')}")

    for field, message in MATCH_FIELDS:
        if row.get(field) is False:
            reasons.append(message)

    expected_keywords = row.get("expected_keywords") or []
    try:
        keyword_coverage = float(row.get("keyword_coverage") or 0.0)
    except (TypeError, ValueError):
        keyword_coverage = 0.0
    if expected_keywords and keyword_coverage < min_keyword_coverage:
        reasons.append(
            f"关键词覆盖率低于阈值：{keyword_coverage} < {min_keyword_coverage}"
        )
    return reasons


def copy_if_present(target: dict[str, Any], source: dict[str, Any], keys: tuple[str, ...]) -> None:
    """从 source 复制非空字段到 target。

    调用顺序：main() -> build_bad_case() -> copy_if_present()。
    """
    for key in keys:
        value = source.get(key)
        if value not in (None, "", []):
            target[key] = value


def build_bad_case(
    *,
    row: dict[str, Any],
    original: dict[str, Any],
    reasons: list[str],
) -> dict[str, Any]:
    """把一条失败评测行转换成 eval_sets 可复跑样本。

    调用顺序：main() -> select_bad_cases() -> build_bad_case()。
    """
    source_case_id = str(row.get("case_id") or original.get("case_id") or "unknown")
    bad_case: dict[str, Any] = {
        "case_id": f"bad_{source_case_id}",
        "source_case_id": source_case_id,
        "query": row.get("question") or original.get("query") or original.get("question") or "",
        "bad_case_reasons": reasons,
        "grading_notes": "；".join(reasons),
        "observed_hit_type": row.get("hit_type") or "",
        "observed_effective_source": row.get("effective_source_filter") or "",
        "observed_prompt_profile": row.get("prompt_profile") or "",
        "observed_keyword_coverage": row.get("keyword_coverage"),
        "observed_answer_preview": row.get("answer_preview") or str(row.get("answer") or "")[:300],
    }

    # 运行参数：重新评测时需要沿用原样本的数据域和过滤条件。
    copy_if_present(
        bad_case,
        {**original, **row},
        (
            "scenario_id",
            "source_filter",
            "tenant_id",
            "dataset_id",
            "visibility",
            "user_role",
            "kb_version",
        ),
    )

    # 期望字段：优先复用原始 eval_set，报告中已有的 expected_* 作为补充。
    copy_if_present(
        bad_case,
        {**row, **original},
        (
            "expected_hit_type",
            "expected_effective_source",
            "expected_prompt_profile",
            "expected_source_contains",
            "expected_keywords",
        ),
    )
    return bad_case


def select_bad_cases(
    report: dict[str, Any],
    *,
    min_keyword_coverage: float,
    max_items: int,
) -> list[dict[str, Any]]:
    """从报告 rows 中筛选失败样本并转换为 eval_set 条目。

    调用顺序：main() -> select_bad_cases()。
    """
    originals = load_original_cases(report)
    bad_cases: list[dict[str, Any]] = []
    for row in report["rows"]:
        if not isinstance(row, dict):
            continue
        reasons = failure_reasons(row, min_keyword_coverage=min_keyword_coverage)
        if not reasons:
            continue
        original = originals.get(str(row.get("case_id") or ""), {})
        bad_cases.append(build_bad_case(row=row, original=original, reasons=reasons))
        if max_items and len(bad_cases) >= max_items:
            break
    return bad_cases


def write_bad_cases(path: str | Path, bad_cases: list[dict[str, Any]]) -> Path:
    """写出 Bad Case 评测集。

    调用顺序：main() -> write_bad_cases()。
    """
    output_path = project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bad_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    调用顺序：main() -> parse_args()。
    """
    parser = argparse.ArgumentParser(description="从本地评测报告中提取 Bad Case 到 eval_sets。")
    parser.add_argument("--report", required=True, help="评测报告路径，例如 reports/evaluation/core_chain_latest.json")
    parser.add_argument("--output", default="eval_sets/local_bad_cases.json", help="输出评测集路径")
    parser.add_argument("--min-keyword-coverage", type=float, default=0.7, help="关键词覆盖率低于该阈值时进入 Bad Case")
    parser.add_argument("--max-items", type=int, default=0, help="最多导出多少条；0 表示不限制")
    return parser.parse_args()


def main() -> None:
    """命令行入口。

    调用顺序：用户命令 -> main()。
    """
    configure_utf8_stdio()
    args = parse_args()
    report = load_report(args.report)
    bad_cases = select_bad_cases(
        report,
        min_keyword_coverage=args.min_keyword_coverage,
        max_items=args.max_items,
    )
    output_path = write_bad_cases(args.output, bad_cases)
    print_json(
        {
            "ok": True,
            "report": str(project_path(args.report)),
            "output": str(output_path),
            "bad_case_count": len(bad_cases),
            "next_steps": [
                "人工检查 output 中的 expected_* 和 grading_notes",
                "python scripts/promote_bad_cases_to_regression.py --source eval_sets/local_bad_cases.json --target eval_sets/your_regression_set.json",
                "python scripts/evaluate_core_chain.py --dataset eval_sets/your_regression_set.json --output reports/evaluation/your_regression_set_latest.json",
                "python scripts/quality/check_evaluation_gate.py --report reports/evaluation/your_regression_set_latest.json",
            ],
        }
    )


if __name__ == "__main__":
    main()
