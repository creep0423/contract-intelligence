"""把用户点踩反馈导出为可复核 Bad Case 草稿。

用户反馈不是评测真值。这个脚本只做一件事：把 `qa_feedback` 中的低质量反馈转成
`eval_sets/local_feedback_bad_cases.json` 这类复核草稿。人工补齐 `expected_*` 后，再用
`promote_bad_cases_to_regression.py` 合并进正式回归集。

用法示例：

    python scripts/export_feedback_bad_cases.py \
      --scenario enterprise_knowledge \
      --output eval_sets/local_feedback_bad_cases.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa_core.memory.feedback import get_feedback_store
from scripts.common import PROJECT_ROOT, configure_utf8_stdio, print_json


DEFAULT_OUTPUT = PROJECT_ROOT / "eval_sets" / "local_feedback_bad_cases.json"


def project_path(path: str | Path) -> Path:
    """把命令行路径解析为项目内绝对路径。

    参数：
        path: 命令行传入的路径；绝对路径原样返回，相对路径按项目根目录拼接。

    返回：
        解析后的 Path 对象。

    调用顺序：write_bad_cases() -> project_path()。
    """

    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _source_text(source: dict[str, Any]) -> str:
    """从来源快照中提取便于人工复核的标识。

    依次尝试 file_name / standard_question / source / content，拼接后截断到 300 字符。

    参数：
        source: 来源快照字典，含 metadata 与 content。

    返回：
        拼接后的来源标识文本。

    调用顺序：build_feedback_bad_case() -> _source_text()。
    """

    metadata = source.get("metadata") or {}
    parts = [
        metadata.get("file_name"),
        metadata.get("standard_question"),
        metadata.get("source"),
        source.get("content"),
    ]
    return " | ".join(str(part).strip() for part in parts if str(part or "").strip())[:300]


def _observed_source_filter(row: dict[str, Any]) -> str:
    """从来源快照里读取实际召回到的 source，仅作为 observed 信息。

    只取第一个带 source 元数据的来源；这是观测值而非期望值，复核时不能据此判定对错。

    参数：
        row: 反馈记录字典，含 sources 列表。

    返回：
        第一个来源的 source 值；没有时返回空字符串。

    调用顺序：build_feedback_bad_case() -> _observed_source_filter()。
    """

    for source in row.get("sources") or []:
        metadata = source.get("metadata") or {}
        value = metadata.get("source")
        if value:
            return str(value)
    return ""


def build_feedback_bad_case(row: dict[str, Any]) -> dict[str, Any]:
    """把一条用户反馈记录转换成 Bad Case 复核草稿。（★★★ 核心）

    生成 feedback_{id} 形式的稳定 case_id / source_case_id，把用户评分与备注写进
    bad_case_reasons；expected_* 字段留待人工补齐后再合并进正式回归集。

    参数：
        row: 一条反馈记录，至少包含 id 与 question。

    返回：
        Bad Case 复核草稿字典。

    异常：
        ValueError: 缺少 id 或 question。

    调用顺序：export_feedback_bad_cases() -> build_feedback_bad_case() ->
        _observed_source_filter() / _source_text()。
    """

    feedback_id = str(row.get("id") or "").strip()
    if not feedback_id:
        raise ValueError("反馈记录缺少 id，无法生成稳定 case_id")
    question = str(row.get("question") or "").strip()
    if not question:
        raise ValueError(f"反馈记录 {feedback_id} 缺少 question")

    comment = str(row.get("comment") or "").strip()
    reasons = [f"用户反馈：{row.get('rating') or 'not_useful'}"]
    if comment:
        reasons.append(f"用户备注：{comment}")

    bad_case: dict[str, Any] = {
        "case_id": f"feedback_{feedback_id}",
        "source_case_id": f"feedback_{feedback_id}",
        "feedback_id": int(feedback_id),
        "query": question,
        "bad_case_reasons": reasons,
        "grading_notes": "；".join(reasons) + "。请人工复核 expected_* 字段后再合并进正式回归集。",
        "observed_answer_preview": str(row.get("answer") or "")[:500],
        "observed_effective_source": _observed_source_filter(row),
        "observed_sources": [_source_text(source) for source in row.get("sources") or [] if isinstance(source, dict)],
    }

    for key in ("scenario_id", "tenant_id", "dataset_id"):
        value = row.get(key)
        if value not in (None, "", []):
            bad_case[key] = value
    return bad_case


def export_feedback_bad_cases(
    rows: list[dict[str, Any]],
    *,
    max_items: int = 0,
) -> list[dict[str, Any]]:
    """批量转换反馈记录。

    按顺序逐条转换，达到 max_items 上限即停止。

    参数：
        rows: 反馈记录列表。
        max_items: 最多导出的条数；0 表示不限制。

    返回：
        Bad Case 草稿列表。

    调用顺序：main() -> export_feedback_bad_cases() -> build_feedback_bad_case()。
    """

    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(build_feedback_bad_case(row))
        if max_items and len(output) >= max_items:
            break
    return output


def write_bad_cases(path: str | Path, bad_cases: list[dict[str, Any]]) -> Path:
    """写出 Bad Case 草稿 JSON 数组。

    自动创建父目录，以 UTF-8 格式化输出。

    参数：
        path: 输出文件路径。
        bad_cases: Bad Case 草稿列表。

    返回：
        实际写入的 Path 对象。

    调用顺序：main() -> project_path() -> write_bad_cases()。
    """

    output_path = project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bad_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。

    参数：
        无。

    返回：
        配置好 scenario / rating / limit / max-items / output 的 ArgumentParser。

    调用顺序：main() -> build_parser()。
    """

    parser = argparse.ArgumentParser(description="Export low-quality user feedback as reviewable bad case drafts.")
    parser.add_argument("--scenario", default="", help="按 scenario_id 过滤；为空表示不过滤。")
    parser.add_argument("--rating", default="not_useful", help="反馈评分，默认 not_useful。")
    parser.add_argument("--limit", type=int, default=200, help="最多读取多少条反馈。")
    parser.add_argument("--max-items", type=int, default=0, help="最多导出多少条；0 表示不限制。")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)), help="输出 JSON 路径。")
    return parser


def main() -> None:
    """命令行入口。

    从反馈存储读取低质量反馈，转成 Bad Case 草稿并写出，最后打印汇总与后续步骤。

    参数：
        无（参数由 build_parser() 解析）。

    返回：
        无；汇总信息通过 print_json() 输出。

    调用顺序：命令行入口 -> main() -> build_parser() -> export_feedback_bad_cases() ->
        write_bad_cases() -> print_json()。
    """

    configure_utf8_stdio()
    args = build_parser().parse_args()
    rows = get_feedback_store().list_bad_feedback(
        limit=args.limit,
        scenario_id=args.scenario or None,
        rating=args.rating,
    )
    bad_cases = export_feedback_bad_cases(rows, max_items=args.max_items)
    output_path = write_bad_cases(args.output, bad_cases)
    print_json(
        {
            "ok": True,
            "rating": args.rating,
            "scenario_id": args.scenario or None,
            "feedback_count": len(rows),
            "bad_case_count": len(bad_cases),
            "output": str(output_path),
            "next_steps": [
                "人工补齐 output 中的 expected_* 和 grading_notes",
                "python scripts/promote_bad_cases_to_regression.py --source eval_sets/local_feedback_bad_cases.json --target eval_sets/your_regression_set.json",
                "python scripts/evaluate_core_chain.py --dataset eval_sets/your_regression_set.json --output reports/evaluation/your_regression_set_latest.json",
                "python scripts/quality/check_evaluation_gate.py --report reports/evaluation/your_regression_set_latest.json",
            ],
        }
    )


if __name__ == "__main__":
    main()
