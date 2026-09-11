"""离线 OCR 清洗入口。

该脚本用于处理扫描件、图片 PDF 或图片附件，输出待人工复核的 Markdown 和 JSON 报告。
它不会写 Milvus，不会激活知识库版本，也不会被在线问答调用。

示例：
python scripts/ocr/run_offline_ocr.py --input data_packs/enterprise_realistic_pack/中医临床诊疗智能助手.pdf --output-dir reports/ocr
python scripts/ocr/run_offline_ocr.py --input-dir incoming_scans --output-dir reports/ocr/batch_001
"""

from __future__ import annotations

# ── 标准库 ──
# argparse: 命令行参数解析（--input, --output-dir, --lang 等）
import argparse
# json: 标准 JSON 序列化
import json
# sys: 系统功能（sys.path 修改 + sys.exit 退出码）
import sys
# pathlib.Path: 文件路径操作
from pathlib import Path

# ── 路径设置 ──
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── 导入核心模块 ──
# create_ocr_engine: 创建 PaddleOCR 引擎实例（支持中英文等多语言）
# recognize_file: 对单个文件执行 OCR 识别，输出 Markdown
from qa_core.indexing.ocr_documents import create_ocr_engine, recognize_file
# configure_utf8_stdio: Windows UTF-8 输出保护
# utc_now: 生成 UTC 时间戳
# write_json_file: 写入 JSON 报告文件
from scripts.common import configure_utf8_stdio, utc_now, write_json_file
# collect_inputs: 收集 --input 和 --input-dir 指定的所有待 OCR 文件
# resolve_project_path: 将相对路径解析为项目根目录下的绝对路径
from scripts.ocr.path_utils import collect_inputs, resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    调用顺序：命令行入口 -> build_parser()。
    """
    parser = argparse.ArgumentParser(description="Run offline OCR and create reviewable markdown files.")
    parser.add_argument("--input", default="", help="单个扫描件、图片 PDF 或图片文件。")
    parser.add_argument("--input-dir", default="", help="批量 OCR 输入目录。")
    parser.add_argument("--output-dir", default=str(Path("reports") / "ocr"), help="OCR Markdown 和报告输出目录。")
    parser.add_argument("--report-output", default="", help="JSON 报告路径，默认写到 output-dir/ocr_report.json。")
    parser.add_argument("--lang", default="ch", help="PaddleOCR 语言配置，中文资料默认 ch。")
    parser.add_argument("--min-confidence", type=float, default=0.78, help="低于该平均置信度时标记为需重新扫描或人工整理。")
    parser.add_argument("--dpi", type=int, default=180, help="PDF 渲染 DPI。")
    parser.add_argument("--max-pages", type=int, default=0, help="每个 PDF 最多处理页数，0 表示不限制。")
    return parser


def main() -> None:
    """执行离线 OCR 并保存报告。

    流程：
      1. 收集输入文件（单文件或批量目录）
      2. 创建 PaddleOCR 引擎
      3. 逐个文件执行 OCR 识别 → 输出 Markdown
      4. 汇总结果：成功数、失败数、待复核数
      5. 写入 JSON 报告
      6. 有失败文件时返回非零退出码

    调用顺序：命令行入口 -> main()。
    """
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args()

    # 步骤 1：收集待 OCR 文件
    inputs = collect_inputs(args.input, args.input_dir)
    if not inputs:
        parser.error("必须通过 --input 或 --input-dir 提供至少一个 OCR 文件。")

    output_dir = resolve_project_path(args.output_dir)

    # 步骤 2：创建 OCR 引擎（PaddleOCR，支持中英文等多语言）
    engine = create_ocr_engine(lang=args.lang)

    # 步骤 3：逐文件执行 OCR 识别
    results = []
    failures = []
    for path in inputs:
        try:
            result = recognize_file(
                path,
                output_dir=output_dir,
                engine=engine,
                min_confidence=args.min_confidence,   # 低于该置信度标记为需要重新扫描
                dpi=args.dpi,                          # PDF 渲染 DPI（越高越清晰但越慢）
                max_pages=args.max_pages,              # 0 表示不限制页数
            )
            print(result)
            results.append(result.as_dict())
        except Exception as exc:
            # 单个文件失败不中断整体流程，继续处理下一文件
            failures.append({"path": str(path), "error": str(exc)})

    # 步骤 4：汇总报告
    report = {
        "report_type": "offline_ocr",
        "created_at": utc_now(),
        "input_count": len(inputs),
        "success_count": len(results),
        "failure_count": len(failures),
        "ready_for_review_count": sum(1 for item in results if item.get("ready_for_review")),
        "output_dir": str(output_dir),
        "results": results,
        "failures": failures,
        "next_step": "人工复核 Markdown 后，再放入对应场景 data/<source>_data 目录并重建知识库版本。",
    }
    # 步骤 5：写入报告
    report_path = resolve_project_path(args.report_output) if args.report_output else output_dir / "ocr_report.json"
    write_json_file(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # 步骤 6：有失败文件时返回非零退出码
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
