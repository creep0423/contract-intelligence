"""Docling 文档解析增强后端冒烟检查。

该脚本只验证 `DOCUMENT_PARSER_BACKEND=docling` 的加载与解析能力，不写入 Milvus，
也不创建知识库版本。默认会生成一个临时 HTML 样例，方便在新环境中快速确认 Docling
是否安装可用；也可以通过 `--input` 指定真实 PDF、DOCX、PPTX 或 HTML 文件。

用法：
    python scripts/tools/docling_parser_smoke.py
    python scripts/tools/docling_parser_smoke.py --input 你的复杂版面资料.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from qa_core.config.settings import get_settings  # noqa: E402
from qa_core.indexing.document_loaders import load_file  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。

    参数：
        无。

    返回：
        配置好 input / preview-chars 参数的 ArgumentParser。

    调用顺序：main() -> build_parser()。
    """
    parser = argparse.ArgumentParser(description="Smoke test Docling document parser backend.")
    parser.add_argument("--input", default="", help="可选：指定一个 PDF/DOCX/PPTX/HTML 文件。")
    parser.add_argument("--preview-chars", type=int, default=600, help="输出正文预览长度。")
    return parser


def create_sample_html(directory: Path) -> Path:
    """生成一个最小 HTML 样例，用于不依赖业务资料的解析验证。

    样例包含标题、段落和表格，覆盖基础版面结构；只在未指定 --input 时使用。

    参数：
        directory: 临时目录，样例文件写入其中。

    返回：
        生成的 HTML 文件路径。

    调用顺序：resolve_input() -> create_sample_html()。
    """
    path = directory / "docling_smoke_sample.html"
    path.write_text(
        "\n".join(
            [
                "<html><body>",
                "<h1>Docling parser smoke</h1>",
                "<p>预算审批超过 8000 元时，需要补充预算占用说明。</p>",
                "<table>",
                "<tr><th>材料</th><th>状态</th></tr>",
                "<tr><td>预算占用记录</td><td>必填</td></tr>",
                "</table>",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return path


def resolve_input(value: str, temp_dir: Path) -> Path:
    """解析输入文件；未传入时生成临时 HTML 样例。

    相对路径按项目根目录解析；指定文件不存在时直接报错，避免冒烟误用真实资料。

    参数：
        value: --input 命令行值；为空表示使用临时样例。
        temp_dir: 临时目录，用于生成样例 HTML。

    返回：
        解析后的输入文件路径。

    异常：
        FileNotFoundError: 指定输入文件不存在。

    调用顺序：main() -> resolve_input() -> create_sample_html()。
    """
    if not value:
        return create_sample_html(temp_dir)
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在：{path}")
    return path


def main() -> None:
    """运行 Docling loader 冒烟检查。

    锁定 docling 后端 -> 解析输入文件 -> 调用 load_file -> 打印解析结果与元数据；
    RuntimeError 时输出失败 JSON 并以退出码 1 结束。

    参数：
        无（参数由 build_parser() 解析）。

    返回：
        无；结果通过 print() 输出。

    调用顺序：命令行入口 -> main() -> build_parser() -> resolve_input()。
    """
    parser = build_parser()
    args = parser.parse_args()
    # 原因：冒烟必须锁定 docling 后端，避免本地 .env 或默认配置选了别的后端导致结果失真。
    os.environ["DOCUMENT_PARSER_BACKEND"] = "docling"

    # 原因：settings 带模块级缓存，必须先清空，否则刚才设置的后端不会生效。
    get_settings.cache_clear()

    with tempfile.TemporaryDirectory() as temp_dir:
        path = resolve_input(args.input, Path(temp_dir))
        try:
            documents = load_file(path)
        except RuntimeError as exc:
            print(
                {
                    "ok": False,
                    "parser_backend": get_settings().document_parser_backend,
                    "input": str(path),
                    "error": str(exc),
                }
            )
            raise SystemExit(1) from exc

    content = "\n\n".join(document.page_content for document in documents).strip()
    metadata = [document.metadata for document in documents]
    print(
        {
            "ok": bool(content),
            "parser_backend": get_settings().document_parser_backend,
            "input": str(path),
            "document_count": len(documents),
            "metadata": metadata,
            "preview": content[: args.preview_chars],
        }
    )


if __name__ == "__main__":
    main()
