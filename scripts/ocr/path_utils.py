"""
离线 OCR 脚本的路径解析工具。

所有路径统一解析到项目根目录（PROJECT_ROOT），避免 IDE 工作目录不同
导致文件找不到。支持 PDF、PNG、JPG、BMP、TIFF 等常见扫描件和图片格式。
"""

from __future__ import annotations

from pathlib import Path

from scripts.common import PROJECT_ROOT

# 离线 OCR 支持的输入文件后缀（扫描件和图片格式）
# 不包含 .docx、.pptx 等原生文档格式——这些不需要 OCR
OCR_SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def resolve_project_path(path_value: str | Path) -> Path:
    """把相对路径固定解析到项目根目录，避免 IDE 工作目录不同导致文件找不到。

    调用顺序：命令行入口 -> resolve_project_path()。
    """
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def is_ocr_supported_input(path: Path) -> bool:
    """判断输入文件是否属于离线 OCR 支持的扫描件或图片格式。

    调用顺序：命令行入口 -> is_ocr_supported_input()。
    """
    return path.suffix.lower() in OCR_SUPPORTED_SUFFIXES


def collect_inputs(input_path: str | None, input_dir: str | None) -> list[Path]:
    """收集本次需要 OCR 的文件。

    调用顺序：命令行入口 -> collect_inputs()。
    """
    files: list[Path] = []
    if input_path:
        files.append(resolve_project_path(input_path))
    if input_dir:
        root = resolve_project_path(input_dir)
        files.extend(path for path in sorted(root.rglob("*")) if path.is_file() and is_ocr_supported_input(path))
    unique: list[Path] = []
    for path in files:
        if path not in unique:
            unique.append(path)
    return unique
