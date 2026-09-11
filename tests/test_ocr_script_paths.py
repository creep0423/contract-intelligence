"""验证 OCR 工具脚本在不同启动目录下均能解析项目路径。

调用顺序：课程示例、测试或命令行入口 -> 本模块公开接口。
"""

from pathlib import Path

from scripts.common import PROJECT_ROOT
from scripts.ocr.path_utils import collect_inputs, resolve_project_path


def test_resolve_project_path_anchors_relative_paths_to_project_root() -> None:
    """验证相对路径被解析为项目根目录下的绝对路径。

    调用顺序：pytest/unittest 测试入口 -> test_resolve_project_path_anchors_relative_paths_to_project_root()。
    """
    path = resolve_project_path("data_packs/enterprise_realistic_pack")

    assert path == PROJECT_ROOT / "data_packs" / "enterprise_realistic_pack"


def test_resolve_project_path_keeps_absolute_paths() -> None:
    """验证绝对路径传入时保持原样不变。

    调用顺序：pytest/unittest 测试入口 -> test_resolve_project_path_keeps_absolute_paths()。
    """
    absolute = PROJECT_ROOT / "data_packs"

    assert resolve_project_path(absolute) == absolute


def test_collect_inputs_accepts_project_relative_file() -> None:
    """验证 collect_inputs 接受项目相对路径并将其转换为绝对路径。

    调用顺序：pytest/unittest 测试入口 -> test_collect_inputs_accepts_project_relative_file()。
    """
    input_path = "data_packs/enterprise_realistic_pack/中医临床诊疗智能助手.pdf"

    assert collect_inputs(input_path, "") == [PROJECT_ROOT / Path(input_path)]
