# -*- coding: utf-8 -*-
"""检查 V1.0 讲义和章节实操代码是否保持冻结。

V1.0 的 `docs/` 与 `codealong/` 已经作为交付资产定版。V2.0 资料必须放入
`v2/docs/` 和 `v2/codealong/`，不能继续修改 V1.0 目录。

用法：
    python scripts/check_v1_freeze.py
    python scripts/check_v1_freeze.py --base v1.0.4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_PATHS = ("docs", "codealong")


def run_git(args: list[str]) -> str:
    """运行 git 命令并返回标准输出。

    调用顺序：命令行入口 -> run_git()。
    """
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def diff_name_status(args: list[str]) -> list[str]:
    """读取 git diff --name-status 输出。

    调用顺序：命令行入口 -> diff_name_status()。
    """
    output = run_git(["diff", "--name-status", *args, "--", *FROZEN_PATHS])
    return [line for line in output.splitlines() if line.strip()]


def collect_changes(base: str) -> list[str]:
    """收集相对基线、暂存区和工作区中触碰冻结目录的变更。

    调用顺序：命令行入口 -> collect_changes()。
    """
    merge_base = run_git(["merge-base", base, "HEAD"])
    changes: list[str] = []
    changes.extend(diff_name_status([f"{merge_base}..HEAD"]))
    changes.extend(diff_name_status(["--cached"]))
    changes.extend(diff_name_status([]))
    return sorted(set(changes))


def main() -> None:
    """执行冻结检查。

    调用顺序：命令行入口 -> main()。
    """
    parser = argparse.ArgumentParser(description="检查 V1.0 讲义和章节实操代码是否被修改。")
    parser.add_argument("--base", default="v1.0.4", help="V1.0 冻结基线，默认 v1.0.4。")
    args = parser.parse_args()

    try:
        changes = collect_changes(args.base)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr)
        sys.exit(exc.returncode)

    if changes:
        print("V1.0 冻结目录被修改，请把 V2.0 资料放到 v2/ 目录：")
        for line in changes:
            print(f"- {line}")
        sys.exit(1)

    print("V1.0 冻结检查通过：docs/ 与 codealong/ 未发生变更。")


if __name__ == "__main__":
    main()
