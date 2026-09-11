# -*- coding: utf-8 -*-
# ============================================================================
# 重建产品运行时启用的业务场景知识库版本
# ============================================================================
# 默认仅初始化或重建合同履约风控场景：
#   python scripts/rebuild_scenarios.py --reset-collections
#
# 离线回归时仍可显式指定历史场景，但它们不会进入产品运行时：
#   python scripts/rebuild_scenarios.py --scenarios enterprise_knowledge --reset-collections
#
# 脚本通过子进程逐个调用 `scripts/rebuild_kb_version.py`，保证单场景已有的
# FAQ/文档入库、质量门禁、版本激活和 Milvus schema reset 逻辑完全复用。
#
# 产品运行时场景：
#   tender_contract_risk     — 企业合同履约风控
#
# 用法示例：
#   # 重建合同履约风控场景（含 Collection 重置）
#   python scripts\rebuild_scenarios.py --reset-collections
#
#   # 离线回归时显式重建一个历史 staged 版本，不激活也不跑质量门禁
#   python scripts\rebuild_scenarios.py --scenarios enterprise_knowledge --no-activate --no-quality-gate
#
#   # 预演模式（只打印命令不执行）
#   python scripts\rebuild_scenarios.py --dry-run
#
#   # 遇到失败继续执行后续场景
#   python scripts\rebuild_scenarios.py --continue-on-failure
# ============================================================================

"""按产品运行时场景清单构建并激活知识库版本。

调用顺序：课程示例、测试或命令行入口 -> 本模块公开接口。
"""

from __future__ import annotations

# argparse: 命令行参数解析
import argparse

# subprocess: 子进程管理（通过 subprocess.run 调用 rebuild_kb_version.py）
import subprocess

# sys: 系统功能（sys.executable 获取当前 Python 解释器路径，sys.exit 退出码）
import sys

# time: 时间功能（time.perf_counter 高精度计时）
import time

# dataclasses: 数据类定义（ScenarioRunResult）
from dataclasses import dataclass

# pathlib.Path: 文件路径操作
from pathlib import Path

# ── 常量定义 ──

# PROJECT_ROOT: 项目根目录绝对路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DEFAULT_REBUILD_SCRIPT: 单场景重建脚本的默认路径
DEFAULT_REBUILD_SCRIPT = PROJECT_ROOT / "scripts" / "rebuild_kb_version.py"

# 当前产品只重建唯一启用的合同履约风控场景；历史场景目录不参与运行时构建。
DEFAULT_ALL_SCENARIOS = (
    "tender_contract_risk",
)

@dataclass(frozen=True)
class ScenarioRunResult:
    """单个场景重建结果。

    用于在 main() 汇总表中统一展示每个场景的成功/失败状态、退出码和耗时，
    也用于决定是否继续执行后续场景（--continue-on-failure 未传时，首个失败即中断）。

    设计决策：
    - frozen=True：结果一旦生成就不可变，避免在汇总循环里被意外修改导致统计失真。
      批量脚本里结果对象会被多次读取（汇总打印、失败筛选、退出码判断），
      不可变保证每次读到的都是 run_one() 返回时的原始值。
    - 用 dataclass 而非普通 dict：字段固定且语义明确，IDE 自动补全友好，
      避免用 dict 时拼写错字段名（如 "returncode" vs "return_code"）在运行时才报错。
    - 不存 stdout/stderr：子进程输出直接继承父进程 stdout，这里只保留判断汇总
      需要的 4 个字段，避免结果对象膨胀。

    Attributes:
        scenario_id: 场景标识，例如 "tender_contract_risk"。
        ok: 是否成功，等价于 returncode == 0。汇总表中显示为 OK / FAILED(code)。
        elapsed_seconds: 执行耗时（秒），用 time.perf_counter() 高精度计时，
            包含子进程启动、FAQ 入库、文档入库、质量门禁、版本激活的全流程。
            用于汇总表中按耗时排序或判断哪个场景异常缓慢。
        returncode: 子进程退出码。0=成功；非 0=失败（rebuild_kb_version.py
            在质量门禁失败时 sys.exit(1)，其他异常可能返回不同码）。

    调用顺序：命令行入口 -> run_one() -> ScenarioRunResult -> main() 汇总。
    """
    scenario_id: str
    ok: bool
    elapsed_seconds: float
    returncode: int


def parse_scenarios(value: str) -> list[str]:
    """解析逗号分隔的场景 ID 列表字符串为有序去空白列表。

    示例：
      "enterprise_knowledge,equipment_ops" → ["enterprise_knowledge", "equipment_ops"]
      "  a , b , c  " → ["a", "b", "c"]
      "" → []
      "  ,,  " → []

    设计决策：
    - 不校验场景是否存在：parse_scenarios 只做字符串切分，场景存在性校验留给
      rebuild_kb_version.py 内部的 resolve_scenario() 报错。这样本函数职责单一，
      且错误信息更精确（resolve_scenario 会列出当前注册的全部场景便于排查）。
    - 保留传入顺序：不排序，按用户传入的顺序执行。用户可能希望先重建依赖场景
      再重建主场景，顺序由命令行控制。
    - 空字符串过滤：split(",") 后可能产生空串（连续逗号、首尾逗号），
      用 if item.strip() 过滤掉，避免把空场景 ID 传给 rebuild_kb_version.py。

    Args:
        value: 命令行传入的逗号分隔字符串，例如 "enterprise_knowledge,equipment_ops"。

    Returns:
        去空白、去空串后的场景 ID 列表，保留传入顺序。空字符串返回空列表。

    调用顺序：命令行入口 -> main() -> parse_scenarios()。
    """
    return [item.strip() for item in value.split(",") if item.strip()]


def build_command(args: argparse.Namespace, scenario_id: str) -> list[str]:
    """把批量重建的命令行参数映射为单场景 rebuild_kb_version.py 的参数列表。

    每个场景固定使用 --new-version --force，体现"批量重建"的语义：
    - --new-version：每个场景创建全新的 kb_version，不复用已有 active 版本，
      避免把重建结果混入旧版本导致 chunk 来源混乱。
    - --force：忽略文件 fingerprint，强制重新 embedding 所有文档。批量重建
      通常用于初始化或 schema 变更后，此时即使 fingerprint 未变也需要全量重写。

    Args:
        args: 批量脚本的命令行参数命名空间，包含 reset_collections / quality_gate /
            activate / description / tenant_id / dataset_id / visibility / allowed_role
            等字段。这些字段会被映射到 rebuild_kb_version.py 的对应参数。
        scenario_id: 目标场景 ID，例如 "enterprise_knowledge"。

    Returns:
        命令参数列表，第一项是 Python 解释器路径（sys.executable），第二项是
        rebuild_kb_version.py 的绝对路径，后续是 -- 开头的参数。
        列表格式可直接传给 subprocess.run()，无需 shell=True，避免 shell 注入风险。

    设计决策：
    - 用 sys.executable 而非裸 "python"：确保批量脚本用哪个解释器跑，子进程就用
     哪个解释器。避免 PATH 里找到的 python 是另一个环境（如系统 Python 而非
      conda env），导致依赖找不到。
    - 可选参数条件追加：只有 args.xxx 为真时才追加对应 --xxx 参数，避免向
      rebuild_kb_version.py 传入空值导致 argparse 报错。
    - --allowed-role 可重复：用 for 循环逐个追加，支持一个场景配置多个角色。

    调用顺序：命令行入口 -> run_one() -> build_command() -> subprocess.run()。
    """
    # 命令前两项固定：当前 Python 解释器 + rebuild_kb_version.py 绝对路径
    # 用 sys.executable 确保子进程用同一套依赖（conda env / venv），
    # 避免系统 PATH 里的 python 找不到项目依赖（pymilvus / langchain 等）
    command = [
        sys.executable,                          # 当前 Python 解释器绝对路径
        str(Path(args.rebuild_script)),          # rebuild_kb_version.py 的绝对路径
        "--scenario", scenario_id,               # 目标场景 ID
        "--new-version",                         # 创建新版本（批量重建语义：不复用旧版本）
        "--force",                               # 强制重新 embedding（忽略 fingerprint）
    ]
    # 可选参数：只有显式传入时才追加，避免向子进程传空值
    # --reset-collections：schema 变更时先 drop 重建 Milvus collection
    if args.reset_collections:
        command.append("--reset-collections")
    # --quality-gate：激活前必须通过入库质量门禁（默认启用，--no-quality-gate 关闭）
    if args.quality_gate:
        command.append("--quality-gate")
    # --activate：质量门禁通过后激活为新 active 版本（默认启用，--no-activate 关闭）
    if args.activate:
        command.append("--activate")
    # --description：版本描述，写入 kb_versions.description 字段便于审计
    if args.description:
        command.extend(["--description", args.description])
    # 数据隔离字段：写入每条 chunk metadata，支持租户/数据集/可见级别/角色过滤
    if args.tenant_id:
        command.extend(["--tenant-id", args.tenant_id])
    if args.dataset_id:
        command.extend(["--dataset-id", args.dataset_id])
    if args.visibility:
        command.extend(["--visibility", args.visibility])
    # --allowed-role 可重复传入，每个角色追加一次 --allowed-role
    for role in args.allowed_role or []:
        command.extend(["--allowed-role", role])
    return command


def run_one(args: argparse.Namespace, scenario_id: str) -> ScenarioRunResult:
    """执行单个场景的重建子进程，阻塞等待完成并返回结果。

    通过 subprocess.run 调用 rebuild_kb_version.py，子进程独立运行完整的
    单场景重建流程：解析场景 → 入库 → 质量门禁 → 版本激活。父进程阻塞等待
    子进程退出，收集退出码和耗时后返回 ScenarioRunResult。

    Args:
        args: 批量脚本的命令行参数命名空间，传给 build_command() 构造子进程命令。
        scenario_id: 目标场景 ID。

    Returns:
        ScenarioRunResult 实例，包含 scenario_id / ok / elapsed_seconds / returncode。
        dry-run 模式下返回 ok=True, elapsed=0.0, returncode=0（不实际执行子进程）。

    设计决策：
    - 用 subprocess.run 阻塞而非 Popen 异步：批量重建是串行依赖场景，每个场景
      都要独占 Milvus/MySQL 资源，并发执行会导致 collection drop/create 互相干扰。
      阻塞执行保证一个场景完全结束后再启动下一个，资源隔离干净。
    - cwd=PROJECT_ROOT：子进程工作目录设为项目根，确保 rebuild_kb_version.py
      内部的相对路径（如 scenario.toml 查找、data_root 解析）与直接运行时一致。
      如果不设 cwd，子进程会继承父进程的工作目录，在容器或 IDE 里可能不是项目根。
    - 打印分隔线和命令：每个场景开始前打印 88 字符分隔线 + 完整命令，便于在
      批量输出中快速定位某个场景的日志段。命令打印也方便手动复现单场景重建。
    - time.perf_counter() 计时：用高精度计时器（不受系统时钟调整影响），
      汇总表中展示每个场景耗时，帮助发现异常缓慢的场景（如资料量过大或 Milvus 慢查询）。

    调用顺序：命令行入口 -> main() 循环 -> run_one() -> build_command() -> subprocess.run()。
    """
    # 构造子进程命令（Python 解释器 + rebuild_kb_version.py + 场景参数）
    command = build_command(args, scenario_id)
    # 打印场景分隔线，便于在批量输出中定位某个场景的日志段
    print("\n" + "=" * 88)
    print(f"Rebuilding scenario: {scenario_id}")
    # 打印完整命令，方便手动复现单场景重建（复制粘贴即可跑）
    print("Command:", " ".join(command))
    print("=" * 88)

    # 高精度计时起点（不受系统时钟调整影响）
    started = time.perf_counter()

    # ── dry-run 模式：只打印命令不执行，返回成功结果用于预演验证 ──
    if args.dry_run:
        return ScenarioRunResult(
            scenario_id=scenario_id,
            ok=True,
            elapsed_seconds=0.0,
            returncode=0,
        )

    # ── 真实执行：subprocess.run 阻塞等待子进程退出 ──
    # cwd=PROJECT_ROOT 确保子进程工作目录在项目根，相对路径（scenario.toml 等）能正确解析
    # 不用 shell=True：命令是 list 格式，避免 shell 注入风险
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - started
    return ScenarioRunResult(
        scenario_id=scenario_id,
        # returncode == 0 表示子进程成功完成入库 + 质量门禁 + 激活
        ok=completed.returncode == 0,
        elapsed_seconds=elapsed,
        returncode=completed.returncode,
    )


def main() -> None:
    """批量重建多个业务场景的知识库版本。

    执行流程：
      1. 构造命令行解析器并解析参数
      2. 校验参数语义冲突（activate 依赖 quality_gate）
      3. 解析目标场景列表
      4. 逐个场景调用 run_one()：
         - 每个场景独立子进程执行，串行不并发（Milvus 资源隔离）
         - 某场景失败后：
           * --continue-on-failure → 继续下一个场景
           * 否则 → 中断执行，已完成的场景结果仍然保留
      5. 打印汇总表格（场景名 / 状态 / 耗时）
      6. 有失败场景时 sys.exit(1)，便于 CI/CD 判定批量重建是否成功

    设计决策：
    - set_defaults(quality_gate=True, activate=True)：默认启用质量门禁和激活，
      因为批量重建通常用于初始化或全量重建，产物要直接上线。
      关闭时必须显式传 --no-quality-gate / --no-activate，避免误操作跳过门禁。
    - activate 依赖 quality_gate：激活版本必须先通过质量门禁，否则会把有问题的
      候选版本切到线上。parser.error 在质量门禁关闭但激活开启时报错。
    - 串行执行而非并发：每个场景都要独占 Milvus collection（--reset-collections 时
      会 drop/create），并发会导致 collection 互相干扰。串行保证资源隔离干净。
    - 失败时退出码 1：CI/CD 根据退出码判定批量重建是否成功，非 0 触发告警或阻断发布。

    命令行参数说明：
      --scenarios: 逗号分隔的场景 ID 列表，默认全部 8 个冻结场景。
      --reset-collections: 重建前 drop 每个 scenario 的 FAQ/Doc Milvus collection。
          schema 变更（如迁移到 BM25 BuiltInFunction）时必须传。
      --no-quality-gate: 关闭入库质量门禁（不推荐生产环境使用）。
      --no-activate: 只创建 staged 版本不激活，用于预演或调试。
      --description: 版本描述，写入 kb_versions.description 字段。
      --tenant-id / --dataset-id / --visibility / --allowed-role: 数据隔离字段，
          透传给 rebuild_kb_version.py，写入每条 chunk metadata。
      --rebuild-script: rebuild_kb_version.py 的路径。容器内挂载路径不同时需显式指定。
      --continue-on-failure: 某场景失败后继续执行后续场景，用于批量重建时
          希望一次性看到所有失败场景而不是停在第一个。
      --dry-run: 只打印命令不执行，用于预演验证命令构造是否正确。

    调用顺序：命令行入口 -> main() -> parse_scenarios() -> run_one() 循环 -> 汇总输出。
    """
    # ── 第一步：构造命令行解析器 ──
    parser = argparse.ArgumentParser(description="Batch rebuild scenario knowledge base versions.")
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_ALL_SCENARIOS),
        help="Comma-separated scenario ids. Defaults to all 8 frozen business scenarios.",
    )
    parser.add_argument(
        "--reset-collections", action="store_true",
        help="Drop each scenario FAQ/Doc Milvus collection before rebuild.",
    )
    # dest="quality_gate" + action="store_false"：传 --no-quality-gate 时 quality_gate=False
    # 配合下方 set_defaults(quality_gate=True) 实现默认启用、显式关闭
    parser.add_argument(
        "--no-quality-gate", dest="quality_gate", action="store_false",
        help="Disable ingestion quality gate. Not recommended for production.",
    )
    parser.add_argument(
        "--no-activate", dest="activate", action="store_false",
        help="Only create staged versions; do not activate them.",
    )
    parser.add_argument("--description", default="batch rebuild scenarios", help="Version description.")
    # 数据隔离字段：透传给 rebuild_kb_version.py，写入 chunk metadata 支持检索过滤
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--visibility", default=None)
    parser.add_argument("--allowed-role", action="append", default=None)
    parser.add_argument(
        "--rebuild-script", default=str(DEFAULT_REBUILD_SCRIPT),
        help="Path to rebuild_kb_version.py. When running through a host volume mounted "
             "at /work inside the api container, use --rebuild-script /app/scripts/rebuild_kb_version.py.",
    )
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="Continue after a scenario fails.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them.")

    # set_defaults 确保 --no-quality-gate / --no-activate 未传时默认启用
    # 批量重建通常用于初始化或全量重建，产物要直接上线，因此默认启用质量门禁和激活
    parser.set_defaults(quality_gate=True, activate=True)
    args = parser.parse_args()

    # ── 第二步：校验参数语义冲突 ──
    # activate 依赖 quality_gate：激活版本必须先通过质量门禁，否则会把有问题的候选版本切到线上
    if args.activate and not args.quality_gate:
        parser.error("--no-quality-gate can only be used together with --no-activate.")

    # ── 第三步：解析目标场景列表 ──
    scenarios = parse_scenarios(args.scenarios)
    if not scenarios:
        # 空场景列表直接报错，避免无意义地走到汇总输出
        parser.error("--scenarios is empty.")

    # ── 第四步：逐场景串行执行重建 ──
    # 串行而非并发：每个场景都要独占 Milvus collection（--reset-collections 时会 drop/create），
    # 并发会导致 collection 互相干扰，串行保证资源隔离干净
    results: list[ScenarioRunResult] = []
    for scenario_id in scenarios:
        result = run_one(args, scenario_id)
        results.append(result)
        if not result.ok and not args.continue_on_failure:
            # 首个失败即中断，但已完成的场景结果仍然保留，会在汇总表中展示
            break

    # ── 第五步：打印汇总表格 ──
    # 格式：场景名(28字符) 状态(12字符) 耗时(8字符)
    # 状态：OK 或 FAILED(退出码)，便于快速定位失败场景
    print("\nBatch rebuild summary")
    print("-" * 88)
    for result in results:
        status = "OK" if result.ok else f"FAILED({result.returncode})"
        print(f"{result.scenario_id:28s} {status:12s} {result.elapsed_seconds:8.2f}s")

    # ── 第六步：有失败场景时返回非零退出码 ──
    # CI/CD 根据退出码判定批量重建是否成功，非 0 触发告警或阻断发布
    failed = [item for item in results if not item.ok]
    if failed:
        print("\nFailed scenarios:", ", ".join(item.scenario_id for item in failed))
        sys.exit(1)


if __name__ == "__main__":
    # 当脚本直接运行时，__name__ == "__main__"
    main()
