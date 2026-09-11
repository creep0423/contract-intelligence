# -*- coding: utf-8 -*-
"""V1 封版前一键验收入口。

默认执行不依赖已启动服务和真实 Milvus 运行栈的快速检查：Python 编译、纯单测、
文档构建、文档一致性、工程守护、Docker Compose 配置和 CH08 章节结构检查。

打开 `--include-evaluation` 时，会额外执行主链路评测、评测门禁和 Bad Case 候选导出。
打开 `--include-performance` 时，会额外执行主链路性能基线采集和性能门禁。
如果同时打开 `--include-docker`，这部分评测会在 Compose 的 api 容器里执行，便于和
真实运行环境对齐。

需要真实容器或已启动 API 服务时，再显式打开可选检查：
    python scripts/verify_v1_release.py --include-runtime-tests
    python scripts/verify_v1_release.py --include-evaluation
    python scripts/verify_v1_release.py --include-performance
    python scripts/verify_v1_release.py --include-docker
    python scripts/verify_v1_release.py --include-api --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections.abc import Callable

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.common import (
    PROJECT_ROOT,
    configure_utf8_stdio,
    run_command_step,
    utc_now,
    write_json_file,
)


DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "verification" / "v1_release_latest.json"
DEFAULT_EVALUATION_DATASET = "eval_sets/multi_scenario_smoke.json"
DEFAULT_EVALUATION_REPORT_PATH = "reports/evaluation/v1_release_core_chain_latest.json"
DEFAULT_EVALUATION_GATE_PATH = "reports/verification/v1_release_evaluation_gate.json"
DEFAULT_EVALUATION_BAD_CASES_PATH = "reports/verification/v1_release_bad_cases.json"
DEFAULT_PERFORMANCE_DATASET = "eval_sets/phase1_performance_baseline.json"
DEFAULT_PERFORMANCE_REPORT_PATH = "reports/performance/v1_release_performance_latest.json"
DEFAULT_PERFORMANCE_GATE_PATH = "reports/verification/v1_release_performance_gate.json"


def _python_command(*args: str) -> list[str]:
    """返回当前解释器命令，保证虚拟环境和容器内运行一致。

    调用顺序：命令行入口 -> _python_command()。
    """
    return [sys.executable, *args]


def _docker_ch08_command() -> list[str]:
    """返回 CH08 Docker 集成验收命令。

    调用顺序：命令行入口 -> _docker_ch08_command()。
    """
    return [
        "docker",
        "compose",
        "--env-file",
        ".env.compose",
        "run",
        "--rm",
        "-v",
        f"{PROJECT_ROOT}:/work",
        "-w",
        "/work/codealong/chapters/ch08_milvus_hybrid_search",
        "api",
        "python",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_hybrid_search.py",
    ]


def _compose_api_python_command(*args: str) -> list[str]:
    """返回通过 Compose 的 api 容器执行 Python 的命令。"""
    return [
        "docker",
        "compose",
        "--env-file",
        ".env.compose",
        "run",
        "--rm",
        "api",
        "python",
        *args,
    ]


def _ch12_admin_import_command() -> list[str]:
    """返回 CH12 admin 模块独立导入检查命令。

    调用顺序：命令行入口 -> _ch12_admin_import_command()。
    """
    code = (
        "import qa_core.api.admin as admin; "
        "assert any(route.path == '/api/admin/status' for route in admin.router.routes)"
    )
    return _compose_api_python_command("-c", code)


def _evaluation_runner(args: argparse.Namespace) -> Callable[..., list[str]]:
    """根据执行模式选择评测运行器。"""
    return _compose_api_python_command if args.include_docker else _python_command


def _performance_runner(args: argparse.Namespace) -> Callable[..., list[str]]:
    """根据执行模式选择性能验收运行器。"""
    return _compose_api_python_command if args.include_docker else _python_command


def _evaluation_steps(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """返回主链路评测、门禁和 Bad Case 导出步骤。"""
    runner = _evaluation_runner(args)
    dataset = str(args.evaluation_dataset or DEFAULT_EVALUATION_DATASET)
    report_path = str(args.evaluation_report_path or DEFAULT_EVALUATION_REPORT_PATH)
    gate_path = str(args.evaluation_gate_path or DEFAULT_EVALUATION_GATE_PATH)
    bad_cases_path = str(args.evaluation_bad_cases_path or DEFAULT_EVALUATION_BAD_CASES_PATH)
    return [
        (
            "evaluation_core_chain",
            runner(
                "scripts/evaluate_core_chain.py",
                "--dataset",
                dataset,
                "--limit",
                str(args.evaluation_limit),
                "--output",
                report_path,
            ),
        ),
        (
            "evaluation_gate",
            runner(
                "scripts/quality/check_evaluation_gate.py",
                "--report",
                report_path,
                "--gate-output",
                gate_path,
            ),
        ),
        (
            "evaluation_bad_cases",
            runner(
                "scripts/extract_bad_cases_from_report.py",
                "--report",
                report_path,
                "--output",
                bad_cases_path,
            ),
        ),
    ]


def _performance_steps(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """返回主链路性能基线采集和性能门禁步骤。"""
    runner = _performance_runner(args)
    dataset = str(args.performance_dataset or DEFAULT_PERFORMANCE_DATASET)
    report_path = str(args.performance_report_path or DEFAULT_PERFORMANCE_REPORT_PATH)
    gate_path = str(args.performance_gate_path or DEFAULT_PERFORMANCE_GATE_PATH)
    collect_args = [
        "scripts/quality/collect_performance_baseline.py",
        "--dataset",
        dataset,
        "--limit",
        str(args.performance_limit),
        "--output",
        report_path,
        "--allow-errors",
    ]
    if args.performance_no_warmup:
        collect_args.append("--no-warmup")
    gate_args = [
        "scripts/quality/check_performance_gate.py",
        "--report",
        report_path,
        "--gate-output",
        gate_path,
        "--max-error-rate",
        str(args.performance_max_error_rate),
        "--max-avg-total-ms",
        str(args.performance_max_avg_total_ms),
        "--max-p95-total-ms",
        str(args.performance_max_p95_total_ms),
        "--max-avg-first-token-ms",
        str(args.performance_max_avg_first_token_ms),
        "--max-p95-first-token-ms",
        str(args.performance_max_p95_first_token_ms),
    ]
    return [
        ("performance_baseline", runner(*collect_args)),
        ("performance_gate", runner(*gate_args)),
    ]


def build_steps(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """根据命令行参数生成验收步骤。

    调用顺序：命令行入口 -> build_steps()。
    """
    steps: list[tuple[str, list[str]]] = [
        (
            "python_compile",
            _python_command("-m", "compileall", "app.py", "qa_core", "scripts", "tests", "-q"),
        ),
        (
            "source_comment_coverage",
            _python_command("scripts/course/check_source_comment_coverage.py", "--fail-on-issues"),
        ),
        (
            "v1_maintenance_binding_tests",
            _python_command("-m", "unittest", "discover", "-s", "tests", "-p", "test_v1_maintenance_bindings.py"),
        ),
        (
            "ch08_codealong_pure_contract",
            _python_command(
                "-m",
                "unittest",
                "discover",
                "-s",
                "codealong/chapters/ch08_milvus_hybrid_search/tests",
                "-p",
                "test_collection_contract.py",
            ),
        ),
        ("codealong_alignment", _python_command("scripts/course/check_codealong_alignment.py")),
        ("codealong_compile", _python_command("-m", "compileall", "codealong/chapters", "-q")),
        ("ch12_admin_import", _ch12_admin_import_command()),
        ("mkdocs_v1_build", _python_command("-m", "mkdocs", "build", "--strict")),
        ("docs_consistency", _python_command("scripts/course/check_docs_consistency.py")),
        ("project_guardrails", _python_command("scripts/check_project_guardrails.py")),
        ("docker_compose_config", ["docker", "compose", "--env-file", ".env.compose", "config", "--quiet"]),
        ("git_diff_check", ["git", "diff", "--check"]),
    ]

    if args.include_evaluation:
        steps.extend(_evaluation_steps(args))

    if args.include_performance:
        steps.extend(_performance_steps(args))

    if args.include_runtime_tests:
        steps.extend(
            [
                (
                    "v1_intent_and_scenarios_tests",
                    _python_command("-m", "unittest", "discover", "-s", "tests", "-p", "test_intent_and_scenarios.py"),
                ),
                (
                    "v1_retrieval_and_prompt_tests",
                    _python_command("-m", "unittest", "discover", "-s", "tests", "-p", "test_retrieval_and_prompt.py"),
                ),
                (
                    "v1_stream_event_tests",
                    _python_command("-m", "unittest", "discover", "-s", "tests", "-p", "test_stream_query_events.py"),
                ),
            ]
        )

    if args.include_docker:
        steps.append(("ch08_codealong_docker_collection_contract", _docker_ch08_command()))

    if args.include_api:
        api_args = ["--base-url", args.base_url, "--scenario", args.scenario]
        if args.admin_token:
            api_args.extend(["--admin-token", args.admin_token])
        steps.extend(
            [
                ("acceptance_smoke_api", _python_command("scripts/acceptance_smoke.py", *api_args)),
                ("api_e2e_smoke", _python_command("scripts/api_e2e_smoke.py", *api_args)),
            ]
        )

    return steps


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。

    调用顺序：命令行入口 -> build_parser()。
    """
    parser = argparse.ArgumentParser(description="Run V1 release verification checks.")
    parser.add_argument("--include-runtime-tests", action="store_true", help="运行依赖项目完整 Python 依赖的 V1 运行时单测。")
    parser.add_argument("--include-evaluation", action="store_true", help="运行 V1 主链路评测、门禁和 Bad Case 导出。")
    parser.add_argument("--include-performance", action="store_true", help="运行 V1 主链路性能基线和性能门禁。")
    parser.add_argument("--include-docker", action="store_true", help="运行需要 Docker Compose 的集成验收。")
    parser.add_argument("--include-api", action="store_true", help="运行需要已启动 FastAPI 服务的 API 验收。")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API 验收目标服务地址。")
    parser.add_argument("--admin-token", default="", help="API 验收管理令牌；为空时由子脚本读取运行配置。")
    parser.add_argument("--scenario", default="enterprise_knowledge", help="API 验收使用的场景 ID。")
    parser.add_argument("--evaluation-dataset", default=str(DEFAULT_EVALUATION_DATASET), help="V1 评测集路径。")
    parser.add_argument("--evaluation-limit", type=int, default=20, help="V1 评测样本数量。")
    parser.add_argument("--evaluation-report-path", default=str(DEFAULT_EVALUATION_REPORT_PATH), help="V1 评测报告输出路径。")
    parser.add_argument("--evaluation-gate-path", default=str(DEFAULT_EVALUATION_GATE_PATH), help="V1 评测门禁输出路径。")
    parser.add_argument("--evaluation-bad-cases-path", default=str(DEFAULT_EVALUATION_BAD_CASES_PATH), help="V1 Bad Case 候选输出路径。")
    parser.add_argument("--performance-dataset", default=str(DEFAULT_PERFORMANCE_DATASET), help="V1 性能样本集路径。")
    parser.add_argument("--performance-limit", type=int, default=6, help="V1 性能样本数量。")
    parser.add_argument("--performance-report-path", default=str(DEFAULT_PERFORMANCE_REPORT_PATH), help="V1 性能基线报告输出路径。")
    parser.add_argument("--performance-gate-path", default=str(DEFAULT_PERFORMANCE_GATE_PATH), help="V1 性能门禁输出路径。")
    parser.add_argument("--performance-no-warmup", action="store_true", help="性能采集不做预热，冷启动请求也纳入统计。")
    parser.add_argument("--performance-max-error-rate", type=float, default=0.0, help="性能门禁最大错误率。")
    parser.add_argument("--performance-max-avg-total-ms", type=float, default=15000.0, help="性能门禁最大平均总耗时。")
    parser.add_argument("--performance-max-p95-total-ms", type=float, default=30000.0, help="性能门禁最大 P95 总耗时。")
    parser.add_argument("--performance-max-avg-first-token-ms", type=float, default=8000.0, help="性能门禁最大平均首 token 耗时。")
    parser.add_argument("--performance-max-p95-first-token-ms", type=float, default=15000.0, help="性能门禁最大 P95 首 token 耗时。")
    parser.add_argument("--output", default=str(DEFAULT_REPORT_PATH), help="验收报告 JSON 输出路径。")
    return parser


def main() -> int:
    """执行全部验收步骤并输出 JSON 报告。

    调用顺序：命令行入口 -> main()。
    """
    configure_utf8_stdio()
    args = build_parser().parse_args()
    step_results = [run_command_step(name, command) for name, command in build_steps(args)]
    payload = {
        "report_type": "v1_release_verification",
        "created_at": utc_now(),
        "ok": all(step.ok for step in step_results),
        "include_runtime_tests": bool(args.include_runtime_tests),
        "include_evaluation": bool(args.include_evaluation),
        "include_performance": bool(args.include_performance),
        "include_docker": bool(args.include_docker),
        "include_api": bool(args.include_api),
        "evaluation_runner": "docker" if args.include_docker else "host",
        "evaluation_dataset": str(args.evaluation_dataset),
        "evaluation_limit": int(args.evaluation_limit),
        "evaluation_report_path": str(args.evaluation_report_path),
        "evaluation_gate_path": str(args.evaluation_gate_path),
        "evaluation_bad_cases_path": str(args.evaluation_bad_cases_path),
        "performance_runner": "docker" if args.include_docker else "host",
        "performance_dataset": str(args.performance_dataset),
        "performance_limit": int(args.performance_limit),
        "performance_report_path": str(args.performance_report_path),
        "performance_gate_path": str(args.performance_gate_path),
        "step_count": len(step_results),
        "steps": [
            {
                "name": step.name,
                "ok": step.ok,
                "returncode": step.returncode,
                "elapsed_ms": step.elapsed_ms,
                "command": step.command,
                "stdout_preview": step.stdout_preview,
                "stderr_preview": step.stderr_preview,
            }
            for step in step_results
        ],
    }
    output_path = Path(args.output)
    write_json_file(output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"验收报告已写入：{output_path}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
