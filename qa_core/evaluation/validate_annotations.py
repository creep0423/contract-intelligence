"""人工 Gold annotation 校验 CLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qa_core.evaluation.loader import AnnotationValidationError, load_benchmark
from qa_core.evaluation.runner import DEFAULT_MANIFEST


def main() -> int:
    parser = argparse.ArgumentParser(description="校验合同 Benchmark manifest、文档和 Gold 引用")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--suite", choices=("fast", "full"), default="full")
    args = parser.parse_args()
    try:
        loaded = load_benchmark(Path(args.manifest), suite=args.suite)
    except AnnotationValidationError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(
        json.dumps(
            {
                "valid": True,
                "benchmark_version": loaded.manifest.benchmark_version,
                "dataset_version": loaded.manifest.dataset_version,
                "case_count": len(loaded.cases),
                "synthetic_count": sum(case.synthetic for case in loaded.cases),
                "real_uat_count": sum(not case.synthetic for case in loaded.cases),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
