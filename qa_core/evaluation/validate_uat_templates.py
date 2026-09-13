"""Contract Intelligence Real UAT 模板校验 CLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qa_core.evaluation.uat_templates import TEMPLATE_FILE_MODELS, validate_template_directory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_ROOT = PROJECT_ROOT / "data_packs" / "contract_uat_templates"


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 Industrial Equipment Real UAT 安全模板")
    parser.add_argument("--root", default=str(DEFAULT_TEMPLATE_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    errors = validate_template_directory(root)
    print(
        json.dumps(
            {
                "valid": not errors,
                "template_root": str(root),
                "template_files": sorted(TEMPLATE_FILE_MODELS),
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
