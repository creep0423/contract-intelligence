"""Bad Case 合并为正式回归样本的单元测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.promote_bad_cases_to_regression import build_plan


def test_promote_bad_cases_replaces_existing_case_and_keeps_traceability(tmp_path: Path) -> None:
    """同名 case_id 命中时，默认替换正式回归集里的样本并保留 bad_case_id。"""

    source_path = tmp_path / "local_bad_cases.json"
    target_path = tmp_path / "enterprise_regression.json"
    output_path = tmp_path / "enterprise_regression_merged.json"

    source_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "bad_case_001",
                    "source_case_id": "case_001",
                    "query": "VPN 排查要记录哪些信息？",
                    "scenario_id": "enterprise_knowledge",
                    "source_filter": "it",
                    "expected_hit_type": "rag",
                    "expected_keywords": ["截图", "账号锁定"],
                    "grading_notes": "答案必须覆盖截图和账号锁定。",
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    target_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "case_001",
                    "query": "旧样本",
                    "scenario_id": "enterprise_knowledge",
                    "expected_keywords": ["旧"],
                },
                {
                    "case_id": "case_keep",
                    "query": "保留样本",
                    "scenario_id": "enterprise_knowledge",
                    "expected_keywords": ["保留"],
                },
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    plan = build_plan(
        SimpleNamespace(
            source=str(source_path),
            target=str(target_path),
            output=str(output_path),
            conflict="replace",
            dry_run=False,
        )
    )

    merged = json.loads(output_path.read_text(encoding="utf-8"))

    assert plan["stats"] == {"inserted": 0, "replaced": 1, "skipped": 0}
    assert plan["merged_case_count"] == 2
    assert merged[0]["case_id"] == "case_001"
    assert merged[0]["bad_case_id"] == "bad_case_001"
    assert merged[0]["query"] == "VPN 排查要记录哪些信息？"
    assert merged[0]["expected_hit_type"] == "rag"
    assert merged[0]["grading_notes"] == "答案必须覆盖截图和账号锁定。"
    assert merged[1]["case_id"] == "case_keep"


def test_promote_bad_cases_appends_new_case_without_dry_run(tmp_path: Path) -> None:
    """新的 bad case 会追加进正式回归集。"""

    source_path = tmp_path / "local_bad_cases.json"
    target_path = tmp_path / "enterprise_regression.json"

    source_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "bad_case_002",
                    "source_case_id": "case_002",
                    "question": "报销需要保留哪些材料？",
                    "scenario_id": "enterprise_knowledge",
                    "source_filter": "finance",
                    "expected_hit_type": "faq_direct",
                    "expected_keywords": ["发票", "审批记录"],
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    plan = build_plan(
        SimpleNamespace(
            source=str(source_path),
            target=str(target_path),
            output="",
            conflict="replace",
            dry_run=True,
        )
    )

    assert plan["dry_run"] is True
    assert plan["stats"] == {"inserted": 1, "replaced": 0, "skipped": 0}
    assert not target_path.exists()
