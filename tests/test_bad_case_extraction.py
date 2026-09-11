"""Bad Case 提取脚本的轻量单元测试。

这些测试只验证 JSON 报告到 eval_set 样本的转换，不加载 QAService、Milvus、模型或
LangChain 文档解析依赖。
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.extract_bad_cases_from_report import select_bad_cases


def test_select_bad_cases_preserves_original_expected_source_contains(tmp_path: Path) -> None:
    """失败样本进入 Bad Case 时，保留原始评测集里的 expected_source_contains。

    调用顺序：pytest -> select_bad_cases() -> load_original_cases() -> build_bad_case()。
    """
    dataset_path = tmp_path / "eval_cases.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "case_001",
                    "query": "VPN 排查要记录哪些信息？",
                    "scenario_id": "enterprise_knowledge",
                    "source_filter": "it",
                    "expected_source_contains": ["it_support.md", "VPN 连接排查"],
                    "expected_keywords": ["客户端版本", "账号锁定"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report = {
        "dataset": str(dataset_path),
        "rows": [
            {
                "case_id": "case_001",
                "question": "VPN 排查要记录哪些信息？",
                "source_recall_hit": False,
                "keyword_coverage": 0.5,
                "expected_keywords": ["客户端版本", "账号锁定"],
                "hit_type_matched": True,
                "source_inference_matched": True,
                "prompt_profile_matched": True,
            }
        ],
    }

    bad_cases = select_bad_cases(report, min_keyword_coverage=0.7, max_items=0)

    assert len(bad_cases) == 1
    assert bad_cases[0]["case_id"] == "bad_case_001"
    assert bad_cases[0]["query"] == "VPN 排查要记录哪些信息？"
    assert bad_cases[0]["expected_source_contains"] == ["it_support.md", "VPN 连接排查"]
    assert "预期来源没有召回" in bad_cases[0]["bad_case_reasons"]


def test_select_bad_cases_ignores_clean_rows() -> None:
    """指标全部通过的样本不会进入 Bad Case。

    调用顺序：pytest -> select_bad_cases() -> failure_reasons()。
    """
    report = {
        "dataset": "",
        "rows": [
            {
                "case_id": "case_ok",
                "question": "报销材料有哪些？",
                "source_recall_hit": True,
                "keyword_coverage": 1.0,
                "expected_keywords": ["发票", "审批"],
                "hit_type_matched": True,
                "source_inference_matched": True,
                "prompt_profile_matched": True,
            }
        ],
    }

    bad_cases = select_bad_cases(report, min_keyword_coverage=0.7, max_items=0)

    assert bad_cases == []
