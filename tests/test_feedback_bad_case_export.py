# -*- coding: utf-8 -*-
"""用户反馈导出 Bad Case 草稿的单元测试。"""

from __future__ import annotations

from scripts.export_feedback_bad_cases import build_feedback_bad_case, export_feedback_bad_cases


def _feedback_row(feedback_id: int) -> dict:
    """构造一条点踩反馈记录。"""
    return {
        "id": feedback_id,
        "session_id": "session-1",
        "scenario_id": "enterprise_knowledge",
        "tenant_id": "default",
        "dataset_id": "default",
        "question": "VPN 连不上应该怎么处理？",
        "answer": "请联系管理员。",
        "rating": "not_useful",
        "comment": "没有给出排障步骤",
        "sources": [
            {
                "content": "VPN 故障排查资料片段",
                "metadata": {
                    "source": "it",
                    "file_name": "vpn_troubleshooting.pdf",
                    "standard_question": "VPN 连不上怎么处理？",
                },
            }
        ],
        "created_at": "2026-07-20T10:00:00+08:00",
    }


def test_build_feedback_bad_case_preserves_review_context() -> None:
    """点踩反馈导出时保留问题、备注、来源和运行上下文。"""
    bad_case = build_feedback_bad_case(_feedback_row(7))

    assert bad_case["case_id"] == "feedback_7"
    assert bad_case["source_case_id"] == "feedback_7"
    assert bad_case["feedback_id"] == 7
    assert bad_case["query"] == "VPN 连不上应该怎么处理？"
    assert bad_case["scenario_id"] == "enterprise_knowledge"
    assert bad_case["tenant_id"] == "default"
    assert bad_case["dataset_id"] == "default"
    assert bad_case["observed_effective_source"] == "it"
    assert "请联系管理员" in bad_case["observed_answer_preview"]
    assert "没有给出排障步骤" in "；".join(bad_case["bad_case_reasons"])
    assert "vpn_troubleshooting.pdf" in bad_case["observed_sources"][0]
    assert "expected_*" in bad_case["grading_notes"]


def test_export_feedback_bad_cases_respects_max_items() -> None:
    """导出数量可以被 max_items 截断，便于小批量人工复核。"""
    bad_cases = export_feedback_bad_cases([_feedback_row(1), _feedback_row(2)], max_items=1)

    assert [case["case_id"] for case in bad_cases] == ["feedback_1"]
