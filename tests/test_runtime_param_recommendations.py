"""运行时参数建议脚本的轻量测试。"""

from __future__ import annotations

from scripts.quality.recommend_runtime_params import build_core_chain_advice, build_recommendation_report


def _calibration_report(*, sufficient: bool = True) -> dict:
    return {
        "current_policy": {
            "faq_direct_score_threshold": 0.72,
            "intent_model_min_score": 0.55,
        },
        "candidate_policy": {
            "faq_direct_score_threshold": 0.84,
            "intent_model_min_score": 0.62,
        },
        "faq_direct": {
            "ok": True,
            "sample_counts": {
                "positive": 4 if sufficient else 3,
                "negative": 4 if sufficient else 3,
                "collection_failures": 0,
            },
        },
        "intent_model": {
            "ok": True,
            "selected": {"total": 10},
        },
    }


def _clean_eval_report() -> dict:
    return {"ok": True, "errors": 0, "rows": []}


def _clean_intent_report() -> dict:
    return {"ok": True, "critical_failure_count": 0}


def test_insufficient_faq_samples_keep_current_threshold() -> None:
    """FAQ 校准样本不足时，不采纳候选直出阈值。"""
    report = build_recommendation_report(
        calibration_report=_calibration_report(sufficient=False),
        evaluation_report=_clean_eval_report(),
        intent_policy_report=_clean_intent_report(),
        min_faq_positive=4,
        min_faq_negative=4,
        min_intent_cases=10,
        min_delta=0.01,
    )

    faq_item = report["recommendations"][0]
    assert faq_item["parameter"] == "FAQ_DIRECT_SCORE_THRESHOLD"
    assert faq_item["action"] == "keep_current"
    assert faq_item["recommended"] == 0.72
    assert "positive_samples=3<required=4" in faq_item["reason"]


def test_sufficient_reports_allow_candidate_after_review() -> None:
    """评测和样本量都通过时，输出可复核候选值。"""
    report = build_recommendation_report(
        calibration_report=_calibration_report(sufficient=True),
        evaluation_report=_clean_eval_report(),
        intent_policy_report=_clean_intent_report(),
        min_faq_positive=4,
        min_faq_negative=4,
        min_intent_cases=10,
        min_delta=0.01,
    )

    actions = [item["action"] for item in report["recommendations"][:2]]
    assert actions == ["apply_candidate_after_review", "apply_candidate_after_review"]
    assert report["recommendations"][0]["recommended"] == 0.84
    assert report["recommendations"][1]["recommended"] == 0.62


def test_intent_policy_failure_blocks_candidate_changes() -> None:
    """意图策略门禁失败时，不允许直接采纳阈值候选。"""
    report = build_recommendation_report(
        calibration_report=_calibration_report(sufficient=True),
        evaluation_report=_clean_eval_report(),
        intent_policy_report={"ok": False, "critical_failure_count": 1},
        min_faq_positive=4,
        min_faq_negative=4,
        min_intent_cases=10,
        min_delta=0.01,
    )

    assert report["safe_to_apply_count"] == 0
    assert all(item["action"] == "keep_current" for item in report["recommendations"][:2])


def test_core_chain_false_direct_creates_raise_threshold_advice() -> None:
    """主链路出现误直出时，给出提高直出阈值的方向性建议。"""
    advice = build_core_chain_advice(
        {
            "rows": [
                {
                    "case_id": "case_false_direct",
                    "hit_type": "faq_direct",
                    "expected_hit_type": "rag",
                }
            ]
        }
    )

    assert advice
    assert advice[0]["action"] == "raise_threshold_or_add_exact_guard"
    assert advice[0]["evidence_cases"] == ["case_false_direct"]
