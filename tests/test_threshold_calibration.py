"""数据驱动阈值校准测试。"""

from __future__ import annotations

from scripts.intent.calibrate_thresholds import (
    BinaryObservation,
    evaluate_binary_threshold,
    select_faq_threshold,
    select_intent_model_threshold,
)


def _observations() -> list[BinaryObservation]:
    return [
        BinaryObservation("p1", 0.91, True),
        BinaryObservation("p2", 0.87, True),
        BinaryObservation("p3", 0.83, True),
        BinaryObservation("p4", 0.80, True),
        BinaryObservation("n1", 0.79, False),
        BinaryObservation("n2", 0.68, False),
        BinaryObservation("n3", 0.55, False),
        BinaryObservation("n4", 0.40, False),
    ]


def test_binary_threshold_metrics_distinguish_false_direct_and_missed_direct() -> None:
    """验证阈值指标能够区分误直出和漏直出。

    调用顺序：业务模块或命令行入口 -> test_binary_threshold_metrics_distinguish_false_direct_and_missed_direct()。
    """
    metrics = evaluate_binary_threshold(_observations(), 0.82)
    assert metrics["tp"] == 3
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.75


def test_selector_obeys_precision_recall_and_false_direct_constraints() -> None:
    """验证阈值候选同时满足精确率、召回率和误直出约束。

    调用顺序：业务模块或命令行入口 -> test_selector_obeys_precision_recall_and_false_direct_constraints()。
    """
    result = select_faq_threshold(
        _observations(),
        min_precision=1.0,
        min_recall=0.75,
        max_false_direct_rate=0.0,
    )
    assert result["ok"] is True
    assert result["selected"]["threshold"] >= 0.80
    assert result["selected"]["precision"] == 1.0
    assert result["selected"]["recall"] >= 0.75


def test_selector_does_not_emit_candidate_without_observations() -> None:
    """验证缺少样本时不会产生拍脑门阈值。

    调用顺序：业务模块或命令行入口 -> test_selector_does_not_emit_candidate_without_observations()。
    """
    result = select_faq_threshold([], min_precision=0.95, min_recall=0.8, max_false_direct_rate=0.05)
    assert result["ok"] is False
    assert result["selected"] is None


def test_intent_selector_keeps_current_threshold_when_candidates_tie() -> None:
    """验证候选效果相同时保持当前生产阈值。

    调用顺序：业务模块或命令行入口 -> test_intent_selector_keeps_current_threshold_when_candidates_tie()。
    """
    cases = [{"expected_intent": "FAQ_QUERY", "expected_route": "retrieval", "critical": True}]
    rows = [
        {
            "intent": {
                "intent": "FAQ_QUERY",
                "candidate_intents": [
                    {"source": "rule", "intent": "FAQ_QUERY", "score": 0.82, "reason": "strong_faq_rule"},
                    {"source": "model", "intent": "FAQ_QUERY", "score": 0.90, "reason": "bert"},
                ],
            }
        }
    ]
    result = select_intent_model_threshold(cases, rows, current_threshold=0.55)
    assert result["selected"]["threshold"] == 0.55
