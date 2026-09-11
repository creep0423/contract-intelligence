"""验证规则与 BERT 模型的意图仲裁策略。

调用顺序：课程示例、测试或命令行入口 -> 本模块公开接口。
"""

from __future__ import annotations

import unittest

from qa_core.intent.classifier import IntentResult, classify_intent
from qa_core.intent.decision import apply_intent_decision_gateway
from qa_core.scenarios.registry import get_scenario_registry


class IntentDecisionGatewayTests(unittest.TestCase):
    """验证企业级意图决策网关的仲裁和保护规则。

    调用顺序：业务模块 -> IntentDecisionGatewayTests。
    """
    def setUp(self) -> None:
        """准备每个测试共享的场景、输入和依赖替身。

        调用顺序：测试或业务入口 -> IntentDecisionGatewayTests.setUp()。
        """
        self.scenario = get_scenario_registry().resolve("enterprise_knowledge")

    def test_default_policy_keeps_rule_first_path(self) -> None:
        """验证默认仲裁策略仍以规则候选作为稳定底座。

        调用顺序：测试或业务入口 -> IntentDecisionGatewayTests.test_default_policy_keeps_rule_first_path()。
        """
        result = classify_intent("会议室预订入口在哪里", [], self.scenario)
        payload = result.as_dict()

        self.assertEqual(result.intent, "KNOWLEDGE_QUERY")
        self.assertEqual(result.rule_score, 0.6)
        self.assertEqual(result.decision_policy, "rule_model_agreed")
        self.assertGreater(result.confidence, result.rule_score)
        self.assertIn("domain:enterprise_knowledge", result.risk_tags)
        self.assertEqual(payload["candidate_intents"][0]["source"], "rule")
        self.assertTrue(any(candidate["source"] == "model" for candidate in payload["candidate_intents"]))
        self.assertIsNotNone(payload["model_score"])
        self.assertEqual(payload["model_version"], "bert-intent-v1")
        self.assertEqual(payload["policy_version"], "intent-policy-v1-bert")

    def test_gateway_keeps_rule_result_as_final_decision(self) -> None:
        """验证规则与模型一致时输出完整最终决策字段。

        调用顺序：测试或业务入口 -> IntentDecisionGatewayTests.test_gateway_keeps_rule_result_as_final_decision()。
        """
        result = classify_intent("新人入职流程有哪些", [], self.scenario)
        self.assertEqual(result.intent, "FAQ_QUERY")
        self.assertEqual(result.decision_policy, "rule_model_agreed")
        self.assertGreater(result.confidence, result.rule_score)
        self.assertEqual(result.model_version, "bert-intent-v1")
        self.assertTrue(any(candidate["source"] == "model" for candidate in result.candidate_intents))

    def test_model_conflict_keeps_rule_with_guarded_score(self) -> None:
        """验证规则与模型冲突时保持规则并降低最终分数。

        调用顺序：测试或业务入口 -> IntentDecisionGatewayTests.test_model_conflict_keeps_rule_with_guarded_score()。
        """
        rule_result = IntentResult(intent="KNOWLEDGE_QUERY", rule_score=0.84, reason="unit_test_rule")
        result = apply_intent_decision_gateway("新人入职流程有哪些", [], self.scenario, rule_result)

        self.assertEqual(result.intent, "KNOWLEDGE_QUERY")
        self.assertEqual(result.decision_policy, "rule_model_conflict_guarded")
        self.assertEqual(result.confidence, 0.68)
        self.assertEqual(result.model_score, result.as_dict()["model_score"])
        self.assertTrue(any(candidate["source"] == "model" for candidate in result.candidate_intents))

    def test_model_can_promote_default_rule_to_faq_query(self) -> None:
        """验证模型只能接管低确定性的默认知识意图。

        调用顺序：测试或业务入口 -> IntentDecisionGatewayTests.test_model_can_promote_default_rule_to_faq_query()。
        """
        result = classify_intent("如何重置密码", [], self.scenario)

        self.assertEqual(result.intent, "FAQ_QUERY")
        self.assertEqual(result.rule_score, 0.6)
        self.assertEqual(result.decision_policy, "model_assisted_default")
        self.assertEqual(result.reason, "default_knowledge_model_assisted")
        self.assertEqual(result.model_version, "bert-intent-v1")
        self.assertGreater(result.confidence, 0.9)

    def test_model_cannot_promote_follow_up_without_history(self) -> None:
        """验证没有历史时模型不能把问题提升为追问。

        调用顺序：测试或业务入口 -> IntentDecisionGatewayTests.test_model_cannot_promote_follow_up_without_history()。
        """
        result = classify_intent("这个需要多久", [], self.scenario)

        self.assertEqual(result.intent, "KNOWLEDGE_QUERY")
        self.assertEqual(result.decision_policy, "model_follow_up_without_history_guarded")
        self.assertFalse(result.requires_rewrite)

    def test_non_retrieval_intent_uses_deterministic_route_policy(self) -> None:
        """验证直答类意图不调用检索意图模型。

        调用顺序：测试或业务入口 -> IntentDecisionGatewayTests.test_non_retrieval_intent_uses_deterministic_route_policy()。
        """
        rule_result = IntentResult(intent="GREETING", rule_score=1.0, reason="unit_test_direct")
        result = apply_intent_decision_gateway("你好", [], self.scenario, rule_result)

        self.assertEqual(result.intent, "GREETING")
        self.assertEqual(result.decision_policy, "deterministic_route")
        self.assertEqual(result.confidence, 1.0)
        self.assertIsNone(result.model_score)
        self.assertEqual(result.policy_version, "intent-policy-v1-bert")


if __name__ == "__main__":
    unittest.main()
