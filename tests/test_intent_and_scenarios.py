"""意图识别、问题类别和场景配置的纯逻辑测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from qa_core.indexing.source_normalization import normalize_faq_source
from qa_core.intent.classifier import classify_direct_intent, classify_intent, infer_source
from qa_core.intent.question_category import infer_question_category
from qa_core.pipeline.query_input import normalize_user_query
from qa_core.scenarios.boundary import detect_scenario_boundary, detect_source_boundary, rank_source_matches, score_source_map
from qa_core.scenarios.registry import ScenarioDefinition, ScenarioRegistry, get_scenario_registry


RUNTIME_SCENARIO_ROOT = Path(__file__).resolve().parents[1] / "scenarios"


class QuestionCategoryTests(unittest.TestCase):
    """验证问题类别能驱动不同检索策略和提示词模板。

    调用顺序：pytest/unittest 测试入口 -> QuestionCategoryTests。
    """

    def test_infer_risk_categories(self) -> None:
        """验证各类风险问题能映射到正确的问题类别。

        调用顺序：pytest/unittest 测试入口 -> QuestionCategoryTests.test_infer_risk_categories()。
        """
        self.assertEqual(infer_question_category("发票和退款规则是什么"), "pricing")
        self.assertEqual(infer_question_category("合同隐私条款有什么风险"), "compliance")
        self.assertEqual(infer_question_category("设备出现温度告警怎么排查"), "troubleshooting")
        self.assertEqual(infer_question_category("企业入职流程都包括哪些内容"), "summary")
        self.assertEqual(infer_question_category("请介绍一下产品功能"), "default")


class ScenarioRegistryTests(unittest.TestCase):
    """验证当前项目冻结的 8 个业务场景均已注册。

    调用顺序：pytest/unittest 测试入口 -> ScenarioRegistryTests。
    """

    def test_all_frozen_business_scenarios_are_registered(self) -> None:
        """验证项目冻结的 8 个业务场景均已注册到场景注册表。

        调用顺序：pytest/unittest 测试入口 -> ScenarioRegistryTests.test_all_frozen_business_scenarios_are_registered()。
        """
        registry = get_scenario_registry()
        scenario_ids = {scenario.scenario_id for scenario in registry.list_scenarios()}
        self.assertEqual(len(scenario_ids), 8)
        self.assertIn("enterprise_knowledge", scenario_ids)
        self.assertIn("saas_support", scenario_ids)
        self.assertIn("equipment_ops", scenario_ids)
        self.assertIn("compliance_qa", scenario_ids)
        self.assertIn("cross_border_risk", scenario_ids)
        self.assertIn("tender_contract_risk", scenario_ids)
        self.assertIn("insurance_claims", scenario_ids)
        self.assertIn("engineering_project_qa", scenario_ids)

    def test_contract_intelligence_exposes_only_contract_runtime_scenario(self) -> None:
        """历史配置可离线复用，但生产运行时只允许合同场景。"""
        runtime_registry = ScenarioRegistry(RUNTIME_SCENARIO_ROOT)
        self.assertEqual(
            [scenario.scenario_id for scenario in runtime_registry.list_scenarios()],
            ["tender_contract_risk"],
        )

    def test_enterprise_source_patterns_are_used_for_source_inference(self) -> None:
        """验证企业知识场景的源推理模式能正确匹配到对应来源。

        调用顺序：pytest/unittest 测试入口 -> ScenarioRegistryTests.test_enterprise_source_patterns_are_used_for_source_inference()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        self.assertEqual(infer_source("新人入职流程怎么走", scenario), "hr")
        self.assertEqual(infer_source("VPN 连不上怎么处理", scenario), "it")
        self.assertEqual(infer_source("员工报销需要准备哪些材料", scenario), "finance")

    def test_source_inference_exposes_ranked_candidates_for_diagnostics(self) -> None:
        """验证源推理暴露排名候选项供诊断使用。

        调用顺序：pytest/unittest 测试入口 -> ScenarioRegistryTests.test_source_inference_exposes_ranked_candidates_for_diagnostics()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")

        matches = rank_source_matches("新人入职流程怎么走，员工报销材料是否也要提交", scenario)

        self.assertGreaterEqual(len(matches), 2)
        self.assertEqual(matches[0].source, "hr")
        self.assertEqual(matches[0].score, matches[1].score)
        self.assertEqual(matches[0].confidence, 1.0)

    def test_source_score_counts_each_case_insensitive_match_once(self) -> None:
        """验证 source 分数不重复计算原文和小写文本的同一次命中。"""
        scenario = ScenarioDefinition.from_mapping(
            {
                "scenario_id": "source_score",
                "display_name": "Source Score",
                "industry": "test",
                "assistant_name": "Test Assistant",
                "business_domain": "Test Domain",
                "support_contact": "Test Support",
                "valid_sources": ["it", "hr"],
                "faq_collection": "source_score_faq",
                "doc_collection": "source_score_doc",
                "source_patterns": {"it": "VPN|账号", "hr": "员工|入职"},
            }
        )

        scores = score_source_map("VPN 账号 vpn", scenario)

        self.assertEqual(scores, {"it": 38})

    def test_source_order_only_breaks_equal_scores(self) -> None:
        """验证 valid_sources 顺序只负责同分决胜，不污染相关性分数。"""
        scenario = ScenarioDefinition.from_mapping(
            {
                "scenario_id": "source_tie",
                "display_name": "Source Tie",
                "industry": "test",
                "assistant_name": "Test Assistant",
                "business_domain": "Test Domain",
                "support_contact": "Test Support",
                "valid_sources": ["first", "second"],
                "faq_collection": "source_tie_faq",
                "doc_collection": "source_tie_doc",
                "source_patterns": {"first": "甲乙", "second": "丙丁"},
            }
        )

        matches = rank_source_matches("甲乙和丙丁", scenario)

        self.assertEqual([(item.source, item.score) for item in matches], [("first", 12), ("second", 12)])

    def test_cross_border_source_patterns_are_used_for_source_inference(self) -> None:
        """验证跨境合规场景的源推理模式能正确匹配到制裁和支付来源。

        调用顺序：pytest/unittest 测试入口 -> ScenarioRegistryTests.test_cross_border_source_patterns_are_used_for_source_inference()。
        """
        scenario = get_scenario_registry().resolve("cross_border_risk")
        self.assertEqual(infer_source("交易对手命中制裁名单怎么办", scenario), "sanction")
        self.assertEqual(infer_source("信用证不符点如何处理", scenario), "payment")

    def test_tender_and_insurance_patterns_are_used_for_source_inference(self) -> None:
        """验证招投标和保险场景的源推理模式。

        调用顺序：pytest/unittest 测试入口 -> ScenarioRegistryTests.test_tender_and_insurance_patterns_are_used_for_source_inference()。
        """
        tender = get_scenario_registry().resolve("tender_contract_risk")
        insurance = get_scenario_registry().resolve("insurance_claims")
        self.assertEqual(infer_source("投标文件缺少授权书有什么风险", tender), "bidding")
        self.assertEqual(infer_source("非标准付款条款需要谁复核", tender), "contract")
        self.assertEqual(infer_source("理赔申请需要哪些材料", insurance), "claim_material")
        self.assertEqual(infer_source("哪些情况可能属于除外责任", insurance), "exclusion")

    def test_engineering_project_patterns_are_used_for_source_inference(self) -> None:
        """验证工程项目场景的源推理模式能正确匹配到图纸、质量、安全等来源。

        调用顺序：pytest/unittest 测试入口 -> ScenarioRegistryTests.test_engineering_project_patterns_are_used_for_source_inference()。
        """
        scenario = get_scenario_registry().resolve("engineering_project_qa")
        self.assertEqual(infer_source("图纸变更后旧版本还能作为施工依据吗", scenario), "drawing")
        self.assertEqual(infer_source("隐蔽工程验收需要哪些资料", scenario), "quality")
        self.assertEqual(infer_source("高处作业前必须做哪些安全资料", scenario), "safety")
        self.assertEqual(infer_source("施工图纸和强制性规范冲突时怎么办", scenario), "specification")

    def test_inactive_scenario_cannot_drive_runtime_boundary(self) -> None:
        """验证离线保留的历史场景不会参与 Contract Intelligence 运行时边界判断。

        调用顺序：pytest/unittest 测试入口 -> ScenarioRegistryTests.test_scenario_boundary_detects_question_from_other_business_scene()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        decision = detect_scenario_boundary("安全技术交底只有口头说明可以吗？", scenario)
        self.assertFalse(decision.crossed)
        self.assertEqual(decision.matched_scenario_id, "")
        self.assertEqual(decision.matched_source, "")

    def test_inactive_scenario_cannot_override_weak_runtime_match(self) -> None:
        """验证历史场景的强命中也不能覆盖单一合同运行时。"""
        scenario = get_scenario_registry().resolve("enterprise_knowledge")

        decision = detect_scenario_boundary("入职后需要做安全技术交底和高处作业防护吗？", scenario)

        self.assertFalse(decision.crossed)
        self.assertEqual(decision.matched_scenario_id, "")
        self.assertEqual(decision.matched_source, "")

    def test_source_boundary_detects_wrong_selected_source(self) -> None:
        """验证源边界检测能识别错误选择的来源。

        调用顺序：pytest/unittest 测试入口 -> ScenarioRegistryTests.test_source_boundary_detects_wrong_selected_source()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        decision = detect_source_boundary("员工报销需要准备哪些材料？", scenario, "hr")
        self.assertTrue(decision.mismatched)
        self.assertEqual(decision.matched_source, "finance")

    def test_source_boundary_allows_matching_selected_source(self) -> None:
        """验证源边界检测对正确选择的来源通过。

        调用顺序：pytest/unittest 测试入口 -> ScenarioRegistryTests.test_source_boundary_allows_matching_selected_source()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        decision = detect_source_boundary("员工报销需要准备哪些材料？", scenario, "finance")
        self.assertFalse(decision.mismatched)


class IntentClassifierTests(unittest.TestCase):
    """验证当前意图识别输出通用知识问答意图，不再输出课程咨询意图。

    调用顺序：pytest/unittest 测试入口 -> IntentClassifierTests。
    """

    def test_query_normalization_keeps_pure_greeting_but_strips_opening_words(self) -> None:
        """验证查询归一化保留纯问候语但去除开场白词汇。

        调用顺序：pytest/unittest 测试入口 -> IntentClassifierTests.test_query_normalization_keeps_pure_greeting_but_strips_opening_words()。
        """
        cases = [
            ("你好", "你好"),
            ("你好，请问新人入职流程有哪些？", "新人入职流程有哪些？"),
            ("您好，VPN 连不上怎么处理？", "VPN 连不上怎么处理？"),
            ("在吗，员工报销需要准备哪些材料？", "员工报销需要准备哪些材料？"),
            ("麻烦帮我看下账号权限怎么申请", "账号权限怎么申请"),
            ("帮我分析一下这个问题", "帮我分析一下这个问题"),
            ("转人工", "转人工"),
        ]

        for raw_query, expected in cases:
            with self.subTest(raw_query=raw_query):
                self.assertEqual(normalize_user_query(raw_query), expected)

    def test_direct_intent_guard_handles_protocol_questions_before_retrieval(self) -> None:
        """验证直接意图守卫在检索前处理协议类问题（问候、转人工、超出范围）。

        调用顺序：pytest/unittest 测试入口 -> IntentClassifierTests.test_direct_intent_guard_handles_protocol_questions_before_retrieval()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")

        greeting = classify_direct_intent("你好", scenario)
        self.assertIsNotNone(greeting)
        self.assertEqual(greeting.intent, "GREETING")
        self.assertTrue(greeting.direct_answer)

        human_service = classify_direct_intent("转人工", scenario)
        self.assertIsNotNone(human_service)
        self.assertEqual(human_service.intent, "HUMAN_SERVICE")
        self.assertTrue(human_service.direct_answer)
        self.assertIsNone(human_service.suggested_source)

        hr_human_service = classify_direct_intent("HR客服电话", scenario)
        self.assertIsNotNone(hr_human_service)
        self.assertEqual(hr_human_service.intent, "HUMAN_SERVICE")
        self.assertTrue(hr_human_service.direct_answer)
        self.assertIsNone(hr_human_service.suggested_source)

        out_of_scope = classify_direct_intent("彩票怎么买", scenario)
        self.assertIsNotNone(out_of_scope)
        self.assertEqual(out_of_scope.intent, "OUT_OF_SCOPE")
        self.assertTrue(out_of_scope.direct_answer)
        self.assertIsNone(out_of_scope.suggested_source)

        self.assertIsNone(classify_direct_intent("新人入职流程怎么走", scenario))
        self.assertIsNone(classify_direct_intent("你好，新人入职流程有哪些？", scenario))

    def test_route_layer_owns_direct_intents(self) -> None:
        """验证路由层拥有直接意图，同时检索层仍返回业务意图。

        调用顺序：pytest/unittest 测试入口 -> IntentClassifierTests.test_route_layer_owns_direct_intents()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")

        query = "新人入职客服电话"
        direct = classify_direct_intent(query, scenario)
        retrieval_intent = classify_intent(query, [], scenario)

        self.assertIsNotNone(direct)
        self.assertEqual(direct.intent, "HUMAN_SERVICE")
        self.assertTrue(direct.direct_answer)
        self.assertIsNone(direct.suggested_source)
        self.assertEqual(retrieval_intent.intent, "KNOWLEDGE_QUERY")
        self.assertEqual(retrieval_intent.suggested_source, "hr")

    def test_business_knowledge_question_uses_knowledge_intent(self) -> None:
        """验证业务知识类问题使用 KNOWLEDGE_QUERY 意图。

        调用顺序：pytest/unittest 测试入口 -> IntentClassifierTests.test_business_knowledge_question_uses_knowledge_intent()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        result = classify_intent("新人入职流程怎么走", [], scenario)
        self.assertEqual(result.intent, "KNOWLEDGE_QUERY")
        self.assertEqual(result.suggested_source, "hr")

    def test_greeting_prefix_business_question_keeps_business_intent(self) -> None:
        """验证带问候前缀的业务问题仍保持业务意图而非问候意图。

        调用顺序：pytest/unittest 测试入口 -> IntentClassifierTests.test_greeting_prefix_business_question_keeps_business_intent()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        effective_query = normalize_user_query("你好，新人入职流程有哪些？")
        result = classify_intent(effective_query, [], scenario)

        self.assertNotEqual(result.intent, "GREETING")
        self.assertEqual(result.intent, "FAQ_QUERY")
        self.assertEqual(result.reason, "source_question_shape_rule")
        self.assertEqual(result.suggested_source, "hr")

    def test_strong_rules_split_faq_knowledge_and_default_paths(self) -> None:
        """验证强规则能正确区分 FAQ、知识和默认路径。

        调用顺序：pytest/unittest 测试入口 -> IntentClassifierTests.test_strong_rules_split_faq_knowledge_and_default_paths()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")

        cases = [
            ("开票失败怎么办", "FAQ_QUERY", "strong_faq_rule", 0.82),
            ("VPN是什么", "FAQ_QUERY", "direct_faq_shape_rule", 0.86),
            ("新人入职流程说明", "KNOWLEDGE_QUERY", "strong_knowledge_rule", 0.84),
            ("帮我分析一下这个问题", "KNOWLEDGE_QUERY", "default_knowledge", 0.6),
        ]

        for query, expected_intent, expected_reason, expected_score in cases:
            with self.subTest(query=query):
                result = classify_intent(query, [], scenario)
                self.assertEqual(result.intent, expected_intent)
                self.assertEqual(result.reason, expected_reason)
                self.assertEqual(result.rule_score, expected_score)

    def test_short_direct_faq_shape_prefers_faq_intent(self) -> None:
        """验证简短直接的 FAQ 形态问题优先使用 FAQ 意图。

        调用顺序：pytest/unittest 测试入口 -> IntentClassifierTests.test_short_direct_faq_shape_prefers_faq_intent()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        result = classify_intent("员工报销需要准备哪些材料？", [], scenario)
        self.assertEqual(result.intent, "FAQ_QUERY")
        self.assertEqual(result.reason, "source_question_shape_rule")
        self.assertEqual(result.rule_score, 0.85)
        self.assertEqual(result.suggested_source, "finance")
        self.assertGreater(result.source_score, 0)
        self.assertGreater(result.source_confidence, 0)
        self.assertTrue(result.source_candidates)
        self.assertIn("source_candidates", result.as_dict())

    def test_saas_faq_question_uses_faq_intent_with_scenario_source(self) -> None:
        """验证 SaaS 支持场景的 FAQ 问题使用对应的场景源。

        调用顺序：pytest/unittest 测试入口 -> IntentClassifierTests.test_saas_faq_question_uses_faq_intent_with_scenario_source()。
        """
        scenario = get_scenario_registry().resolve("saas_support")
        result = classify_intent("发票什么时候可以开", [], scenario)
        self.assertEqual(result.intent, "FAQ_QUERY")
        self.assertEqual(result.suggested_source, "billing")

    def test_source_question_shape_uses_faq_rule_for_routing(self) -> None:
        """验证带有来源线索的问题形态使用 FAQ 规则进行路由。

        调用顺序：pytest/unittest 测试入口 -> IntentClassifierTests.test_source_question_shape_uses_faq_rule_for_routing()。
        """
        scenario = get_scenario_registry().resolve("cross_border_risk")
        result = classify_intent("交易对手命中制裁名单怎么办", [], scenario)
        self.assertEqual(result.intent, "FAQ_QUERY")
        self.assertEqual(result.reason, "source_question_shape_rule")
        self.assertEqual(result.suggested_source, "sanction")


class FaqIngestionTests(unittest.TestCase):
    """验证 FAQ CSV 入库前的分类归一化规则。

    调用顺序：pytest/unittest 测试入口 -> FaqIngestionTests。
    """

    def test_normalize_faq_source_uses_scenario_valid_sources_and_patterns(self) -> None:
        """验证 FAQ 来源归一化使用场景的有效来源和匹配模式。

        调用顺序：pytest/unittest 测试入口 -> FaqIngestionTests.test_normalize_faq_source_uses_scenario_valid_sources_and_patterns()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        self.assertEqual(normalize_faq_source("finance", scenario=scenario), "finance")
        self.assertEqual(normalize_faq_source("财务报销", scenario=scenario), "finance")
        self.assertEqual(normalize_faq_source("账号权限", scenario=scenario, question="VPN 账号权限怎么申请"), "it")
        self.assertEqual(normalize_faq_source("入职流程", scenario=scenario), "hr")

    def test_normalize_faq_source_rejects_empty_source(self) -> None:
        """验证空来源被 normalize_faq_source 拒绝。

        调用顺序：pytest/unittest 测试入口 -> FaqIngestionTests.test_normalize_faq_source_rejects_empty_source()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        with self.assertRaises(ValueError):
            normalize_faq_source("", scenario=scenario)
