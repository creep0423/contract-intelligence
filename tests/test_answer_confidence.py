"""最终答案置信度测试。"""

from __future__ import annotations

import unittest

from langchain_core.documents import Document

from qa_core.pipeline.confidence import (
    calculate_evidence_confidence,
    calculate_generation_confidence,
    combine_answer_confidence,
    confidence_level,
    faq_exact_match,
    normalize_retrieval_score,
)


class AnswerConfidenceTests(unittest.TestCase):
    """验证 answer_confidence 和检索 score 分离。

    调用顺序：pytest/unittest 测试入口 -> AnswerConfidenceTests。
    """

    def test_faq_exact_match_has_high_confidence(self) -> None:
        """验证 FAQ 标准问题精确匹配会得到高置信度。

        调用顺序：pytest/unittest 测试入口 -> AnswerConfidenceTests.test_faq_exact_match_has_high_confidence()。
        """
        result = calculate_evidence_confidence(
            hit_type="faq_direct",
            retrieval_top_score=0.35,
            context_count=1,
            source_count=1,
            intent_rule_score=0.98,
            query="员工报销需要准备哪些材料？",
            raw_query="员工报销需要准备哪些材料？",
            rewritten_query="员工报销需要准备哪些材料？",
            faq_exact_match=True,
        )

        self.assertEqual(result.level, "high")
        self.assertEqual(result.score, 0.95)
        self.assertIn("faq_exact_match", result.reasons)

    def test_history_rewrite_lowers_but_does_not_replace_retrieval_signal(self) -> None:
        """验证追问改写会降低最终置信度，但不改变检索排序分。

        调用顺序：pytest/unittest 测试入口 -> AnswerConfidenceTests.test_history_rewrite_lowers_but_does_not_replace_retrieval_signal()。
        """
        normal = calculate_evidence_confidence(
            hit_type="rag",
            retrieval_top_score=0.8,
            context_count=3,
            source_count=2,
            intent_rule_score=0.84,
            query="新人入职流程怎么走？",
            raw_query="新人入职流程怎么走？",
            rewritten_query="新人入职流程怎么走？",
        )
        rewritten = calculate_evidence_confidence(
            hit_type="rag",
            retrieval_top_score=0.8,
            context_count=3,
            source_count=2,
            intent_rule_score=0.84,
            query="这个呢？",
            raw_query="这个呢？",
            rewritten_query="新人入职流程怎么走？",
        )

        self.assertLess(rewritten.score, normal.score)
        self.assertEqual(rewritten.signals["normalized_retrieval_score"], normal.signals["normalized_retrieval_score"])
        self.assertIn("history_rewrite_used", rewritten.reasons)

    def test_insufficient_context_confidence_is_low(self) -> None:
        """验证无上下文分支会输出低置信度。

        调用顺序：pytest/unittest 测试入口 -> AnswerConfidenceTests.test_insufficient_context_confidence_is_low()。
        """
        result = calculate_evidence_confidence(
            hit_type="insufficient_context",
            retrieval_top_score=0.2,
            context_count=0,
            source_count=0,
            intent_rule_score=0.6,
            query="完全没有资料的问题",
            raw_query="完全没有资料的问题",
            rewritten_query="完全没有资料的问题",
        )

        self.assertEqual(result.level, "low")
        self.assertIn("insufficient_context", result.reasons)

    def test_faq_exact_match_compares_standard_question(self) -> None:
        """验证 FAQ 精确匹配只看标准问题字段。

        调用顺序：pytest/unittest 测试入口 -> AnswerConfidenceTests.test_faq_exact_match_compares_standard_question()。
        """
        doc = Document(
            page_content="FAQ content",
            metadata={"standard_question": "是否支持开发票？", "answer": "支持。"},
        )

        self.assertTrue(faq_exact_match("是否支持开发票？", doc))
        self.assertFalse(faq_exact_match("可以开发票吗？", doc))

    def test_positive_logits_are_smoothed_for_confidence_only(self) -> None:
        """验证大于 1 的重排 logits 只在置信度中平滑压缩。

        调用顺序：pytest/unittest 测试入口 -> AnswerConfidenceTests.test_positive_logits_are_smoothed_for_confidence_only()。
        """
        self.assertAlmostEqual(normalize_retrieval_score(3.0), 0.75)

    def test_rag_formula_caps_context_and_source_contributions(self) -> None:
        """验证文档 RAG 公式及上下文、来源数量封顶。"""
        result = calculate_evidence_confidence(
            hit_type="rag",
            retrieval_top_score=0.8,
            context_count=10,
            source_count=10,
            intent_rule_score=0.8,
            query="制度要求是什么",
            raw_query="制度要求是什么",
            rewritten_query="制度要求是什么",
        )

        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.level, "high")

    def test_confidence_level_boundaries_are_stable(self) -> None:
        """验证前端高、中、低等级边界。"""
        self.assertEqual(confidence_level(0.82), "high")
        self.assertEqual(confidence_level(0.55), "medium")
        self.assertEqual(confidence_level(0.54), "low")

    def test_generation_confidence_verifies_inline_citations_and_context_overlap(self) -> None:
        """验证生成后核验会检查行内引用和上下文词面支撑。"""
        docs = [
            Document(
                page_content="新人入职需要完成材料提交、身份核验、账号开通和培训签到。",
                metadata={"source": "hr"},
            )
        ]

        result = calculate_generation_confidence(
            answer="新人入职需要完成材料提交、身份核验、账号开通和培训签到。[1]",
            context_docs=docs,
        )

        self.assertEqual(result["status"], "verified")
        self.assertGreaterEqual(result["score"], 0.82)
        self.assertIn("generation_grounded", result["reasons"])

    def test_generation_confidence_does_not_count_reference_footer_as_inline_citation(self) -> None:
        """验证末尾参考来源不等同于每个事实都有行内引用。"""
        docs = [
            Document(
                page_content="新人入职需要完成材料提交、身份核验。",
                metadata={"source": "hr"},
            )
        ]

        result = calculate_generation_confidence(
            answer="新人入职需要完成材料提交、身份核验。\n\n参考来源：[1] onboarding.md",
            context_docs=docs,
        )

        self.assertIn("low_inline_citation_coverage", result["reasons"])
        self.assertEqual(result["signals"]["inline_citation_count"], 0)

    def test_generation_confidence_flags_invalid_citation_number(self) -> None:
        """验证引用不存在的上下文编号会降低生成后置信度。"""
        docs = [
            Document(
                page_content="报销需要提交发票和审批单。",
                metadata={"source": "finance"},
            )
        ]

        result = calculate_generation_confidence(
            answer="报销需要提交发票和审批单。[3]",
            context_docs=docs,
        )

        self.assertIn("invalid_citation_reference", result["reasons"])
        self.assertEqual(result["signals"]["invalid_citation_numbers"], [3])

    def test_combine_answer_confidence_uses_conservative_min_score(self) -> None:
        """验证最终答案置信度取证据分和生成核验分中更保守的一侧。"""
        evidence = {
            "score": 0.95,
            "level": "high",
            "label": "高",
            "reasons": ["rag_with_context"],
            "signals": {"hit_type": "rag"},
        }
        generation = {
            "score": 0.62,
            "status": "partial",
            "reasons": ["low_inline_citation_coverage"],
            "signals": {"citation_coverage": 0.0},
        }

        result = combine_answer_confidence(evidence, generation)

        self.assertEqual(result["score"], 0.62)
        self.assertEqual(result["level"], "medium")
        self.assertEqual(result["evidence_confidence"]["score"], 0.95)
        self.assertEqual(result["generation_verification"]["score"], 0.62)


if __name__ == "__main__":
    unittest.main()
