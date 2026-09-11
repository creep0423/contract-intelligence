"""检索过滤、上下文构建、排序和 Prompt Profile 的纯逻辑测试。"""

from __future__ import annotations

import os
import time
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from pymilvus import DataType, Function, FunctionType
from pymilvus.exceptions import MilvusException

from qa_core.config.settings import PROJECT_ROOT, get_settings
from qa_core.config.rules import get_rule_config
from qa_core.governance.data_scope import resolve_data_scope
from qa_core.intent.classifier import IntentResult, classify_intent
from qa_core.intent.question_category import infer_question_category, is_table_query
from qa_core.pipeline.citations import enforce_answer_citations, has_source_citation
from qa_core.pipeline.context import build_context, direct_faq_answer, select_context_docs
from qa_core.pipeline.rag import debug_retrieval
from qa_core.pipeline.query_variants import generate_query_variants
from qa_core.pipeline.runtime import RAGQueryContext
from qa_core.pipeline.steps import decide_route, should_try_faq_fast_path
from qa_core.prompts.selector import build_answer_prompt_profile
from qa_core.retrieval.filters import build_source_expr, validate_source_filter
from qa_core.retrieval.milvus_compat import hybrid_index_params, hybrid_search_params
from qa_core.retrieval.ranking import merge_hits_by_document, normalize_queries, sort_hits_by_score
from qa_core.retrieval.results import RetrievalHit, RetrievalResult
from qa_core.retrieval.store import MilvusHybridStore
from qa_core.retrieval.strategy import build_retrieval_plan
from qa_core.scenarios.registry import get_scenario_registry


class RuntimeSettingsTests(unittest.TestCase):
    """验证运行时路径配置不会受脚本启动目录影响。

    调用顺序：pytest/unittest 测试入口 -> RuntimeSettingsTests。
    """

    def test_relative_model_paths_are_resolved_from_project_root(self) -> None:
        """验证相对模型路径会从项目根目录解析为绝对路径。

        调用顺序：pytest/unittest 测试入口 -> RuntimeSettingsTests.test_relative_model_paths_are_resolved_from_project_root()。
        """
        old_embedding = os.environ.get("EMBEDDING_MODEL_PATH")
        old_reranker = os.environ.get("RERANKER_MODEL_PATH")
        old_intent = os.environ.get("INTENT_MODEL_PATH")
        try:
            os.environ["EMBEDDING_MODEL_PATH"] = "models/bge-m3"
            os.environ["RERANKER_MODEL_PATH"] = "models/bge-reranker-large"
            os.environ["INTENT_MODEL_PATH"] = "models/bert_intent_classifier_v1"
            get_settings.cache_clear()

            settings = get_settings()

            self.assertEqual(settings.embedding_model_path, str(PROJECT_ROOT / "models" / "bge-m3"))
            self.assertEqual(settings.reranker_model_path, str(PROJECT_ROOT / "models" / "bge-reranker-large"))
            self.assertEqual(settings.intent_model_path, str(PROJECT_ROOT / "models" / "bert_intent_classifier_v1"))
        finally:
            if old_embedding is None:
                os.environ.pop("EMBEDDING_MODEL_PATH", None)
            else:
                os.environ["EMBEDDING_MODEL_PATH"] = old_embedding
            if old_reranker is None:
                os.environ.pop("RERANKER_MODEL_PATH", None)
            else:
                os.environ["RERANKER_MODEL_PATH"] = old_reranker
            if old_intent is None:
                os.environ.pop("INTENT_MODEL_PATH", None)
            else:
                os.environ["INTENT_MODEL_PATH"] = old_intent
            get_settings.cache_clear()


class RetrievalFilterTests(unittest.TestCase):
    """验证 source、版本和数据域会进入 Milvus 过滤表达式。

    调用顺序：pytest/unittest 测试入口 -> RetrievalFilterTests。
    """

    def test_build_source_expr_with_scope_and_version(self) -> None:
        """验证构建的过滤表达式包含 source、版本和数据域信息。

        调用顺序：pytest/unittest 测试入口 -> RetrievalFilterTests.test_build_source_expr_with_scope_and_version()。
        """
        scope = resolve_data_scope(tenant_id="tenant_a", dataset_id="dataset_1", visibility="internal", user_role="admin")
        expr = build_source_expr(
            "billing",
            kb_version="kb_v1",
            data_scope=scope,
        )
        self.assertIn('source == "billing"', expr)
        self.assertIn('kb_version == "kb_v1"', expr)
        self.assertIn('tenant_id == "tenant_a"', expr)
        self.assertIn('dataset_id == "dataset_1"', expr)
        self.assertIn('array_contains(allowed_roles, "admin")', expr)

    def test_doc_source_expr_uses_reference_window_only(self) -> None:
        """验证文档源过滤表达式使用引用窗口而非直接版本号。

        调用顺序：pytest/unittest 测试入口 -> RetrievalFilterTests.test_doc_source_expr_uses_reference_window_only()。
        """
        class FakeVersion:
            """模拟版本对象的测试替身。

            调用顺序：pytest/unittest 测试入口 -> FakeVersion。
            """

            version_seq = 8

        class FakeStore:
            """模拟版本存储的测试替身。

            调用顺序：pytest/unittest 测试入口 -> FakeStore。
            """

            def resolve_version(self, requested):
                """测试辅助方法：记录请求参数并返回模拟的版本对象。

                调用顺序：pytest/unittest 测试入口 -> FakeStore.resolve_version()。
                """
                self.requested = requested
                return FakeVersion()

        with patch("qa_core.retrieval.filters.get_kb_version_store", return_value=FakeStore()):
            expr = build_source_expr(
                "hr",
                kb_version="kb_v8",
                scenario_id="enterprise_knowledge",
                source_type="doc",
            )

        self.assertIn('source == "hr"', expr)
        self.assertNotIn('kb_version == "kb_v8"', expr)
        self.assertIn("valid_from_seq <= 8", expr)
        self.assertIn("(valid_to_seq == 0 or valid_to_seq > 8)", expr)

    def test_validate_source_filter_rejects_invalid_source(self) -> None:
        """验证 validate_source_filter 拒绝无效 source 名称。

        调用顺序：pytest/unittest 测试入口 -> RetrievalFilterTests.test_validate_source_filter_rejects_invalid_source()。
        """
        with self.assertRaises(ValueError):
            validate_source_filter("unknown", valid_sources=["billing"])


class ContextBuilderTests(unittest.TestCase):
    """验证 FAQ 直出和上下文构建的保守规则。

    调用顺序：pytest/unittest 测试入口 -> ContextBuilderTests。
    """

    def test_faq_fast_path_rules_are_loaded_from_config_file(self) -> None:
        """验证 FAQ 快速路径规则从配置文件中加载。

        调用顺序：pytest/unittest 测试入口 -> ContextBuilderTests.test_faq_fast_path_rules_are_loaded_from_config_file()。
        """
        rules = get_rule_config()

        self.assertEqual(rules.faq_fast_path.max_chars, 48)
        self.assertTrue(rules.faq_fast_path.hint_matches("报销需要谁审批"))
        self.assertTrue(rules.query_variants.is_short_structured_question("报销流程是什么"))
        self.assertTrue(any("流程" in rule.when_any for rule in rules.query_variants.replacements))
        self.assertEqual(rules.intent_rule_scores.strong_faq, 0.82)
        self.assertEqual(rules.intent_rule_scores.knowledge, 0.84)
        self.assertEqual(rules.intent_rule_scores.source_question_shape, 0.85)
        self.assertEqual(rules.intent_rule_scores.direct_faq_shape, 0.86)
        self.assertEqual(rules.retrieval_strategy.low_rule_score_threshold, 0.70)
        self.assertEqual(rules.retrieval_strategy.low_rule_score_direct_threshold, 0.86)
        self.assertEqual(rules.retrieval_strategy.follow_up_faq_top_k_min, 24)

    def test_direct_faq_answer_requires_exact_match_or_threshold(self) -> None:
        """验证 direct_faq_answer 需要精确匹配或达到阈值分数。

        调用顺序：pytest/unittest 测试入口 -> ContextBuilderTests.test_direct_faq_answer_requires_exact_match_or_threshold()。
        """
        doc = Document(
            page_content="是否支持开发票",
            metadata={"standard_question": "是否支持开发票", "answer": "支持，具体以系统规则为准。"},
        )
        self.assertEqual(direct_faq_answer("是否支持开发票", doc, score=0.1, threshold=0.9), "支持，具体以系统规则为准。")
        self.assertEqual(direct_faq_answer("可以开票吗", doc, score=0.95, threshold=0.9), "支持，具体以系统规则为准。")
        self.assertIsNone(direct_faq_answer("可以开票吗", doc, score=0.3, threshold=0.9))

    def test_faq_fast_path_only_targets_short_complete_questions(self) -> None:
        """验证 FAQ 快速路径仅针对简短完整的问题。

        调用顺序：pytest/unittest 测试入口 -> ContextBuilderTests.test_faq_fast_path_only_targets_short_complete_questions()。
        """
        scenario = get_scenario_registry().resolve("engineering_project_qa")
        self.assertTrue(should_try_faq_fast_path("安全技术交底只有口头说明可以吗？", scenario))
        self.assertFalse(
            should_try_faq_fast_path(
                "请把本项目所有安全资料、质量资料、进度资料、图纸资料和规范资料做一个完整总结，并说明每类资料的风险边界。",
                scenario,
            )
        )

    def test_route_decision_handles_out_of_scope_before_faq_store(self) -> None:
        """验证路由决策在访问 FAQ 存储前处理超范围问题。

        调用顺序：pytest/unittest 测试入口 -> ContextBuilderTests.test_route_decision_handles_out_of_scope_before_faq_store()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        context = RAGQueryContext(
            history=object(),
            query="彩票怎么买",
            source_filter=None,
            scenario=scenario,
            data_scope=resolve_data_scope(),
            session_id="unit-test",
            trace_id="trace-test",
            started=time.perf_counter(),
            active_kb_version="kb_test",
        )

        with patch("qa_core.pipeline.steps.get_faq_store") as get_faq_store:
            decision = decide_route(context)

        self.assertEqual(decision.route, "direct_answer")
        self.assertIsNotNone(decision.answer)
        self.assertIn("超出了", decision.answer)
        self.assertEqual(context.intent_payload["intent"], "OUT_OF_SCOPE")
        self.assertEqual(context.retrieval_info["route"], "direct_answer")
        get_faq_store.assert_not_called()

    def test_route_decision_models_faq_exact_as_route_not_intent_type(self) -> None:
        """验证路由决策将 FAQ 精确匹配建模为路由而非意图类型。

        调用顺序：pytest/unittest 测试入口 -> ContextBuilderTests.test_route_decision_models_faq_exact_as_route_not_intent_type()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        context = RAGQueryContext(
            history=object(),
            query="员工报销需要准备哪些材料？",
            source_filter=None,
            scenario=scenario,
            data_scope=resolve_data_scope(),
            session_id="unit-test",
            trace_id="trace-test",
            started=time.perf_counter(),
            active_kb_version="kb_test",
        )
        faq_result = RetrievalResult(
            hits=[
                RetrievalHit(
                    document=Document(
                        page_content="员工报销需要准备哪些材料？",
                        metadata={"standard_question": "员工报销需要准备哪些材料？", "answer": "请准备发票、审批单和付款凭证。"},
                    ),
                    score=0.42,
                )
            ],
            query="员工报销需要准备哪些材料？",
            source_type="faq",
        )

        with patch("qa_core.pipeline.steps.get_faq_store") as get_faq_store:
            get_faq_store.return_value.search_many.return_value = faq_result
            decision = decide_route(context)

        self.assertEqual(decision.route, "faq_exact")
        self.assertIsNotNone(decision.intent)
        self.assertEqual(decision.intent.intent, "FAQ_QUERY")
        self.assertEqual(decision.answer, "请准备发票、审批单和付款凭证。")
        self.assertEqual(context.retrieval_info["route"], "faq_exact")
        self.assertEqual(context.hit_type, "faq_direct")

    def test_debug_retrieval_returns_direct_route_without_building_plan(self) -> None:
        """验证 debug_retrieval 对直接路由返回结果而不构建检索计划。

        调用顺序：pytest/unittest 测试入口 -> ContextBuilderTests.test_debug_retrieval_returns_direct_route_without_building_plan()。
        """
        with (
            patch("qa_core.pipeline.runtime.resolve_active_kb_version", return_value="kb_test"),
            patch("qa_core.pipeline.rag.prepare_retrieval") as prepare_retrieval,
        ):
            result = debug_retrieval(
                history=object(),
                query="我要转人工",
                source_filter=None,
                session_id="unit-test",
                scenario_id="enterprise_knowledge",
                kb_version="kb_test",
            )

        self.assertEqual(result["route"], "direct_answer")
        self.assertIn("人工支持", result["answer"])
        self.assertEqual(result["answer_confidence"]["level"], "high")
        self.assertEqual(result["answer_confidence"]["generation_verification"]["status"], "not_applicable")
        self.assertIsNone(result["retrieval_plan"])
        self.assertEqual(result["faq_sources"], [])
        self.assertEqual(result["doc_sources"], [])
        prepare_retrieval.assert_not_called()

    def test_debug_retrieval_returns_faq_sources_for_exact_route(self) -> None:
        """验证 debug_retrieval 对精确路由返回 FAQ 来源信息。

        调用顺序：pytest/unittest 测试入口 -> ContextBuilderTests.test_debug_retrieval_returns_faq_sources_for_exact_route()。
        """
        faq_result = RetrievalResult(
            hits=[
                RetrievalHit(
                    document=Document(
                        page_content="员工报销需要准备哪些材料？",
                        metadata={
                            "standard_question": "员工报销需要准备哪些材料？",
                            "answer": "请准备发票、审批单和付款凭证。",
                            "source": "finance",
                        },
                    ),
                    score=0.55,
                )
            ],
            query="员工报销需要准备哪些材料？",
            source_type="faq",
        )

        with (
            patch("qa_core.pipeline.runtime.resolve_active_kb_version", return_value="kb_test"),
            patch("qa_core.pipeline.steps.get_faq_store") as get_faq_store,
            patch("qa_core.pipeline.rag.prepare_retrieval") as prepare_retrieval,
        ):
            get_faq_store.return_value.search_many.return_value = faq_result
            result = debug_retrieval(
                history=object(),
                query="员工报销需要准备哪些材料？",
                source_filter=None,
                session_id="unit-test",
                scenario_id="enterprise_knowledge",
                kb_version="kb_test",
            )

        self.assertEqual(result["route"], "faq_exact")
        self.assertEqual(result["answer"], "请准备发票、审批单和付款凭证。")
        self.assertEqual(result["intent"]["intent"], "FAQ_QUERY")
        self.assertEqual(result["answer_confidence"]["reasons"], ["faq_exact_match"])
        self.assertEqual(result["answer_confidence"]["generation_verification"]["status"], "not_applicable")
        self.assertIsNone(result["retrieval_plan"])
        self.assertEqual(result["faq_sources"][0]["metadata"]["standard_question"], "员工报销需要准备哪些材料？")
        self.assertEqual(result["doc_sources"], [])
        prepare_retrieval.assert_not_called()

    def test_build_context_deduplicates_and_labels_sources(self) -> None:
        """验证 build_context 对文档去重并标注来源。

        调用顺序：pytest/unittest 测试入口 -> ContextBuilderTests.test_build_context_deduplicates_and_labels_sources()。
        """
        docs = [
            Document(page_content="第一段内容", metadata={"file_name": "a.md"}),
            Document(page_content="第一段内容", metadata={"file_name": "a.md"}),
            Document(page_content="第二段内容", metadata={"standard_question": "标准问题"}),
        ]
        context = build_context(docs)
        self.assertIn("[1] 来源：a.md", context)
        self.assertIn("[2] 来源：标准问题", context)
        self.assertEqual(context.count("第一段内容"), 1)

    def test_build_context_labels_table_row_location(self) -> None:
        """验证 build_context 标注表格行的位置信息。

        调用顺序：pytest/unittest 测试入口 -> ContextBuilderTests.test_build_context_labels_table_row_location()。
        """
        context = build_context(
            [
                Document(
                    page_content="施工照片状态：必需",
                    metadata={
                        "file_name": "hidden_acceptance_materials.csv",
                        "content_type": "table_row",
                        "sheet_name": "csv",
                        "row_number": 2,
                    },
                )
            ]
        )
        self.assertIn("hidden_acceptance_materials.csv / 工作表：csv / 第 2 行", context)

    def test_select_context_docs_filters_low_score_faq_and_doc_hits(self) -> None:
        """验证 select_context_docs 过滤低分 FAQ 和文档命中。

        调用顺序：pytest/unittest 测试入口 -> ContextBuilderTests.test_select_context_docs_filters_low_score_faq_and_doc_hits()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        intent = classify_intent("新人入职流程怎么走", [], scenario)
        plan = build_retrieval_plan("新人入职流程怎么走", intent)
        faq_hits = [
            RetrievalHit(
                document=Document(
                    page_content="低分 FAQ",
                    metadata={"standard_question": "低分 FAQ", "answer": "不应进入上下文"},
                ),
                score=plan.min_context_score - 0.01,
            )
        ]
        doc_hits = [
            RetrievalHit(
                document=Document(page_content="低分文档", metadata={"file_name": "low.md"}),
                score=plan.min_context_score - 0.01,
            )
        ]
        self.assertEqual(select_context_docs(faq_hits, doc_hits, plan), [])

    def test_select_context_docs_deduplicates_parent_and_applies_budget(self) -> None:
        """验证 select_context_docs 对父文档去重并应用 Token 预算。

        调用顺序：pytest/unittest 测试入口 -> ContextBuilderTests.test_select_context_docs_deduplicates_parent_and_applies_budget()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        intent = classify_intent("新人入职流程怎么走", [], scenario)
        plan = build_retrieval_plan("新人入职流程怎么走", intent)
        doc_hits = [
            RetrievalHit(
                document=Document(
                    page_content="子块一",
                    metadata={
                        "parent_id": "parent_1",
                        "parent_content": "甲" * (plan.max_context_doc_chars + 80),
                        "file_name": "a.md",
                    },
                ),
                score=0.9,
            ),
            RetrievalHit(
                document=Document(
                    page_content="子块二",
                    metadata={
                        "parent_id": "parent_1",
                        "parent_content": "重复父块",
                        "file_name": "a.md",
                    },
                ),
                score=0.8,
            ),
        ]
        docs = select_context_docs([], doc_hits, plan)
        self.assertEqual(len(docs), 1)
        self.assertLessEqual(len(docs[0].page_content), plan.max_context_doc_chars)
        self.assertTrue(docs[0].metadata["context_truncated"])

    def test_table_query_prefers_table_rows_in_context(self) -> None:
        """表格类问题应优先把行级证据放入 prompt。

        调用顺序：pytest/unittest 测试入口 -> ContextBuilderTests.test_table_query_prefers_table_rows_in_context()。
        """
        scenario = get_scenario_registry().resolve("engineering_project_qa")
        intent = classify_intent("隐蔽验收资料清单里施工照片是什么状态？", [], scenario)
        plan = build_retrieval_plan("隐蔽验收资料清单里施工照片是什么状态？", intent)
        doc_hits = [
            RetrievalHit(
                document=Document(page_content="普通正文片段", metadata={"file_name": "hidden_acceptance.md"}),
                score=0.99,
            ),
            RetrievalHit(
                document=Document(
                    page_content="表格文件：hidden_acceptance_materials.csv\n行号：2\n- 资料名称：施工照片\n- 状态：必需",
                    metadata={"file_name": "hidden_acceptance_materials.csv", "content_type": "table_row", "row_number": 2},
                ),
                score=0.72,
            ),
        ]
        docs = select_context_docs([], doc_hits, plan)
        self.assertTrue(plan.prefer_table)
        self.assertTrue(plan.faq_direct_exact_only)
        self.assertEqual(docs[0].metadata["content_type"], "table_row")

    def test_enforce_answer_citations_appends_table_source_when_missing(self) -> None:
        """验证 enforce_answer_citations 在缺失引用时补充表格来源。

        调用顺序：pytest/unittest 测试入口 -> ContextBuilderTests.test_enforce_answer_citations_appends_table_source_when_missing()。
        """
        docs = [
            Document(
                page_content="施工照片状态：必需",
                metadata={
                    "file_name": "hidden_acceptance_materials.csv",
                    "content_type": "table_row",
                    "sheet_name": "csv",
                    "row_number": 2,
                },
            )
        ]
        answer = enforce_answer_citations("施工照片状态为必需。", docs)
        self.assertTrue(has_source_citation(answer))
        self.assertIn("hidden_acceptance_materials.csv / 工作表：csv / 第 2 行", answer)

    def test_enforce_answer_citations_keeps_existing_citation(self) -> None:
        """验证 enforce_answer_citations 保留已有引用不变。

        调用顺序：pytest/unittest 测试入口 -> ContextBuilderTests.test_enforce_answer_citations_keeps_existing_citation()。
        """
        answer = enforce_answer_citations("施工照片状态为必需。[1]", [Document(page_content="x")])
        self.assertEqual(answer, "施工照片状态为必需。[1]")

    def test_enforce_answer_citations_appends_missing_table_cells_even_with_citation(self) -> None:
        """表格答案即使已经带引用，也要补齐同一行里被模型漏掉的关键单元格。

        调用顺序：pytest/unittest 测试入口 -> ContextBuilderTests.test_enforce_answer_citations_appends_missing_table_cells_even_with_citation()。
        """
        docs = [
            Document(
                page_content=(
                    "表格文件：claim_material_review.csv\n"
                    "单元格：\n"
                    "- 材料名称：银行卡信息\n"
                    "- 审核状态：必需\n"
                    "- 核验要点：账户名需与被保险人或授权收款人一致\n"
                    "- 处理动作：不一致时进入人工复核"
                ),
                metadata={
                    "file_name": "claim_material_review.csv",
                    "content_type": "table_row",
                    "sheet_name": "csv",
                    "row_number": 2,
                },
            )
        ]
        answer = enforce_answer_citations("银行卡信息不一致时，需进入人工复核 [1]。", docs)
        self.assertIn("表格行要点", answer)
        self.assertIn("账户名需与被保险人或授权收款人一致", answer)
        self.assertIn("[1]", answer)


class RetrievalRankingTests(unittest.TestCase):
    """验证多路查询召回后的去重和排序规则。

    调用顺序：pytest/unittest 测试入口 -> RetrievalRankingTests。
    """

    def test_normalize_queries_keeps_order_and_removes_duplicates(self) -> None:
        """验证 normalize_queries 保留顺序并移除重复项。

        调用顺序：pytest/unittest 测试入口 -> RetrievalRankingTests.test_normalize_queries_keeps_order_and_removes_duplicates()。
        """
        queries = normalize_queries([" Webhook失败 ", "", "回调失败", "Webhook失败", "  "])
        self.assertEqual(queries, ["Webhook失败", "回调失败"])

    def test_merge_hits_keeps_highest_score_for_same_document(self) -> None:
        """验证 merge_hits_by_document 对同一文档保留最高分。

        调用顺序：pytest/unittest 测试入口 -> RetrievalRankingTests.test_merge_hits_keeps_highest_score_for_same_document()。
        """
        first = RetrievalHit(
            document=Document(page_content="同一个 chunk", metadata={"chunk_id": "chunk_1"}),
            score=0.4,
        )
        second = RetrievalHit(
            document=Document(page_content="同一个 chunk 新命中", metadata={"chunk_id": "chunk_1"}),
            score=0.9,
        )
        merged: dict[str, RetrievalHit] = {}
        merge_hits_by_document(merged, [first])
        merge_hits_by_document(merged, [second])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged["chunk_1"].score, 0.9)

    def test_sort_hits_by_score_descending(self) -> None:
        """验证 sort_hits_by_score 按分数降序排列。

        调用顺序：pytest/unittest 测试入口 -> RetrievalRankingTests.test_sort_hits_by_score_descending()。
        """
        hits = [
            RetrievalHit(document=Document(page_content="低分"), score=0.2),
            RetrievalHit(document=Document(page_content="高分"), score=0.8),
        ]
        sorted_hits = sort_hits_by_score(hits)
        self.assertEqual([hit.document.page_content for hit in sorted_hits], ["高分", "低分"])

    def test_source_payloads_include_table_citation(self) -> None:
        """验证来源负载中包含表格引用信息。

        调用顺序：pytest/unittest 测试入口 -> RetrievalRankingTests.test_source_payloads_include_table_citation()。
        """
        result = RetrievalResult(
            hits=[
                RetrievalHit(
                    document=Document(
                        page_content="表格文件：acceptance_checklist.csv\n- 验收项：测试报告",
                        metadata={
                            "file_name": "acceptance_checklist.csv",
                            "content_type": "table_row",
                            "sheet_name": "csv",
                            "row_number": 3,
                            "table_headers": "验收项 | 状态",
                        },
                    ),
                    score=0.8,
                )
            ],
            source_type="doc",
        )
        payload = result.source_payloads()[0]
        self.assertEqual(payload["citation"], "acceptance_checklist.csv / 工作表：csv / 第 3 行")
        self.assertEqual(payload["table"]["row_number"], 3)


class MilvusHybridStoreTests(unittest.TestCase):
    """验证 Milvus 检索边界条件不会把底层兼容异常暴露给用户。

    调用顺序：pytest/unittest 测试入口 -> MilvusHybridStoreTests。
    """

    def test_hybrid_index_and_search_params_match_dense_sparse_order(self) -> None:
        """验证显式索引参数顺序与 vector_field=["dense", "sparse"] 对齐。

        调用顺序：pytest/unittest 测试入口 -> MilvusHybridStoreTests.test_hybrid_index_and_search_params_match_dense_sparse_order()。
        """
        self.assertEqual(
            hybrid_index_params(),
            [
                {
                    "index_type": "HNSW",
                    "metric_type": "L2",
                    "params": {"M": 16, "efConstruction": 200},
                },
                {"index_type": "AUTOINDEX", "metric_type": "BM25", "params": {}},
            ],
        )
        self.assertEqual(
            hybrid_search_params(),
            [
                {"metric_type": "L2", "params": {"ef": 64}},
                {"metric_type": "BM25", "params": {}},
            ],
        )

    def test_empty_query_returns_empty_result_without_touching_store(self) -> None:
        """验证空查询不访问存储直接返回空结果。

        调用顺序：pytest/unittest 测试入口 -> MilvusHybridStoreTests.test_empty_query_returns_empty_result_without_touching_store()。
        """
        store = MilvusHybridStore("unit_test_collection")
        result = store.search(
            "  ",
            k=5,
            source_filter=None,
            source_type="faq",
            rerank=False,
        )
        self.assertEqual(result.hits, [])
        self.assertEqual(result.query, "")

    def test_nq_zero_hybrid_error_requires_collection_rebuild(self) -> None:
        """验证 nq=0 的混合检索错误提示需要重建集合。

        调用顺序：pytest/unittest 测试入口 -> MilvusHybridStoreTests.test_nq_zero_hybrid_error_requires_collection_rebuild()。
        """
        class FakeStore:
            """模拟引发 MilvusException 的存储测试替身。

            调用顺序：pytest/unittest 测试入口 -> FakeStore。
            """

            def similarity_search_with_score(self, *args, **kwargs):
                """测试辅助方法：始终抛出 MilvusException，用于验证混合检索异常处理。

                调用顺序：pytest/unittest 测试入口 -> FakeStore.similarity_search_with_score()。
                """
                raise MilvusException(
                    code=65535,
                    message=(
                        "nq [0] is invalid, nq (number of search vector per search request) "
                        "should be in range [1, 16384], but got 0"
                    ),
                )

        store = MilvusHybridStore("unit_test_collection")
        store._store = FakeStore()

        with self.assertRaisesRegex(RuntimeError, "BM25 Function|reset-collections"):
            store.search(
                "申报要素缺失会有什么风险？",
                k=5,
                source_filter=None,
                source_type="faq",
                rerank=False,
            )

    def test_invalid_hybrid_schema_requires_collection_rebuild(self) -> None:
        """验证无效的混合模式提示需要重建集合。

        调用顺序：pytest/unittest 测试入口 -> MilvusHybridStoreTests.test_invalid_hybrid_schema_requires_collection_rebuild()。
        """
        class FakeField:
            """模拟 Milvus 字段对象的测试替身。

            调用顺序：pytest/unittest 测试入口 -> FakeField。
            """

            def __init__(self, name, dtype, params=None, is_function_output=False):
                """初始化模拟字段对象。

                调用顺序：pytest/unittest 测试入口 -> FakeField.__init__()。
                """
                self.name = name
                self.dtype = dtype
                self.params = params or {}
                self.is_function_output = is_function_output

        class FakeSchema:
            """模拟 Milvus Schema 的测试替身。

            调用顺序：pytest/unittest 测试入口 -> FakeSchema。
            """
            fields = [
                FakeField("pk", DataType.VARCHAR),
                FakeField("text", DataType.VARCHAR, {"enable_analyzer": False}),
                FakeField("dense", DataType.FLOAT_VECTOR),
                FakeField("sparse", DataType.SPARSE_FLOAT_VECTOR),
            ]
            functions = []

        class FakeCollection:
            """模拟 Milvus Collection 的测试替身。

            调用顺序：pytest/unittest 测试入口 -> FakeCollection。
            """
            schema = FakeSchema()
            indexes = []

        class FakeStore:
            """模拟 Milvus 存储的测试替身。

            调用顺序：pytest/unittest 测试入口 -> FakeStore。
            """
            col = FakeCollection()

        store = MilvusHybridStore("unit_test_collection")
        store._store = FakeStore()

        with self.assertRaisesRegex(RuntimeError, "缺少 text -> sparse 的 BM25 Function|reset-collections"):
            store.validate_hybrid_schema()

        class ValidSchema:
            """模拟有效混合模式的测试替身。

            调用顺序：pytest/unittest 测试入口 -> ValidSchema。
            """
            fields = [
                FakeField("pk", DataType.VARCHAR),
                FakeField("text", DataType.VARCHAR, {"enable_analyzer": "true"}),
                FakeField("dense", DataType.FLOAT_VECTOR),
                FakeField("sparse", DataType.SPARSE_FLOAT_VECTOR, is_function_output=True),
            ]
            functions = [
                Function(
                    name="bm25_test",
                    function_type=FunctionType.BM25,
                    input_field_names="text",
                    output_field_names="sparse",
                )
            ]

        class FakeIndex:
            """模拟 Milvus 索引对象的测试替身。"""

            def __init__(self, field_name, params):
                self.field_name = field_name
                self.params = params

        FakeCollection.schema = ValidSchema()
        FakeCollection.indexes = [
            FakeIndex(
                "dense",
                {
                    "index_type": "HNSW",
                    "metric_type": "L2",
                    "params": {"M": 16, "efConstruction": 200},
                },
            ),
            FakeIndex(
                "sparse",
                {"index_type": "AUTOINDEX", "metric_type": "BM25", "params": {}},
            ),
        ]
        store.validate_hybrid_schema()

    def test_old_dense_autoindex_requires_collection_rebuild(self) -> None:
        """验证已有 AUTOINDEX dense collection 不会被静默当作 HNSW 使用。"""

        class FakeField:
            def __init__(self, name, dtype, params=None, is_function_output=False):
                self.name = name
                self.dtype = dtype
                self.params = params or {}
                self.is_function_output = is_function_output

        class FakeSchema:
            fields = [
                FakeField("pk", DataType.VARCHAR),
                FakeField("text", DataType.VARCHAR, {"enable_analyzer": "true"}),
                FakeField("dense", DataType.FLOAT_VECTOR),
                FakeField("sparse", DataType.SPARSE_FLOAT_VECTOR, is_function_output=True),
            ]
            functions = [
                Function(
                    name="bm25_test",
                    function_type=FunctionType.BM25,
                    input_field_names="text",
                    output_field_names="sparse",
                )
            ]

        class FakeIndex:
            def __init__(self, field_name, params):
                self.field_name = field_name
                self.params = params

        class FakeCollection:
            schema = FakeSchema()
            indexes = [
                FakeIndex("dense", {"index_type": "AUTOINDEX", "metric_type": "L2", "params": {}}),
                FakeIndex("sparse", {"index_type": "AUTOINDEX", "metric_type": "BM25", "params": {}}),
            ]

        class FakeStore:
            col = FakeCollection()

        store = MilvusHybridStore("unit_test_collection")
        store._store = FakeStore()

        with self.assertRaisesRegex(RuntimeError, "dense 索引类型应为 HNSW|reset-collections"):
            store.validate_hybrid_schema()


    def test_expire_documents_for_version_updates_valid_to_seq_without_reembedding(self) -> None:
        """验证 expire_documents_for_version 更新 valid_to_seq 而不重新嵌入。

        调用顺序：pytest/unittest 测试入口 -> MilvusHybridStoreTests.test_expire_documents_for_version_updates_valid_to_seq_without_reembedding()。
        """
        class FakeCollection:
            """模拟 Milvus Collection 的测试替身。

            调用顺序：pytest/unittest 测试入口 -> FakeCollection。
            """

            def __init__(self):
                """初始化模拟集合，清空 upsert 记录列表。

                调用顺序：pytest/unittest 测试入口 -> FakeCollection.__init__()。
                """
                self.upserted = []

            def query(self, *, expr, output_fields):
                """测试辅助方法：返回固定文档查询结果。

                调用顺序：pytest/unittest 测试入口 -> FakeCollection.query()。
                """
                return [
                    {
                        "pk": "old_chunk",
                        "text": "旧版 VPN 文档内容。",
                        "dense": [0.1, 0.2],
                        "$meta": {
                            "scenario_id": "enterprise_knowledge",
                            "kb_version": "kb_v1",
                            "valid_from_seq": 1,
                            "valid_to_seq": 0,
                            "source": "it",
                            "chunk_id": "old_chunk",
                        },
                    }
                ]

            def upsert(self, rows):
                """测试辅助方法：记录 upsert 的行数据。

                调用顺序：pytest/unittest 测试入口 -> FakeCollection.upsert()。
                """
                self.upserted.extend(rows)

        class FakeStore:
            """模拟 Milvus 存储的测试替身。

            调用顺序：pytest/unittest 测试入口 -> FakeStore。
            """

            def __init__(self):
                """初始化模拟存储，创建包含 FakeCollection 的实例。

                调用顺序：pytest/unittest 测试入口 -> FakeStore.__init__()。
                """
                self.col = FakeCollection()

        store = MilvusHybridStore("unit_test_collection")
        fake_store = FakeStore()
        store._store = fake_store

        updated = store.expire_documents_for_version(["old_chunk"], valid_to_seq=2)

        self.assertEqual(updated, 1)
        self.assertEqual(fake_store.col.upserted[0]["pk"], "old_chunk")
        self.assertEqual(fake_store.col.upserted[0]["valid_to_seq"], 2)
        self.assertEqual(fake_store.col.upserted[0]["dense"], [0.1, 0.2])

class RetrievalPlanTests(unittest.TestCase):
    """验证意图和问题类别会稳定影响检索计划。

    调用顺序：pytest/unittest 测试入口 -> RetrievalPlanTests。
    """

    def test_knowledge_intent_enables_query_variants_and_expands_docs(self) -> None:
        """验证 KNOWLEDGE_QUERY 意图启用查询变体并扩展文档检索。

        调用顺序：pytest/unittest 测试入口 -> RetrievalPlanTests.test_knowledge_intent_enables_query_variants_and_expands_docs()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        intent = classify_intent("新人入职流程怎么走", [], scenario)
        self.assertEqual(intent.intent, "KNOWLEDGE_QUERY")
        plan = build_retrieval_plan("新人入职流程怎么走", intent)
        self.assertTrue(plan.use_query_variants)
        self.assertIn("knowledge_doc_enriched", plan.reason)
        self.assertEqual(plan.intent_rule_score, intent.rule_score)
        self.assertEqual(plan.intent_decision_score, intent.confidence)
        self.assertIn("rule_score", intent.as_dict())

    def test_low_rule_score_makes_retrieval_plan_more_conservative(self) -> None:
        """验证低决策分数使检索计划更加保守。

        调用顺序：pytest/unittest 测试入口 -> RetrievalPlanTests.test_low_rule_score_makes_retrieval_plan_more_conservative()。
        """
        settings = get_settings()
        rules = get_rule_config().retrieval_strategy
        intent = IntentResult(intent="KNOWLEDGE_QUERY", rule_score=0.6, reason="default_knowledge")

        plan = build_retrieval_plan("帮我分析一下这个问题", intent)

        self.assertEqual(plan.intent_rule_score, 0.6)
        self.assertEqual(plan.intent_decision_score, 0.6)
        self.assertIn("low_rule_score_guard", plan.reason)
        self.assertTrue(plan.faq_direct_exact_only)
        self.assertGreaterEqual(plan.faq_direct_threshold, rules.low_rule_score_direct_threshold)
        self.assertGreaterEqual(plan.doc_top_k, settings.doc_complex_query_top_k)
        self.assertGreaterEqual(plan.final_context_top_n, rules.guard_context_top_n_min)

    def test_strong_faq_rule_keeps_plan_faq_first_without_low_score_guard(self) -> None:
        """验证强 FAQ 规则保持 FAQ 优先且不触发低分守卫。

        调用顺序：pytest/unittest 测试入口 -> RetrievalPlanTests.test_strong_faq_rule_keeps_plan_faq_first_without_low_score_guard()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        rules = get_rule_config().retrieval_strategy
        intent = classify_intent("开票失败怎么办", [], scenario)

        plan = build_retrieval_plan("开票失败怎么办", intent)

        self.assertEqual(intent.intent, "FAQ_QUERY")
        self.assertEqual(intent.reason, "strong_faq_rule")
        self.assertEqual(plan.intent_rule_score, 0.82)
        self.assertEqual(plan.intent_decision_score, intent.confidence)
        self.assertIn("faq_first", plan.reason)
        self.assertNotIn("low_rule_score_guard", plan.reason)
        self.assertFalse(plan.faq_direct_exact_only)
        self.assertGreaterEqual(plan.doc_top_k, rules.strong_faq_doc_top_k_min)

    def test_default_knowledge_path_uses_conservative_retrieval_guard(self) -> None:
        """验证默认知识路径使用保守检索守卫。

        调用顺序：pytest/unittest 测试入口 -> RetrievalPlanTests.test_default_knowledge_path_uses_conservative_retrieval_guard()。
        """
        scenario = get_scenario_registry().resolve("enterprise_knowledge")
        intent = classify_intent("帮我分析一下这个问题", [], scenario)

        plan = build_retrieval_plan("帮我分析一下这个问题", intent)

        self.assertEqual(intent.intent, "KNOWLEDGE_QUERY")
        self.assertEqual(intent.reason, "default_knowledge")
        self.assertEqual(plan.intent_rule_score, 0.6)
        self.assertEqual(plan.intent_decision_score, 0.6)
        self.assertIn("low_rule_score_guard", plan.reason)
        self.assertTrue(plan.faq_direct_exact_only)

    def test_model_conflict_decision_score_triggers_conservative_guard(self) -> None:
        """验证模型/规则冲突后的最终分数会参与检索计划。"""
        rules = get_rule_config().retrieval_strategy
        intent = IntentResult(
            intent="KNOWLEDGE_QUERY",
            rule_score=0.84,
            final_score=0.68,
            reason="unit_test_conflict",
            decision_policy="rule_model_conflict_guarded",
        )

        plan = build_retrieval_plan("入职流程包含哪些步骤？", intent)

        self.assertEqual(plan.intent_rule_score, 0.84)
        self.assertEqual(plan.intent_decision_score, 0.68)
        self.assertIn("low_rule_score_guard", plan.reason)
        self.assertTrue(plan.faq_direct_exact_only)
        self.assertGreaterEqual(plan.faq_direct_threshold, rules.low_rule_score_direct_threshold)

    def test_short_structured_questions_do_not_expand_with_llm_variants(self) -> None:
        """验证简短结构化问题不进行 LLM 查询变体扩展。

        调用顺序：pytest/unittest 测试入口 -> RetrievalPlanTests.test_short_structured_questions_do_not_expand_with_llm_variants()。
        """
        self.assertEqual(generate_query_variants("新人入职流程怎么走", enabled=True), ["新人入职流程怎么走"])
        self.assertEqual(generate_query_variants("那隐蔽工程验收资料呢？", enabled=True), ["那隐蔽工程验收资料呢？"])

    def test_follow_up_rewrite_allows_rule_variants_even_when_short_structured(self) -> None:
        """验证追问重写即使对简短结构化问题也允许规则变体。

        调用顺序：pytest/unittest 测试入口 -> RetrievalPlanTests.test_follow_up_rewrite_allows_rule_variants_even_when_short_structured()。
        """
        variants = generate_query_variants(
            "报销流程是什么；追问：那审批呢",
            enabled=True,
            allow_short_structured=True,
        )

        self.assertEqual(variants[0], "报销流程是什么；追问：那审批呢")
        self.assertIn("报销SOP是什么；追问：那审批呢", variants)
        self.assertIn("报销处理步骤是什么；追问：那审批呢", variants)

    def test_source_guided_short_question_prefers_faq_intent(self) -> None:
        """验证来源引导的简短问题优先使用 FAQ 意图。

        调用顺序：pytest/unittest 测试入口 -> RetrievalPlanTests.test_source_guided_short_question_prefers_faq_intent()。
        """
        scenario = get_scenario_registry().resolve("engineering_project_qa")
        intent = classify_intent("那隐蔽工程验收资料呢？", [], scenario)
        self.assertEqual(intent.intent, "FAQ_QUERY")
        self.assertEqual(intent.suggested_source, "quality")

    def test_table_query_signal_does_not_steal_pricing_prompt_category(self) -> None:
        """验证表格查询信号不会覆盖定价 Prompt 类别。

        调用顺序：pytest/unittest 测试入口 -> RetrievalPlanTests.test_table_query_signal_does_not_steal_pricing_prompt_category()。
        """
        self.assertTrue(is_table_query("付款节点表里质保金金额是多少？"))
        self.assertEqual(infer_question_category("付款节点表里质保金金额是多少？"), "pricing")
        intent = IntentResult(intent="FAQ_QUERY", rule_score=0.84, reason="unit_test", suggested_source="contract")
        plan = build_retrieval_plan("付款节点表里质保金金额是多少？", intent)
        self.assertTrue(plan.prefer_table)
        self.assertTrue(plan.faq_direct_exact_only)
        self.assertEqual(plan.question_category, "pricing")


class PromptProfileTests(unittest.TestCase):
    """验证最终回答模板按问题风险和意图确定性选择。

    调用顺序：pytest/unittest 测试入口 -> PromptProfileTests。
    """

    def setUp(self) -> None:
        """测试前置：获取企业知识场景用于 Prompt 测试。

        调用顺序：pytest/unittest 测试入口 -> PromptProfileTests.setUp()。
        """
        self.scenario = get_scenario_registry().resolve("enterprise_knowledge")

    def test_pricing_question_uses_pricing_guard_before_intent_profile(self) -> None:
        """验证定价类问题优先使用定价守卫模板而非意图模板。

        调用顺序：pytest/unittest 测试入口 -> PromptProfileTests.test_pricing_question_uses_pricing_guard_before_intent_profile()。
        """
        profile = build_answer_prompt_profile("FAQ_QUERY", self.scenario, query="发票和退款规则是什么")
        self.assertEqual(profile.name, "pricing_guard")
        self.assertIn("已确认", profile.system_template)

    def test_business_fund_risk_questions_use_pricing_guard(self) -> None:
        """资金、结算和付款承诺类问题即使是 FAQ，也要使用强口径模板。

        调用顺序：pytest/unittest 测试入口 -> PromptProfileTests.test_business_fund_risk_questions_use_pricing_guard()。
        """
        queries = [
            "没有预算审批可以先采购再报销吗？",
            "信用证软条款有什么风险？",
            "投标保证金异常需要关注什么？",
            "收款账户和被保险人不一致可以打款吗？",
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertEqual(infer_question_category(query), "pricing")
                profile = build_answer_prompt_profile("FAQ_QUERY", self.scenario, query=query)
                self.assertEqual(profile.name, "pricing_guard")

    def test_business_compliance_questions_use_compliance_guard(self) -> None:
        """监管、安全责任和资料真实性问题要使用合规模板，不按普通知识问答处理。

        调用顺序：pytest/unittest 测试入口 -> PromptProfileTests.test_business_compliance_questions_use_compliance_guard()。
        """
        queries = [
            "受限空间作业前需要哪些安全确认？",
            "HS 编码归类存在争议时能先按客户说法申报吗？",
            "最终用途不清楚可以继续发货吗？",
            "既往症未如实告知会有什么影响？",
            "检验批资料和现场实物不一致怎么办？",
            "安全技术交底只有口头说明可以吗？",
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertEqual(infer_question_category(query), "compliance")
                profile = build_answer_prompt_profile("KNOWLEDGE_QUERY", self.scenario, query=query)
                self.assertEqual(profile.name, "compliance_guard")

    def test_faq_intent_uses_faq_answer_profile(self) -> None:
        """验证 FAQ 意图使用 FAQ 回答模板。

        调用顺序：pytest/unittest 测试入口 -> PromptProfileTests.test_faq_intent_uses_faq_answer_profile()。
        """
        profile = build_answer_prompt_profile("FAQ_QUERY", self.scenario, query="怎么修改账号密码")
        self.assertEqual(profile.name, "faq_answer")

    def test_knowledge_intent_uses_knowledge_profile(self) -> None:
        """验证 KNOWLEDGE_QUERY 意图使用知识回答模板。

        调用顺序：pytest/unittest 测试入口 -> PromptProfileTests.test_knowledge_intent_uses_knowledge_profile()。
        """
        profile = build_answer_prompt_profile("KNOWLEDGE_QUERY", self.scenario, query="入职流程怎么走")
        self.assertEqual(profile.name, "knowledge_answer")

    def test_unknown_intent_uses_default_profile(self) -> None:
        """验证未知意图使用默认回答模板。

        调用顺序：pytest/unittest 测试入口 -> PromptProfileTests.test_unknown_intent_uses_default_profile()。
        """
        profile = build_answer_prompt_profile("UNKNOWN", self.scenario, query="请介绍一下")
        self.assertEqual(profile.name, "default_answer")
