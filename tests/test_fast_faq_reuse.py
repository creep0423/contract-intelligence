"""FAQ 快路径候选复用的回归测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from qa_core.pipeline.retrieval_steps import search_faq
from qa_core.retrieval.results import RetrievalHit, RetrievalResult


def _hit(faq_id: str, score: float) -> RetrievalHit:
    """构造带稳定 FAQ 标识的检索命中。"""
    return RetrievalHit(
        document=Document(page_content=faq_id, metadata={"faq_id": faq_id}),
        score=score,
    )


class _Context:
    """只实现 search_faq() 需要的请求上下文字段。"""

    def __init__(self, *, source_filter: str | None = "hr") -> None:
        self.query = "新人入职需要完成哪些流程？"
        self.fast_faq_source_filter = source_filter
        self.fast_faq_top_k = 20
        self.fast_faq_result = RetrievalResult(
            hits=[_hit("faq-original", 0.70), _hit("faq-duplicate", 0.60)],
            query=self.query,
            source_type="faq",
            elapsed_ms=11.0,
        )
        self.retrieval_info: dict[str, object] = {}
        self.scenario = SimpleNamespace(faq_collection="faq_collection")
        self.active_kb_version = "kb_test"
        self.data_scope = object()

    def run_stage(self, _stage_name: str, action):
        """同步执行阶段回调，测试只验证业务参数。"""
        return action()


def _prepared(*, source_filter: str | None = "hr", rerank: bool = False):
    """构造含原问题与两个新增查询变体的检索准备结果。"""
    query = "新人入职需要完成哪些流程？"
    return SimpleNamespace(
        rewritten_query=query,
        query_variants=[query, "新人入职需要完成哪些 SOP？", "新人入职需要完成哪些办理步骤？"],
        effective_source_filter=source_filter,
        plan=SimpleNamespace(run_faq=True, faq_top_k=20, rerank=rerank),
    )


class FastFaqReuseTests(unittest.TestCase):
    """验证 fast FAQ 未精确命中时的增量检索边界。"""

    def test_reuses_original_hits_and_searches_only_new_variants(self) -> None:
        """原问题候选应直接复用，Milvus 只接收两个新增变体。"""
        context = _Context()
        variants_result = RetrievalResult(
            hits=[_hit("faq-duplicate", 0.85), _hit("faq-variant", 0.90)],
            query="variants",
            source_type="faq",
            elapsed_ms=7.0,
        )

        with patch(
            "qa_core.pipeline.retrieval_steps._search_with_cache",
            return_value=variants_result,
        ) as search_with_cache:
            result = search_faq(context, _prepared())

        search_with_cache.assert_called_once()
        self.assertEqual(
            search_with_cache.call_args.kwargs["query_variants"],
            ["新人入职需要完成哪些 SOP？", "新人入职需要完成哪些办理步骤？"],
        )
        self.assertFalse(search_with_cache.call_args.kwargs["rerank"])
        self.assertEqual([hit.document.metadata["faq_id"] for hit in result.hits], ["faq-variant", "faq-duplicate", "faq-original"])
        self.assertEqual(result.elapsed_ms, 18.0)
        self.assertTrue(context.retrieval_info["faq_reused_from_fast_path"])
        self.assertEqual(context.retrieval_info["faq_fast_reused_hit_count"], 2)

    def test_combined_candidates_are_reranked_once(self) -> None:
        """原问题和变体候选合并后，仍须作为同一批候选统一重排。"""
        context = _Context()
        variants_result = RetrievalResult(
            hits=[_hit("faq-variant", 0.90)],
            query="variants",
            source_type="faq",
        )
        reranked = [_hit("faq-original", 0.99), _hit("faq-variant", 0.98)]

        with (
            patch("qa_core.pipeline.retrieval_steps._search_with_cache", return_value=variants_result),
            patch("qa_core.pipeline.retrieval_steps._rerank_merged_faq_hits", return_value=reranked) as rerank_hits,
        ):
            result = search_faq(context, _prepared(rerank=True))

        self.assertEqual(rerank_hits.call_args.args[0], context.query)
        self.assertEqual(rerank_hits.call_args.args[1], _prepared(rerank=True).query_variants)
        self.assertEqual({hit.document.metadata["faq_id"] for hit in rerank_hits.call_args.args[2]}, {"faq-original", "faq-duplicate", "faq-variant"})
        self.assertEqual([hit.document.metadata["faq_id"] for hit in result.hits], ["faq-original", "faq-variant"])

    def test_source_filter_change_falls_back_to_full_faq_search(self) -> None:
        """source_filter 不同会改变数据隔离边界，不能复用旧候选。"""
        context = _Context(source_filter="hr")
        expected = RetrievalResult(hits=[_hit("faq-full", 0.80)], query="full", source_type="faq")

        with patch("qa_core.pipeline.retrieval_steps._search_with_cache", return_value=expected) as search_with_cache:
            result = search_faq(context, _prepared(source_filter="it"))

        self.assertIs(result, expected)
        self.assertEqual(search_with_cache.call_args.kwargs["query_variants"], _prepared(source_filter="it").query_variants)
        self.assertFalse(context.retrieval_info["faq_reused_from_fast_path"])
        self.assertEqual(context.retrieval_info["faq_reuse_reason"], "full_faq_search_required")


if __name__ == "__main__":
    unittest.main()
