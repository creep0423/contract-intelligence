"""V1 企业级缓存策略测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from qa_core.cache import keys as cache_keys
from qa_core.cache.keys import retrieval_cache_key
from qa_core.cache.embedding import CachedEmbeddings
from qa_core.cache.manager import CacheManager
from qa_core.cache.serialization import retrieval_result_from_payload, retrieval_result_to_payload
from qa_core.governance.data_scope import DataScope, resolve_data_scope
from qa_core.retrieval.results import RetrievalHit, RetrievalResult


class FakeRedisJsonCache:
    """测试用 Redis 替身。"""

    def __init__(self) -> None:
        self.data: dict[str, dict] = {}

    def get_json(self, key: str):
        """读取并反序列化 JSON 缓存值。

        调用顺序：测试或业务入口 -> FakeRedisJsonCache.get_json()。
        """
        return self.data.get(key)

    def set_json(self, key: str, payload: dict, ttl_seconds: int) -> None:
        """序列化对象并写入带过期时间的 JSON 缓存。

        调用顺序：测试或业务入口 -> FakeRedisJsonCache.set_json()。
        """
        self.data[key] = payload

    def ping(self) -> bool:
        """检查缓存后端连接是否可用。

        调用顺序：测试或业务入口 -> FakeRedisJsonCache.ping()。
        """
        return True


class FakeNamespaceStore:
    """测试用 namespace store。"""

    def __init__(self) -> None:
        self.epoch = 1

    def get_epoch(self, *, scenario_id: str, tenant_id: str, dataset_id: str) -> int:
        """读取场景缓存命名空间的当前 epoch。

        调用顺序：测试或业务入口 -> FakeNamespaceStore.get_epoch()。
        """
        return self.epoch

    def bump_scenario_epoch(self, scenario_id: str) -> int:
        """递增场景 epoch，使旧版本缓存整体失效。

        调用顺序：测试或业务入口 -> FakeNamespaceStore.bump_scenario_epoch()。
        """
        self.epoch += 1
        return 1

    def list_namespaces(self, *, scenario_id: str | None = None, limit: int = 100):
        """列出当前缓存命名空间及其 epoch 状态。

        调用顺序：测试或业务入口 -> FakeNamespaceStore.list_namespaces()。
        """
        return [{"scenario_id": scenario_id or "enterprise_knowledge", "cache_epoch": self.epoch}]


class EnterpriseCacheTests(unittest.TestCase):
    """验证缓存不会跨版本、跨权限域复用。"""

    def test_retrieval_cache_key_contains_version_and_scope(self) -> None:
        """不同版本和角色生成不同 key。"""
        public_scope = resolve_data_scope(tenant_id="t1", dataset_id="d1", visibility="public", user_role="public")
        admin_scope = resolve_data_scope(tenant_id="t1", dataset_id="d1", visibility="private", user_role="admin")

        base = {
            "kind": "retrieval",
            "scenario_id": "enterprise_knowledge",
            "collection_name": "enterprise_faq",
            "source_type": "faq",
            "data_scope": public_scope,
            "cache_epoch": 3,
            "source_filter": "hr",
            "query_variants": ["新人入职流程"],
            "k": 12,
            "rerank": False,
        }

        key_v1 = retrieval_cache_key(kb_version="kb_v1", **base)
        key_v2 = retrieval_cache_key(kb_version="kb_v2", **base)
        key_admin = retrieval_cache_key(kb_version="kb_v1", **{**base, "data_scope": admin_scope})

        self.assertNotEqual(key_v1, key_v2)
        self.assertNotEqual(key_v1, key_admin)

        with patch.object(cache_keys, "RETRIEVAL_CACHE_SCHEMA_VERSION", "acceptance-next"):
            key_after_algorithm_change = retrieval_cache_key(kb_version="kb_v1", **base)
        self.assertNotEqual(key_v1, key_after_algorithm_change)

    def test_retrieval_result_serialization_roundtrip(self) -> None:
        """RetrievalResult 可以安全写入并恢复。"""
        result = RetrievalResult(
            hits=[
                RetrievalHit(
                    document=Document(page_content="标准答案", metadata={"source": "hr", "answer": "办理入职。"}),
                    score=0.91,
                )
            ],
            query="新人入职流程",
            source_type="faq",
            elapsed_ms=12.5,
        )

        restored = retrieval_result_from_payload(retrieval_result_to_payload(result))

        self.assertEqual(restored.query, "新人入职流程")
        self.assertEqual(restored.source_type, "faq")
        self.assertEqual(restored.hits[0].document.metadata["source"], "hr")
        self.assertEqual(restored.hits[0].score, 0.91)

    def test_cache_manager_invalidates_by_epoch(self) -> None:
        """推进 epoch 后，相同查询不会命中旧 key。"""
        redis_cache = FakeRedisJsonCache()
        namespace_store = FakeNamespaceStore()
        manager = CacheManager(redis_cache=redis_cache, namespace_store=namespace_store, enabled=True)
        scope = DataScope()

        key1 = manager.retrieval_key(
            kind="retrieval",
            scenario_id="enterprise_knowledge",
            collection_name="enterprise_faq",
            source_type="faq",
            data_scope=scope,
            kb_version="kb_v1",
            source_filter=None,
            query_variants=["新人入职流程"],
            k=10,
            rerank=False,
        )
        result = RetrievalResult(
            hits=[RetrievalHit(document=Document(page_content="命中", metadata={}), score=0.9)],
            query="新人入职流程",
            source_type="faq",
        )
        manager.set_retrieval_result(key1, result, source_type="faq")
        self.assertIsNotNone(manager.get_retrieval_result(key1))

        manager.invalidate_scenario("enterprise_knowledge")
        key2 = manager.retrieval_key(
            kind="retrieval",
            scenario_id="enterprise_knowledge",
            collection_name="enterprise_faq",
            source_type="faq",
            data_scope=scope,
            kb_version="kb_v1",
            source_filter=None,
            query_variants=["新人入职流程"],
            k=10,
            rerank=False,
        )

        self.assertNotEqual(key1, key2)
        self.assertIsNone(manager.get_retrieval_result(key2))

        status = manager.status(scenario_id="enterprise_knowledge")
        self.assertEqual(status["stats"]["retrieval_hits"], 1)
        self.assertEqual(status["stats"]["retrieval_misses"], 1)
        self.assertEqual(status["stats"]["retrieval_writes"], 1)
        self.assertEqual(status["stats"]["invalidations"], 1)
        self.assertEqual(status["config"]["faq_ttl_seconds"], manager.settings.cache_faq_ttl_seconds)
        self.assertGreaterEqual(len(status["recent_events"]), 1)

    def test_cached_embeddings_only_cache_query_vectors(self) -> None:
        """query embedding 命中缓存，document embedding 不写业务缓存。"""

        class BaseEmbeddings:
            def __init__(self) -> None:
                self.query_calls = 0
                self.doc_calls = 0

            def embed_query(self, text: str) -> list[float]:
                self.query_calls += 1
                return [1.0, 2.0, 3.0]

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                self.doc_calls += 1
                return [[float(len(text))] for text in texts]

        redis_cache = FakeRedisJsonCache()
        manager = CacheManager(redis_cache=redis_cache, namespace_store=FakeNamespaceStore(), enabled=True)
        base = BaseEmbeddings()
        embeddings = CachedEmbeddings(base)

        with patch("qa_core.cache.embedding.get_cache_manager", return_value=manager):
            self.assertEqual(embeddings.embed_query("新人入职流程"), [1.0, 2.0, 3.0])
            self.assertEqual(embeddings.embed_query("新人入职流程"), [1.0, 2.0, 3.0])
            self.assertEqual(embeddings.embed_documents(["a", "bb"]), [[1.0], [2.0]])

        self.assertEqual(base.query_calls, 1)
        self.assertEqual(base.doc_calls, 1)


if __name__ == "__main__":
    unittest.main()
