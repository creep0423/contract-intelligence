"""检索 store 缓存失效策略测试。"""

from qa_core.pipeline import runtime as runtime_module
from qa_core.retrieval import factory as factory_module


def test_active_kb_version_change_clears_cached_hybrid_stores(monkeypatch) -> None:
    """active 版本变化后，已缓存的 Milvus store wrapper 必须失效。

    调用顺序：pytest/unittest 测试入口 -> test_active_kb_version_change_clears_cached_hybrid_stores()。
    """
    cache_clear_count = 0

    class FakeHybridStoreGetter:
        def __call__(self, collection_name: str) -> object:
            return object()

        def cache_clear(self) -> None:
            nonlocal cache_clear_count
            cache_clear_count += 1

    monkeypatch.setattr(factory_module, "get_hybrid_store", FakeHybridStoreGetter())
    factory_module.clear_retrieval_store_cache()
    try:
        assert cache_clear_count == 1

        factory_module.sync_retrieval_cache_for_active_version("enterprise_knowledge", "kb_v1")
        assert cache_clear_count == 1

        factory_module.sync_retrieval_cache_for_active_version("enterprise_knowledge", "kb_v2")
        assert cache_clear_count == 2
    finally:
        factory_module.clear_retrieval_store_cache()


def test_create_query_context_syncs_cache_for_resolved_active_version(monkeypatch) -> None:
    """未显式指定 kb_version 时，active 版本快照参与检索缓存失效判断。"""
    sync_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(runtime_module, "resolve_active_kb_version", lambda requested, scenario_id: "kb_active")
    monkeypatch.setattr(
        runtime_module,
        "sync_retrieval_cache_for_active_version",
        lambda scenario_id, active_kb_version: sync_calls.append((scenario_id, active_kb_version)),
    )

    context = runtime_module.create_query_context(
        history=None,
        query="入职流程是什么",
        source_filter=None,
        session_id="session-sync-active",
        requested_kb_version=None,
        scenario_id="enterprise_knowledge",
        tenant_id=None,
        dataset_id=None,
        visibility=None,
        user_role=None,
        user_roles=None,
    )

    assert context.active_kb_version == "kb_active"
    assert sync_calls == [("enterprise_knowledge", "kb_active")]


def test_create_query_context_does_not_sync_cache_for_explicit_kb_version(monkeypatch) -> None:
    """显式历史版本查询不代表 active 指针切换，不应清理全局检索 store 缓存。"""
    sync_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        runtime_module,
        "resolve_active_kb_version",
        lambda requested, scenario_id: requested or "kb_active",
    )
    monkeypatch.setattr(
        runtime_module,
        "sync_retrieval_cache_for_active_version",
        lambda scenario_id, active_kb_version: sync_calls.append((scenario_id, active_kb_version)),
    )

    context = runtime_module.create_query_context(
        history=None,
        query="入职流程是什么",
        source_filter=None,
        session_id="session-explicit-version",
        requested_kb_version="kb_history_v1",
        scenario_id="enterprise_knowledge",
        tenant_id=None,
        dataset_id=None,
        visibility=None,
        user_role=None,
        user_roles=None,
    )

    assert context.active_kb_version == "kb_history_v1"
    assert sync_calls == []
