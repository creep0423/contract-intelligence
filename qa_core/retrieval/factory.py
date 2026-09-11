"""检索集合工厂与启动预热。

这个模块只负责“拿到检索对象”和“启动时预热检索依赖”。真正执行 Milvus 查询的逻辑在
`store.py`。这里通过 collection_name 缓存 MilvusHybridStore，避免每次请求重复创建
连接对象。
"""
from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal

from qa_core.config.logging_config import get_logger
from qa_core.governance.kb_versions import resolve_active_kb_version
from qa_core.scenarios.registry import get_scenario_registry, resolve_scenario

logger = get_logger(__name__)

_known_active_versions: dict[str, str] = {}
_retrieval_cache_lock = threading.RLock()
_warmup_lock = threading.Lock()
_warmup_thread: threading.Thread | None = None
_warmup_state: dict[str, object] = {
    "status": "not_started",
    "started_at": None,
    "finished_at": None,
    "elapsed_seconds": None,
    "summary": None,
    "error": None,
}


@lru_cache(maxsize=32)
def get_hybrid_store(collection_name: str):
    """按 collection_name 返回已缓存的 Milvus 混合检索封装，多业务场景间隔离集合连接。

    LRU 缓存最多保留 32 个集合实例，足够覆盖当前冻结场景包，同时避免无界增长。

    把 lru_cache 放在 get_hybrid_store 而非顶层，是为了让 warmup_retrieval_stack 可以
    显式遍历每个 collection 触发缓存填充，而顶层调用方的无参 get_faq_store/get_doc_store
    仍享受缓存复用——预热与按需获取共享同一缓存空间。

    参数：
        collection_name: Milvus collection 名称。

    返回：
        已缓存或新创建的 MilvusHybridStore。

    调用顺序：检索准备或检索执行 -> get_hybrid_store()。
    """
    from qa_core.retrieval.store import MilvusHybridStore

    return MilvusHybridStore(collection_name)

def _active_scenario_collection(kind: Literal["faq", "doc"]) -> str:
    """返回当前默认场景的 FAQ 或文档 collection 名。

    参数：
        kind: faq 表示 FAQ 集合，doc 表示文档集合。

    返回：
        当前默认场景配置的 collection 名。

    调用顺序：检索准备或检索执行 -> _active_scenario_collection()。
    """
    # 解析当前业务场景，获取其对应的集合配置
    scenario = resolve_scenario()
    return scenario.faq_collection if kind == "faq" else scenario.doc_collection

def get_faq_store(collection_name: str | None = None):
    """返回已缓存的 FAQ 混合集合封装。

    参数：
        collection_name: 可选集合名；为空时使用当前默认场景的 faq_collection。

    返回：
        FAQ MilvusHybridStore。

    调用顺序：检索准备或检索执行 -> get_faq_store()。
    """
    return get_hybrid_store(collection_name or _active_scenario_collection("faq"))


def get_doc_store(collection_name: str | None = None):
    """返回已缓存的文档混合集合封装。

    参数：
        collection_name: 可选集合名；为空时使用当前默认场景的 doc_collection。

    返回：
        文档 MilvusHybridStore。

    调用顺序：检索准备或检索执行 -> get_doc_store()。
    """
    return get_hybrid_store(collection_name or _active_scenario_collection("doc"))


def clear_retrieval_store_cache() -> None:
    """清空进程内检索 store 缓存和 active 版本快照。

    用于测试或管理动作。模型缓存不在这里清理，避免每次版本切换都重新加载 BGE / Reranker。

    调用顺序：检索准备或检索执行 -> clear_retrieval_store_cache()。
    """
    with _retrieval_cache_lock:
        get_hybrid_store.cache_clear()
        _known_active_versions.clear()


def sync_retrieval_cache_for_active_version(scenario_id: str, active_kb_version: str) -> None:
    """active kb_version 变化时清空已缓存的 Milvus store wrapper。（★★★ 核心）

    知识库重建可能会 drop/recreate Milvus collection。若 API 进程已经预热过旧 collection，
    LangChain Milvus wrapper 会继续持有旧连接状态，导致 MySQL active 已切换但检索召回为空。
    因此每次请求解析到 active 版本后做一次轻量比对，发现版本变化就清空 store 缓存。

    执行流程：
      1. 查询 _known_active_versions 中记录的上一次版本号。
      2. 无记录（首次请求）或版本相同时不做操作。
      3. 版本变化时：清空 lru_cache 的 Milvus wrapper、清除全部版本快照、更新为当前版本。

    参数：
        scenario_id: 业务场景 ID。
        active_kb_version: 当前请求解析出的 active 知识库版本号。

    调用顺序：检索准备或检索执行 -> sync_retrieval_cache_for_active_version()。
    """
    with _retrieval_cache_lock:
        previous = _known_active_versions.get(scenario_id)
        if previous is None:
            _known_active_versions[scenario_id] = active_kb_version
            return
        if previous == active_kb_version:
            return

        logger.warning(
            "检测到场景 %s 的 active kb_version 从 %s 切换到 %s，清空 Milvus store 缓存。",
            scenario_id,
            previous,
            active_kb_version,
        )
        get_hybrid_store.cache_clear()
        _known_active_versions.clear()
        _known_active_versions[scenario_id] = active_kb_version


def retrieval_warmup_state() -> dict[str, object]:
    """Return current retrieval warmup status for health and diagnostics."""

    with _warmup_lock:
        return dict(_warmup_state)


def start_retrieval_warmup_background() -> dict[str, object]:
    """Start retrieval stack warmup in a daemon thread and return current state."""

    global _warmup_thread
    with _warmup_lock:
        if _warmup_state["status"] == "running":
            return dict(_warmup_state)
        if _warmup_state["status"] == "ready":
            return dict(_warmup_state)
        _warmup_state.update(
            {
                "status": "running",
                "started_at": _utc_now(),
                "finished_at": None,
                "elapsed_seconds": None,
                "summary": None,
                "error": None,
            }
        )
        _warmup_thread = threading.Thread(
            target=_run_retrieval_warmup_in_thread,
            name="retrieval-warmup",
            daemon=True,
        )
        _warmup_thread.start()
        return dict(_warmup_state)


def wait_for_retrieval_warmup(timeout: float | None = None) -> dict[str, object]:
    """等待检索预热完成；失败或超时会抛错，调用方不得接收用户检索请求。

    预热线程仍独立运行，以满足 langchain-milvus 的事件循环要求；但 API lifespan 启动阶段会调用
    本函数等待其完成。这样既保留预热状态诊断，也保证 BGE、Reranker 和 Milvus collection
    已经就绪，不会把冷启动延迟转嫁给第一个用户。

    参数：
        timeout: 最大等待秒数；None 表示等待预热自然完成。

    返回：
        预热成功后的摘要信息。

    异常：
        TimeoutError: 预热超过 timeout 仍未完成。
        RuntimeError: 预热失败，服务不应继续启动。
    """
    global _warmup_thread
    with _warmup_lock:
        status = str(_warmup_state["status"])
        thread = _warmup_thread

    if status == "not_started" or thread is None:
        start_retrieval_warmup_background()
        with _warmup_lock:
            thread = _warmup_thread

    if thread is not None:
        thread.join(timeout=timeout)

    with _warmup_lock:
        state = dict(_warmup_state)

    if thread is not None and thread.is_alive():
        raise TimeoutError(f"检索栈预热超过等待上限：{timeout}s")
    if state["status"] != "ready":
        raise RuntimeError(f"检索栈预热失败：{state.get('error') or state['status']}")
    return dict(state.get("summary") or {})


def _run_retrieval_warmup_in_thread() -> None:
    """Run warmup in a thread with its own event loop for langchain-milvus."""

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    started = time.perf_counter()
    try:
        summary = warmup_retrieval_stack()
        with _warmup_lock:
            _warmup_state.update(
                {
                    "status": "ready",
                    "finished_at": _utc_now(),
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "summary": summary,
                    "error": None,
                }
            )
    except Exception as exc:
        logger.exception("检索栈后台预热失败")
        with _warmup_lock:
            _warmup_state.update(
                {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "summary": None,
                    "error": str(exc),
                }
            )
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _utc_now() -> str:
    """返回 UTC 时间字符串（ISO 8601 格式），用于检索预热耗时计算。

    独立函数而非内联 datetime 调用，是为了方便测试时替换时钟。

    返回：
        形如 "2026-08-02T10:30:00.123456+00:00" 的 ISO 格式字符串。

    调用顺序：warmup_retrieval_stack() -> _utc_now()。
    """
    return datetime.now(timezone.utc).isoformat()


def warmup_retrieval_stack() -> dict[str, object]:
    """服务启动时加载检索模型、全部冻结场景集合和当前 active 版本。任一预热失败直接阻断启动。

    执行流程：
      1. 加载 BGE embedding 模型，并对样例问题做一次向量化。
      2. 检查 Milvus 端点是否可达。
      3. 遍历所有场景，解析 active 知识库版本，并触发 FAQ/文档集合懒加载。
      4. 加载 CrossEncoder reranker，并对样例 query-passage 做一次预测。
      5. 记录预热耗时、场景数量、集合数量和 active 版本。

    异常：
        RuntimeError: Milvus 不可达或关键依赖不可用。
    """
    sample_query = "当前业务资料有哪些处理流程"
    started = time.perf_counter()
    from qa_core.retrieval.milvus_compat import milvus_endpoint_available
    from qa_core.retrieval.models import get_embeddings, get_reranker

    # 获取场景注册表，遍历所有业务场景进行预热
    registry = get_scenario_registry()

    # 加载 BGE 向量模型并绕过 query 缓存做一次真实推理，确保模型权重和推理设备都已热身。
    # 仅调用 CachedEmbeddings.embed_query() 可能命中 Redis，导致第一个真实用户查询仍触发冷推理。
    embeddings = get_embeddings()
    getattr(embeddings, "base_embeddings", embeddings).embed_query(sample_query)

    # 检查 Milvus 服务端点是否可达，不可达则启动失败
    if not milvus_endpoint_available(timeout=3.0):
        raise RuntimeError("Milvus 服务不可达：请先启动 Milvus 2.5+ 服务。")

    warmed_collections: list[str] = []
    active_versions: dict[str, str] = {}
    # 遍历所有场景逐一预热，保证任意场景的第一个用户请求零冷启动延迟
    for scenario in registry.list_runtime_scenarios():
        # 解析每个场景的 active 知识库版本，若缺失则启动时直接暴露
        active_versions[scenario.scenario_id] = resolve_active_kb_version(None, scenario.scenario_id)
        for collection_name in (scenario.faq_collection, scenario.doc_collection):
            # 触发 Milvus 集合的懒加载连接，提前初始化 collection
            _ = get_hybrid_store(collection_name).store
            warmed_collections.append(collection_name)

    # 加载 CrossEncoder 重排模型并对样例对做预测，验证 reranker 可用
    get_reranker().predict([(sample_query, "业务资料包含处理流程、常见问题和操作规范。")])

    summary = {
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "scenario_count": len(active_versions),
        "collection_count": len(warmed_collections),
        "active_versions": active_versions,
    }
    logger.info(
        "检索栈预热完成：耗时 %.2fs，场景数=%s，集合数=%s，active_versions=%s",
        summary["elapsed_seconds"],
        summary["scenario_count"],
        summary["collection_count"],
        active_versions,
    )
    # 清除旧快照后写入当前 active 版本快照，供后续请求的 sync_retrieval_cache_for_active_version 做版本变化比对
    with _retrieval_cache_lock:
        _known_active_versions.clear()
        _known_active_versions.update(active_versions)
    return summary
