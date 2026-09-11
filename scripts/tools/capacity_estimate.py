"""知识库容量与参数压力估算。

这个脚本不访问 Milvus，也不调用 LLM。它基于当前场景资料和配置，估算不同知识库规模下
的 chunk 数、向量存储体积、rerank 候选数量和 prompt 上下文字符量，用于回答面试里常见
的“1 万/10 万 chunk 后怎么优化”的问题。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qa_core.config.settings import PROJECT_ROOT, get_settings
from qa_core.indexing.chunking import split_documents
from qa_core.indexing.document_loaders import get_document_loader_spec, load_file
from qa_core.indexing.document_normalizer import normalize_documents
from qa_core.governance.data_scope import resolve_data_scope
from qa_core.scenarios.registry import resolve_scenario
from qa_core.utils import normalize_source_from_path


CAPACITY_REPORT_DIR = PROJECT_ROOT / "reports" / "capacity"


def scan_current_chunks(scenario_id: str | None) -> dict[str, Any]:
    """按当前 loader 和 splitter 估算现有资料 chunk 数。

    调用顺序：命令行入口 -> scan_current_chunks()。
    """
    scenario = resolve_scenario(scenario_id)
    scope = resolve_data_scope()
    chunks = []
    files = []
    for path in sorted(Path(scenario.data_root).rglob("*")):
        if not path.is_file():
            continue
        spec = get_document_loader_spec(path)
        source = normalize_source_from_path(path.parent)
        if spec is None or source not in scenario.valid_sources:
            continue
        raw_docs = load_file(path)
        normalized = normalize_documents(raw_docs, path, source, "capacity_preview", scenario.scenario_id, scope, ["public"])
        file_chunks, _ = split_documents(normalized)
        chunks.extend(file_chunks)
        files.append({"path": str(path), "source": source, "chunks": len(file_chunks)})
    lengths = [len(chunk.page_content or "") for chunk in chunks]
    return {
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.display_name,
        "files": files,
        "file_count": len(files),
        "chunk_count": len(chunks),
        "avg_chunk_chars": round(sum(lengths) / max(len(lengths), 1), 2),
        "max_chunk_chars": max(lengths) if lengths else 0,
    }


def estimate_scale(base: dict[str, Any], scale_chunks: list[int]) -> list[dict[str, Any]]:
    """估算不同 chunk 规模下的存储和检索压力。

    存储估算基于以下假设：
        - embedding_dim = 1024：Qwen text-embedding-v3 输出维度
        - vector_bytes = embedding_dim × 4：每个 float32 占 4 字节
        - metadata_bytes = 1024：每条 chunk 元数据约 1KB（含 source、scenario_id 等）
        - sparse_overhead = chunk_count × 768：SPLADE 稀疏向量索引估算（768 维 BM25 词汇空间）

    调用顺序：命令行入口 -> estimate_scale()。
    """
    settings = get_settings()
    # Qwen text-embedding-v3 固定输出 1024 维向量
    embedding_dim = 1024
    # 每个维度为 float32，占 4 字节
    vector_bytes = embedding_dim * 4
    # 每条 chunk 的元数据（source、scenario_id、chunk_index 等）约 1KB
    metadata_bytes = 1024
    avg_chunk_chars = max(float(base.get("avg_chunk_chars") or settings.child_chunk_size), 1.0)
    rows = []
    for chunk_count in scale_chunks:
        dense_storage_mb = round(chunk_count * vector_bytes / 1024 / 1024, 2)
        metadata_storage_mb = round(chunk_count * metadata_bytes / 1024 / 1024, 2)
        # SPLADE 稀疏向量索引：基于 768 维 BM25 词汇空间的粗略估算
        sparse_overhead_mb = round(chunk_count * 768 / 1024 / 1024, 2)
        rows.append(
            {
                "chunk_count": chunk_count,
                "estimated_dense_vector_mb": dense_storage_mb,
                "estimated_sparse_index_mb": sparse_overhead_mb,
                "estimated_metadata_mb": metadata_storage_mb,
                "estimated_total_mb_without_milvus_index_replica": round(dense_storage_mb + sparse_overhead_mb + metadata_storage_mb, 2),
                "doc_top_k": settings.doc_top_k,
                "faq_top_k": settings.faq_top_k,
                "rerank_top_n": settings.rerank_top_n,
                "final_context_top_n": settings.final_context_top_n,
                "estimated_prompt_context_chars": int(avg_chunk_chars * settings.final_context_top_n),
                "suggestion": build_scale_suggestion(chunk_count),
            }
        )
    return rows


def build_scale_suggestion(chunk_count: int) -> str:
    """根据规模给出检索优化建议。

    阈值说明：
        - ≤ 10,000：典型中小型企业知识库规模，hybrid + rerank 可直接承载
        - 10,000 ~ 100,000：中型规模，需引入更强的 filtering 和监控
        - > 100,000：大型规模，需异步入库、分区治理和更严格的检索路由

    调用顺序：命令行入口 -> build_scale_suggestion()。
    """
    if chunk_count <= 10000:
        return "当前 hybrid + rerank 策略可直接承载，重点关注评测集和入库质量。"
    if chunk_count <= 100000:
        return "建议按 scenario/source/kb_version 强过滤，控制 query_variants 数量，并监控 rerank 耗时。"
    return "建议引入异步入库任务、分 collection 或分区治理、离线评测抽样和更严格的 source 路由。"


def default_output_path(scenario_id: str) -> Path:
    """构建默认容量报告路径。

    调用顺序：命令行入口 -> default_output_path()。
    """
    CAPACITY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return CAPACITY_REPORT_DIR / f"{stamp}_{scenario_id}.json"


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    调用顺序：命令行入口 -> build_parser()。
    """
    parser = argparse.ArgumentParser(description="Estimate RAG capacity and retrieval pressure.")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--scale", action="append", type=int, default=None, help="目标 chunk 规模，可重复。")
    parser.add_argument("--output", default="")
    return parser


def main() -> None:
    """生成容量评估报告。

    调用顺序：命令行入口 -> main()。
    """
    parser = build_parser()
    args = parser.parse_args()
    base = scan_current_chunks(args.scenario)
    scale_chunks = args.scale or [base["chunk_count"], 10000, 100000, 1000000]
    payload = {
        "report_type": "capacity_estimate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base": base,
        "scales": estimate_scale(base, scale_chunks),
    }
    output_path = Path(args.output) if args.output else default_output_path(str(base["scenario_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
