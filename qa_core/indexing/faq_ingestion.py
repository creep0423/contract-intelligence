"""FAQ CSV 入库链路。

将 FAQ CSV 文件转换为可写入 Milvus 的 Document 对象并提供完整的入库编排。
FAQ 的 page_content 存储标准问题，标准答案放在 metadata.answer 中，
检索时用问题匹配，召回后将答案作为上下文返回给用户。

设计决策：
- FAQ 采用按版本快照重建模式：先删后写，确保新旧版本 FAQ 不混合。
- 使用 pandas 读取 CSV 而非 csv.DictReader：自动处理 BOM/编码推断/空值填充，
  兼容中英文列名（问题/question、答案/answer）。
- FAQ ID 由 scenario_id + kb_version + source + question 的稳定哈希生成，
  同一标准问题不同答案时加入 answer 参与哈希避免 ID 冲突。

依赖分层：
- qa_core.scenarios.registry：场景定义和 valid_sources 白名单。
- qa_core.governance：数据域隔离和知识库版本管理。
- qa_core.retrieval.factory：Milvus FAQ 集合写入。
"""

from __future__ import annotations

import pandas as pd
from langchain_core.documents import Document

from qa_core.config.logging_config import get_logger
from qa_core.quality.faq import _resolve_csv_source
from qa_core.governance.data_scope import resolve_data_scope
from qa_core.governance.kb_versions import get_kb_version_store, version_metadata
from qa_core.indexing.source_normalization import normalize_faq_source
from qa_core.retrieval.factory import get_faq_store
from qa_core.scenarios.registry import resolve_scenario
from qa_core.utils import stable_hash
logger = get_logger(__name__)

def faq_documents_from_csv(
    csv_path: str,
    kb_version: str | None = None,
    version_seq: int | None = None,
    scenario_id: str | None = None,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    allowed_roles: list[str] | None = None,
) -> tuple[list[Document], list[str]]:
    """把 FAQ CSV 转换为可写入 Milvus 的 Document 对象列表。（★★★ 核心）

    每条 FAQ 行生成一个 Document：page_content=标准问题，metadata 包含
    标准答案、source、数据域隔离信息和版本元数据。重复行（相同问题+答案）
    自动跳过。

    参数：
        csv_path: FAQ CSV 文件路径，支持中文列名（问题/答案）和英文列名（question/answer）。
        kb_version: 知识库版本号（可选）。
        version_seq: 版本序号，用于引用式增量的有效期视图（可选）。
        scenario_id: 业务场景标识（可选，默认从 ACTIVE_SCENARIO_ID 读取）。
        tenant_id: 租户 ID（可选）。
        dataset_id: 数据集 ID（可选）。
        visibility: 可见级别（可选）。
        allowed_roles: 允许检索的角色列表（可选）。

    返回：
        (documents_list, faq_ids_list) 元组。
        documents_list: 待写入 Milvus 的 Document 列表。
        faq_ids_list: 对应的 FAQ ID 列表，用于后续删除和统计。

    调用顺序：入库脚本 -> ingest_faq_csv() -> faq_documents_from_csv()。
    """
    scenario = resolve_scenario(scenario_id)
    data_scope = resolve_data_scope(tenant_id=tenant_id, dataset_id=dataset_id, visibility=visibility, user_roles=allowed_roles)
    version_meta = version_metadata(kb_version, scenario.scenario_id, version_seq=version_seq)
    # 原因： pandas 自动处理 BOM/编码推断/空值填充，而 csv.DictReader 只做逐行原始解析，遇到同名列合并或编码抖动需要额外编排才能达到同等健壮性
    data = pd.read_csv(csv_path, encoding="utf-8")
    docs: list[Document] = []
    ids: list[str] = []
    seen_ids: set[str] = set()
    for _, row in data.iterrows():
        # 原因： 中文客户 CSV 可能用中文列名（问题/答案）也可能用英文列名（question/answer），同时兼容两种 header 减少运维沟通成本
        question = str(row.get("问题") or row.get("question") or "").strip()
        answer = str(row.get("答案") or row.get("answer") or "").strip()
        subject = _resolve_csv_source(dict(row))
        # 跳过问题或答案缺失的行：FAQ 必须同时有标准问题和标准答案才有入库价值
        if not question or not answer:
            continue

        source = normalize_faq_source(subject, scenario=scenario, question=question)
        faq_id = stable_hash(scenario.scenario_id, kb_version or "", source, question)
        if faq_id in seen_ids:
            # 同一标准问题但答案不同，使用答案参与 hash，避免 id 冲突。
            faq_id = stable_hash(scenario.scenario_id, kb_version or "", source, question, answer)
        if faq_id in seen_ids:
            # 加入答案后仍然冲突（完全重复行），跳过重复记录
            continue
        seen_ids.add(faq_id)
        docs.append(
            Document(
                # page_content 仅存标准问题，答案放在 metadata 中，检索时用问题匹配
                # 召回后将答案作为上下文返回给用户
                page_content=question,
                metadata={
                    "faq_id": faq_id,
                    "scenario_id": scenario.scenario_id,
                    # FAQ 采用按版本快照重建模式；valid_from_seq/valid_to_seq 只是公共版本字段。
                    "source_type": "faq",
                    "record_type": "faq",
                    "versioning_mode": "snapshot",
                    "version_filter_mode": "kb_version_exact",
                    **data_scope.metadata(allowed_roles=allowed_roles),
                    "standard_question": question,
                    "answer": answer,
                    "source": source,
                    "subject_name": subject,
                    "status": "published",
                    **version_meta,
                },
            )
        )
        ids.append(faq_id)
    return docs, ids


def ingest_faq_csv(
    csv_path: str,
    *,
    scenario_id: str | None = None,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    allowed_roles: list[str] | None = None,
    kb_version: str | None = None,
    create_new_version: bool = False,
    description: str = "",
) -> int:
    """从 CSV 重新构建 FAQ 记录并写入 Milvus FAQ 混合集合。（★★★ 核心）

    完整入库流程：
    1. 解析场景配置，获取或创建知识库版本记录。
    2. 调用 faq_documents_from_csv 将 CSV 转换为 Document 列表。
    3. 先删后写（delete_ids + add_documents）：FAQ 采用整体替换策略，
       确保 FAQ ID 包含 kb_version，新旧版本不混合。
    4. 记录入库统计到版本控制面。

    参数：
        csv_path: FAQ CSV 文件路径，支持中英文列名。
        scenario_id: 业务场景标识（可选）。
        tenant_id: 租户 ID（可选）。
        dataset_id: 数据集 ID（可选）。
        visibility: 可见级别（可选）。
        allowed_roles: 允许检索的角色列表（可选）。
        kb_version: 知识库版本号（可选，不传时使用 active 版本或自动生成）。
        create_new_version: 是否强制创建新版本，默认 False。
        description: 版本描述（创建新版本时使用）。

    返回：
        成功写入的 FAQ 记录数。

    调用顺序：入库脚本或索引服务 -> ingest_faq_csv() -> faq_documents_from_csv()。
    """
    scenario = resolve_scenario(scenario_id)
    version_store = get_kb_version_store(scenario.scenario_id)
    # 确保版本记录存在：如果 create_new=True 则自动生成新版本号；否则使用已有版本或 active 版本
    version = version_store.ensure_version(
        kb_version,
        create_new=create_new_version,
        description=description,
        created_by="ingest_faq_csv",
    )
    active_kb_version = version.kb_version
    docs, ids = faq_documents_from_csv(
        csv_path,
        active_kb_version,
        scenario_id=scenario.scenario_id,
        version_seq=version.version_seq,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        visibility=visibility,
        allowed_roles=allowed_roles,
    )
    store = get_faq_store(scenario.faq_collection)
    # 先删除再写入：FAQ 整体替换策略，确保旧版本 FAQ 不会与新版本 FAQ 残留混合
    store.delete_ids(ids)
    store.add_documents(docs, ids=ids)
    version_store.record_ingest_result(active_kb_version, content_type="faq", count=len(docs))
    logger.info("Ingested %s FAQ records from %s, kb_version: %s", len(docs), csv_path, active_kb_version)
    return len(docs)
