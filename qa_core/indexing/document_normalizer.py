"""入库文档元数据标准化：为文档补充全量标准元数据字段。

写入 Milvus 前，为每个 Document 补充以下标准元数据：
  - 来源信息：source、file_path、file_name、file_type、doc_id。
  - 场景信息：scenario_id、kb_version、version_seq。
  - 数据域隔离：tenant_id、dataset_id、visibility、allowed_roles。
  - 版本管理：valid_from_seq/valid_to_seq 有效期窗口。
  - 内容类型：record_type、content_type、page_index。

设计决策：
- doc_id 基于文件路径和内容生成稳定哈希，同一文件多次入库 doc_id 不变。
- content_type 优先保留 loader 已标记的值（如 table_row），loader 未标记时
  默认为 "text"，保证表格行和普通文本的 metadata 结构统一。
- 版本管理字段统一由 version_metadata() 注入，避免各模块各自拼装。
"""

from __future__ import annotations
from pathlib import Path
from langchain_core.documents import Document
from qa_core.governance.data_scope import DataScope, resolve_data_scope
from qa_core.governance.kb_versions import version_metadata
from qa_core.scenarios.registry import resolve_scenario
from qa_core.utils import file_fingerprint

def normalize_documents(
    documents: list[Document],
    file_path: Path,
    source: str,
    kb_version: str | None = None,
    scenario_id: str | None = None,
    version_seq: int | None = None,
    data_scope: DataScope | None = None,
    allowed_roles: list[str] | None = None,
) -> list[Document]:
    """为文档补充项目标准元数据：source、scenario_id、数据域、文件信息、doc_id、版本信息。（★★★ 核心）

    执行流程：
      1. 基于文件路径和内容生成稳定的 doc_id，用于去重和增量检测。
      2. 解析场景配置，获取 scenario_id 和版本元数据。
      3. 为每个 Document 注入 source、kb_version、version_seq、数据域隔离等元数据。
      4. 保留 loader 已标记的元数据（如 page_index、content_type 等）。

    参数：
        documents: LangChain Document 列表（loader 原始输出）。
        file_path: 源文件路径，用于生成 doc_id 和 file_name/file_path 元数据。
        source: 业务分类 source，必须属于场景 valid_sources。
        kb_version: 知识库版本号。
        scenario_id: 场景 ID。
        version_seq: 版本序列号，用于 valid_from_seq。
        data_scope: 数据域对象，含 tenant_id/dataset_id/visibility 等隔离信息。
        allowed_roles: 允许的角色列表，用于 RBAC 过滤。

    返回：
        标准化后的 Document 列表，每个 Document 的 metadata 包含完整标准字段。

    调用顺序：入库脚本或索引服务 -> normalize_documents()。
    """
    # 基于文件路径和内容生成稳定的 doc_id，同一文件多次入库 doc_id 不变，用于去重和增量检测
    doc_id = file_fingerprint(file_path)
    scenario = resolve_scenario(scenario_id)
    # 未显式传入 data_scope 时使用全局默认值（default tenant/dataset/public），
    # 确保所有文档至少有一个安全的数据域边界
    scope = data_scope or resolve_data_scope()
    version_meta = version_metadata(kb_version, scenario.scenario_id, version_seq=version_seq)
    normalized: list[Document] = []
    for index, doc in enumerate(documents):
        metadata = dict(doc.metadata)
        metadata.update(
            {
                "source": source,
                "scenario_id": scenario.scenario_id,
                # 文档采用引用式增量；在线检索按 version_seq 有效期窗口判断可见性。
                "source_type": "doc",
                "record_type": "doc_chunk",
                "versioning_mode": "reference_incremental",
                "version_filter_mode": "validity_window",
                **scope.metadata(allowed_roles=allowed_roles),
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
                "doc_id": doc_id,
                # page_index 优先使用 loader 自带页码，fallback 到文档列表中的索引位置
                "page_index": metadata.get("page", index),
                # content_type 只覆盖空值：如果 loader 已标记 content_type（如 table_row），保持原值不变
                "content_type": metadata.get("content_type") or "text",
                **version_meta,
            }
        )
        normalized.append(Document(page_content=doc.page_content, metadata=metadata))
    return normalized
