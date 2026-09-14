"""文档切分策略：parent-child 多粒度切片，平衡检索精度与上下文完整性。

把标准化后的 Document 按 parent-child 双层策略切分：
  - parent 块：较大的文本片段（默认 1024 字符），提供完整上下文窗口。
  - child 块：较小的子片段（默认 256 字符），用于精确语义匹配。
  检索时子块命中后携带父块上下文一同返回，实现在精确召回率和上下文
  完整性之间的平衡。

设计决策：
- 表格行和已复核 OCR 文本不经过父子切分——它们已经是治理后的完整证据单元。
- Markdown 文件先通过标题切分器结构化（按 #/##/### 分层），再进入递归切分。
- chunk_id 和 parent_id 由内容及稳定来源位置共同生成，同一 occurrence 重跑保持相同 id。

依赖分层：
- langchain_text_splitters：MarkdownHeaderTextSplitter、RecursiveCharacterTextSplitter。
- qa_core.config.settings：parent_chunk_size / child_chunk_size 等配置。
- qa_core.utils.stable_hash：基于内容生成稳定 id。
"""

from __future__ import annotations
import re
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from qa_core.config.settings import get_settings
from qa_core.document_metadata import is_reviewed_ocr_metadata, is_table_metadata
from qa_core.utils import stable_hash

CHINESE_SEPARATORS = [
    "\n\n",
    "\n",
    "。", "！", "？", "；",
    ";", ".", "!", "?",
    "，", ",",
    " ",
    "",
    # 原因： 中英文混排文档需要同时支持中文句号/感叹号/问号和英文句点/分号作为切分边界，递归切分器按 separator 顺序优先匹配大粒度分隔符
]

# 合同章节/条款标识只影响显式标记为 contract_document 的资料，普通知识库仍走原切分链路。
CONTRACT_SECTION_RE = re.compile(r"^\s*(第[一二三四五六七八九十百零〇0-9]+章)\s*(.*)$")
CONTRACT_CLAUSE_RE = re.compile(
    r"^\s*((?:第[一二三四五六七八九十百零〇0-9]+条)|(?:\d+(?:\.\d+){1,3})|(?:\d+[、.．]))\s*(.*)$"
)
CONTRACT_CLAUSE_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("付款|支付|价款|预付款|尾款|结算", "payment"),
    ("交付|发货|交货|实施|上线", "delivery"),
    ("验收|签收", "acceptance"),
    ("违约|违约金|赔偿|责任上限", "liability"),
    ("解除|终止|续约", "termination"),
    ("保密", "confidentiality"),
    ("知识产权|著作权|专利", "intellectual_property"),
    ("数据安全|个人信息|网络安全", "data_security"),
    ("不可抗力", "force_majeure"),
    ("争议|管辖|仲裁|法院", "dispute_resolution"),
    ("质保|保修|售后", "warranty"),
    ("发票|开票", "invoice"),
)


def infer_contract_clause_type(text: str) -> str:
    """用稳定关键词给合同条款打业务类型标签，便于检索诊断和前端展示。"""
    for pattern, clause_type in CONTRACT_CLAUSE_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            return clause_type
    return "general"


def _contract_structural_blocks(text: str) -> list[tuple[str, str, str]]:
    """按章节和条款边界聚合原文，返回 (section, clause_number, clause_text)。"""
    blocks: list[tuple[str, str, str]] = []
    section = ""
    clause_number = ""
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        content = "\n".join(current).strip()
        if content:
            blocks.append((section, clause_number, content))
        current = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current and current[-1] != "":
                current.append("")
            continue
        section_match = CONTRACT_SECTION_RE.match(line)
        clause_match = CONTRACT_CLAUSE_RE.match(line)
        if section_match:
            flush()
            section = " ".join(part for part in section_match.groups() if part).strip()
            clause_number = ""
            current = [line]
            continue
        if clause_match:
            flush()
            clause_number = clause_match.group(1).rstrip("、.．")
            current = [line]
            continue
        current.append(line)
    flush()
    return blocks


def split_contract_documents(documents: list[Document]) -> tuple[list[Document], list[str]]:
    """在原 parent-child 策略上增加合同结构边界，完整条款优先作为 parent。

    仅供动态合同上传链路调用；没有可识别条款的页面仍按自然段和原递归分隔符切分。
    """
    settings = get_settings()
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_overlap,
        separators=CHINESE_SEPARATORS,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_overlap,
        separators=CHINESE_SEPARATORS,
    )
    chunks: list[Document] = []
    ids: list[str] = []
    for document_index, doc in enumerate(documents):
        blocks = _contract_structural_blocks(doc.page_content)
        if not blocks:
            blocks = [("", "", doc.page_content)]
        for block_index, (section, clause_number, clause_text) in enumerate(blocks):
            if not clause_text.strip():
                continue
            base_metadata = {
                **doc.metadata,
                "content_type": "contract_clause",
                "section": section,
                "clause_number": clause_number,
                "clause_no": clause_number,
                "clause_title": clause_text.splitlines()[0][:160],
                "clause_type": infer_contract_clause_type(clause_text),
                "page_number": int(doc.metadata.get("page_index", 0)) + 1,
            }
            parent_docs = parent_splitter.split_documents(
                [Document(page_content=clause_text, metadata=base_metadata)]
            )
            for parent_index, parent_doc in enumerate(parent_docs):
                parent_content = parent_doc.page_content.strip()
                child_docs = child_splitter.split_documents([parent_doc])
                parent_occurrence = (
                    f"document:{document_index}:page:{base_metadata['page_number']}:"
                    f"block:{block_index}:parent:{parent_index}"
                )
                for child_index, child_doc in enumerate(child_docs):
                    metadata = {
                        **child_doc.metadata,
                        "parent_content": parent_content,
                        "parent_occurrence": parent_occurrence,
                        "chunk_occurrence": f"{parent_occurrence}:child:{child_index}",
                    }
                    parent_id, chunk_id = chunk_identity(child_doc.page_content, metadata)
                    metadata.update({"parent_id": parent_id, "parent_chunk_id": parent_id, "chunk_id": chunk_id})
                    chunks.append(Document(page_content=child_doc.page_content, metadata=metadata))
                    ids.append(chunk_id)
    return chunks, ids

def chunk_identity(page_content: str, metadata: dict) -> tuple[str, str]:
    """基于正文、标准元数据和可选稳定 occurrence 生成 parent_id 与 chunk_id。

    调用顺序：入库脚本或索引服务 -> chunk_identity()。
    """
    parent_content = str(metadata.get("parent_content") or page_content or "").strip()
    parent_occurrence = str(metadata.get("parent_occurrence") or "")
    chunk_occurrence = str(metadata.get("chunk_occurrence") or "")
    if is_table_metadata(metadata):
        # 表格行的 parent_id 额外包含 table_id、sheet_name、row_number，确保同一张表的同一行
        # 在多次入库时 id 稳定，且不同行的 chunk 不会混淆；行列关系由 table_id + row_number 唯一锁定
        parent_id = stable_hash(
            metadata.get("scenario_id"),
            metadata.get("kb_version"),
            metadata.get("embedding_model_version"),
            metadata.get("chunk_schema_version"),
            metadata.get("doc_id"),
            metadata.get("table_id"),
            metadata.get("sheet_name"),
            metadata.get("row_number"),
            parent_occurrence,
            parent_content,
        )
        # 表格行通常不再切分；可选 occurrence 仍用于区分不同来源位置上的重复行。
        chunk_id = stable_hash(parent_id, chunk_occurrence, parent_content)
        return parent_id, chunk_id

    parent_id = stable_hash(
        metadata.get("scenario_id"),
        metadata.get("kb_version"),
        metadata.get("embedding_model_version"),
        metadata.get("chunk_schema_version"),
        metadata.get("doc_id"),
        parent_occurrence,
        parent_content,
    )
    # 子块正文和稳定 occurrence 同时参与 hash：正文保持内容可追溯，occurrence 区分
    # 同一父块或不同页面上文本完全一致的真实出现位置。
    chunk_id = stable_hash(parent_id, chunk_occurrence, page_content)
    return parent_id, chunk_id


def split_documents(documents: list[Document]) -> tuple[list[Document], list[str]]:
    """将文档切成可检索的子块并保留父块上下文。（★★★ 核心）

    子块用于精确召回，父块上下文保存在 metadata.parent_content 中。
    检索时子块命中后携带父块上下文一同返回。

    执行流程：
      1. 对每个 Document，根据 file_type 和 metadata 选择切分策略。
      2. 表格行和已复核 OCR：不切分，整行/整段作为唯一块。
      3. Markdown：先按标题切分（#/##/###），再递归文本切分。
      4. 其他文本：直接按 parent_chunk_size 递归切分。
      5. 对每个父块生成 child 子块，计算稳定 parent_id 和 chunk_id。

    参数：
        documents: 标准化后的 LangChain Document 列表。

    返回：
        (chunks_list, ids_list) 元组。chunks_list 包含所有子块，每个块
        携带 parent_content（父块全文）和 parent_id/chunk_id。
        ids_list 是所有子块的 chunk_id 列表，与 chunks_list 一一对应。

    调用顺序：入库脚本或索引服务 -> split_documents()。
    """
    # 原因： parent-child 分别切分使子块保持精确命中而父块提供完整上下文窗口，比单一切片在精确召回率和上下文完整性之间取得更好平衡
    settings = get_settings()
    markdown_headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_overlap,
        separators=CHINESE_SEPARATORS,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_overlap,
        separators=CHINESE_SEPARATORS,
    )

    # ── 主处理循环：逐个 Document 执行切分 ──
    chunks: list[Document] = []
    ids: list[str] = []
    for doc in documents:
        file_type = str(doc.metadata.get("file_type", "")).lower()
        parent_docs: list[Document]
        if is_table_metadata(doc.metadata) or is_reviewed_ocr_metadata(doc.metadata):
            # 表格行和已复核 OCR 文本都是治理后的完整证据单元。
            # 表格不能被拆散行列关系；OCR 复核稿不能丢失复核状态、置信度和原始文件说明。
            parent_content = doc.page_content.strip()
            # 空行跳过：表格中全空行或 OCR 空白页没有检索价值，不生成 chunk 以节省 Milvus 存储
            if not parent_content:
                continue
            metadata = dict(doc.metadata)
            metadata["parent_content"] = parent_content
            parent_id, chunk_id = chunk_identity(parent_content, metadata)
            metadata.update(
                {
                    "parent_id": parent_id,
                    "chunk_id": chunk_id,
                }
            )
            # 表格/OCR 不经过父子切分：直接以整行/整段作为唯一块，parent_id == chunk_id
            chunks.append(Document(page_content=parent_content, metadata=metadata))
            ids.append(chunk_id)
            continue
        elif file_type == ".md":
            header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=markdown_headers)
            # Markdown 标题会先转成结构化元数据，再进入递归切分，能提升来源标签和上下文质量。
            # 如果 Markdown 解析失败，说明资料格式需要修复；入库阶段应该暴露异常并进入
            # 异常文件报告，而不是悄悄按普通文本切分，造成章节 metadata 丢失。
            header_docs = header_splitter.split_text(doc.page_content)
            for header_doc in header_docs:
                # Markdown 标题切分器会生成新的 Document，这里把原始文件 metadata
                # 补回去，避免切分后丢失 source、file_name、doc_id 等关键字段。
                header_doc.metadata.update(doc.metadata)
            parent_docs = parent_splitter.split_documents(header_docs)
        else:
            # 非 Markdown 普通文本：直接按父块大小切分，不再经过标题解析
            parent_docs = parent_splitter.split_documents([doc])

        # ── 父块循环处理：对每个父块生成子块 ──
        for parent_doc in parent_docs:
            parent_content = parent_doc.page_content
            # parent_id 和 chunk_id 都纳入 kb_version、embedding_model_version 和 chunk_schema_version。
            # 这样同一个文件在两个知识库版本里可以同时存在，不会因为内容相同而主键冲突。
            child_docs = child_splitter.split_documents([parent_doc])
            # 子块为空意味着父块小于最小子块尺寸，此时跳过（父块本身仍可检索）
            for child_doc in child_docs:
                # chunk_id 由父块和子块内容共同决定。同一文件未变化时 id 稳定；文件变化时
                # id 会变化，配合 manifest 删除旧 chunk 后重建。
                metadata = dict(child_doc.metadata)
                metadata["parent_content"] = parent_content
                parent_id, chunk_id = chunk_identity(child_doc.page_content, metadata)
                metadata.update(
                    {
                        "parent_id": parent_id,
                        "chunk_id": chunk_id,
                    }
                )
                chunks.append(Document(page_content=child_doc.page_content, metadata=metadata))
                ids.append(chunk_id)
    return chunks, ids
