"""统一 SourceEvidence 校验，拒绝模型伪造 chunk、引文和页码。"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, TypeVar

from langchain_core.documents import Document
from pydantic import BaseModel

from qa_core.config.settings import get_settings
from qa_core.contracts.schemas import SourceEvidence


T = TypeVar("T", bound=BaseModel)


def _compact(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").translate(
        str.maketrans({"，": ",", "。": ".", "：": ":", "；": ";", "！": "!", "？": "?", "“": '"', "”": '"', "‘": "'", "’": "'"})
    )
    return re.sub(r"\s+", "", normalized)


def validate_source_evidence(
    evidence: SourceEvidence | None,
    allowed_chunks: dict[str, Document],
) -> SourceEvidence | None:
    """证据 chunk 必须在本次模型输入中，引文必须逐字来自该 chunk。"""
    if evidence is None or not evidence.chunk_id:
        return None
    chunk = allowed_chunks.get(str(evidence.chunk_id))
    if chunk is None:
        return None
    quote = str(evidence.quote or "").strip()
    if not quote or _compact(quote) not in _compact(chunk.page_content):
        return None
    page = chunk.metadata.get("page_number")
    return SourceEvidence(
        document_id=str(chunk.metadata.get("contract_document_id") or chunk.metadata.get("doc_id") or "") or None,
        document_name=str(chunk.metadata.get("contract_document_name") or chunk.metadata.get("file_name") or "") or None,
        contract_id=str(chunk.metadata.get("contract_id") or "") or None,
        chunk_id=str(chunk.metadata.get("chunk_id") or "") or None,
        page_number=int(page) if page not in (None, "") else None,
        section=str(chunk.metadata.get("section") or "") or None,
        clause_no=str(chunk.metadata.get("clause_no") or chunk.metadata.get("clause_number") or "") or None,
        clause_title=str(chunk.metadata.get("clause_title") or "") or None,
        quote=quote,
    )


def ground_model_evidence(model: T, chunks: list[Document]) -> T:
    """递归验证任意合同 Structured Output 内的 SourceEvidence。"""
    allowed = {str(doc.metadata.get("chunk_id")): doc for doc in chunks if doc.metadata.get("chunk_id")}
    threshold = get_settings().contract_review_confidence_threshold

    def visit(value: Any) -> Any:
        if isinstance(value, SourceEvidence):
            return validate_source_evidence(value, allowed)
        if isinstance(value, list):
            return [grounded for item in value if (grounded := visit(item)) is not None]
        if not isinstance(value, BaseModel):
            return value
        updates = {name: visit(getattr(value, name)) for name in value.__class__.model_fields}
        if "needs_review" in updates and "confidence" in updates:
            has_fact = bool(updates.get("extracted_value") or updates.get("title") or updates.get("risk_name"))
            missing_evidence = "source_evidence" in updates and updates.get("source_evidence") is None and has_fact
            updates["needs_review"] = bool(updates["needs_review"] or float(updates["confidence"] or 0) < threshold or missing_evidence)
        return value.model_copy(update=updates)

    return visit(model)


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    seen: set[str] = set()
    result: list[Document] = []
    for document in documents:
        key = str(document.metadata.get("chunk_id") or document.page_content)
        if key in seen:
            continue
        seen.add(key)
        result.append(document)
    return result


def build_contract_context(documents: list[Document]) -> list[Document]:
    """按稳定召回顺序去重并限制块数、单块长度和总字符数。"""
    settings = get_settings()
    max_chunks = min(settings.contract_analysis_retrieval_top_k, 24)
    remaining = settings.max_prompt_context_chars
    selected: list[Document] = []
    for document in deduplicate_documents(documents)[:max_chunks]:
        if remaining <= 0:
            break
        content = str(document.page_content or "").strip()
        if not content:
            continue
        content = content[: min(settings.max_context_doc_chars, remaining)].rstrip()
        if not content:
            continue
        selected.append(Document(page_content=content, metadata=dict(document.metadata)))
        remaining -= len(content)
    return selected


def evidence_blocks(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[chunk_id={doc.metadata.get('chunk_id')} document={doc.metadata.get('contract_document_name') or '-'} "
        f"page={doc.metadata.get('page_number', '')} section={doc.metadata.get('section') or '-'} "
        f"clause={doc.metadata.get('clause_no') or doc.metadata.get('clause_number') or '-'}]\n"
        f"{doc.page_content}"
        for doc in build_contract_context(documents)
    )
