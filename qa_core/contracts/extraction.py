"""合同事实抽取服务；不生成义务、风险或摘要。"""

from __future__ import annotations

from pydantic import BaseModel

from qa_core.contracts.evidence import evidence_blocks, ground_model_evidence
from qa_core.contracts.prompts import (
    BASIC_EXTRACTION_SYSTEM_PROMPT,
    COMMERCIAL_TERMS_SYSTEM_PROMPT,
    LEGAL_TERMS_SYSTEM_PROMPT,
    task_user_prompt,
)
from qa_core.contracts.retrieval import ContractChunkRetriever, ContractRetrievalContext
from qa_core.contracts.schemas import (
    BasicExtractionOutput,
    ContractExtractionResult,
    TermsExtractionOutput,
)
from qa_core.contracts.structured_tasks import StructuredTaskRunner


class ContractExtractionService:
    def __init__(self, retriever: ContractChunkRetriever, runner: StructuredTaskRunner) -> None:
        self.retriever = retriever
        self.runner = runner

    def extract(self, context: ContractRetrievalContext) -> ContractExtractionResult:
        basic = self._run(
            context,
            "basic",
            BasicExtractionOutput,
            BASIC_EXTRACTION_SYSTEM_PROMPT,
        )
        commercial = self._run(
            context,
            "commercial",
            TermsExtractionOutput,
            COMMERCIAL_TERMS_SYSTEM_PROMPT,
        )
        legal = self._run(
            context,
            "legal",
            TermsExtractionOutput,
            LEGAL_TERMS_SYSTEM_PROMPT,
        )
        return ContractExtractionResult(
            basic_info=basic.basic_info,
            payment_terms=_dedupe(commercial.payment_terms),
            delivery_terms=_dedupe(commercial.delivery_terms),
            acceptance_terms=_dedupe(commercial.acceptance_terms),
            termination_terms=_dedupe(legal.termination_terms),
            liability_terms=_dedupe(legal.liability_terms),
            confidentiality_terms=_dedupe(legal.confidentiality_terms),
            ip_terms=_dedupe(legal.ip_terms),
            dispute_terms=_dedupe(legal.dispute_terms),
            other_key_terms=_dedupe(commercial.other_key_terms + legal.other_key_terms),
        )

    def _run(self, context, task: str, schema, system_prompt: str):
        chunks = self.retriever.retrieve(context, task)
        if not chunks:
            return schema()
        output = self.runner.invoke(
            schema,
            task_name=task,
            system_prompt=system_prompt,
            user_prompt=task_user_prompt(
                contract_id=context.contract_id,
                evidence_blocks=evidence_blocks(chunks),
            ),
        )
        return ground_model_evidence(output, chunks)


def _dedupe(items: list[BaseModel]) -> list:
    seen: set[tuple[str, str, str]] = set()
    result = []
    for item in items:
        evidence = getattr(item, "source_evidence", None)
        value = str(
            getattr(item, "extracted_value", None)
            or getattr(item, "title", None)
            or getattr(item, "deliverable", None)
            or getattr(item, "standards", None)
            or ""
        ).strip()
        key = (item.__class__.__name__, value, str(evidence.chunk_id if evidence else ""))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
