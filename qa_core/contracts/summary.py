"""合同摘要独立生成任务。"""

from __future__ import annotations

from qa_core.contracts.evidence import evidence_blocks, ground_model_evidence
from qa_core.contracts.prompts import SUMMARY_SYSTEM_PROMPT, task_user_prompt
from qa_core.contracts.retrieval import ContractChunkRetriever, ContractRetrievalContext
from qa_core.contracts.schemas import (
    ContractAnalysisResult,
    ContractSummaryOutput,
    SummarySection,
)
from qa_core.contracts.structured_tasks import StructuredTaskRunner


class ContractSummaryService:
    def __init__(self, retriever: ContractChunkRetriever, runner: StructuredTaskRunner) -> None:
        self.retriever = retriever
        self.runner = runner

    def generate(self, context: ContractRetrievalContext, result: ContractAnalysisResult) -> list[SummarySection]:
        chunks = self.retriever.retrieve(context, "summary")
        if not chunks:
            return []
        facts = result.model_dump(mode="json", exclude={"summary"}, exclude_none=True)
        output = self.runner.invoke(
            ContractSummaryOutput,
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            user_prompt=task_user_prompt(
                contract_id=context.contract_id,
                evidence_blocks=evidence_blocks(chunks),
                confirmed_facts=str(facts),
            ),
        )
        return ground_model_evidence(output, chunks).sections
