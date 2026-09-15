"""仅处理需要语义判断的合同风险。"""

from __future__ import annotations

from qa_core.contracts.evidence import evidence_blocks, ground_model_evidence
from qa_core.contracts.prompts import SEMANTIC_RISK_SYSTEM_PROMPT, task_user_prompt
from qa_core.contracts.retrieval import ContractChunkRetriever, ContractRetrievalContext
from qa_core.contracts.schemas import (
    ContractExtractionResult,
    ContractObligation,
    ContractRisk,
    RiskSource,
    SemanticRiskOutput,
)
from qa_core.contracts.structured_tasks import StructuredTaskRunner


class SemanticRiskAnalysisService:
    def __init__(self, retriever: ContractChunkRetriever, runner: StructuredTaskRunner) -> None:
        self.retriever = retriever
        self.runner = runner

    def analyze(
        self,
        context: ContractRetrievalContext,
        extraction: ContractExtractionResult,
        obligations: list[ContractObligation],
    ) -> list[ContractRisk]:
        chunks = self.retriever.retrieve(context, "semantic_risk")
        if not chunks:
            return []
        facts = {
            "extraction": extraction.model_dump(mode="json", exclude_none=True),
            "obligations": [item.model_dump(mode="json", exclude_none=True) for item in obligations],
        }
        output = self.runner.invoke(
            SemanticRiskOutput,
            task_name="semantic_risk",
            system_prompt=SEMANTIC_RISK_SYSTEM_PROMPT,
            user_prompt=task_user_prompt(
                contract_id=context.contract_id,
                evidence_blocks=evidence_blocks(chunks),
                confirmed_facts=str(facts),
            ),
        )
        grounded = ground_model_evidence(output, chunks)
        risks: list[ContractRisk] = []
        for item in grounded.risks:
            if item.risk_type == "potential_conflict":
                document_ids = {
                    evidence.document_id
                    for evidence in ([item.source_evidence] if item.source_evidence else []) + item.supporting_evidence
                    if evidence and evidence.document_id
                }
                if len(document_ids) < 2:
                    continue
            risks.append(item.model_copy(update={"risk_source": RiskSource.LLM}))
        return risks
