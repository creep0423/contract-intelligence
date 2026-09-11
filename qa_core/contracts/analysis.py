"""合同分析流程编排；事实、义务、规则风险、语义风险和摘要分阶段执行。"""

from __future__ import annotations

from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings
from qa_core.contracts.extraction import ContractExtractionService
from qa_core.contracts.obligations import ContractObligationService
from qa_core.contracts.retrieval import ContractChunkRetriever, ContractRetrievalContext
from qa_core.contracts.risk import apply_obligation_risk_levels, deduplicate_risks, evaluate_deterministic_risks
from qa_core.contracts.schemas import ContractAnalysisResult
from qa_core.contracts.semantic_risk import SemanticRiskAnalysisService
from qa_core.contracts.store import ContractStore
from qa_core.contracts.structured_tasks import StructuredTaskRunner
from qa_core.contracts.summary import ContractSummaryService


logger = get_logger(__name__)


class ContractAnalysisService:
    """一次分析运行保留独立 run id；详情只读取最新成功运行。"""

    def __init__(
        self,
        store: ContractStore,
        *,
        retriever: ContractChunkRetriever | None = None,
        runner: StructuredTaskRunner | None = None,
    ) -> None:
        self.store = store
        self.retriever = retriever or ContractChunkRetriever()
        self.runner = runner or StructuredTaskRunner()
        self.extraction_service = ContractExtractionService(self.retriever, self.runner)
        self.obligation_service = ContractObligationService(self.retriever, self.runner)
        self.semantic_risk_service = SemanticRiskAnalysisService(self.retriever, self.runner)
        self.summary_service = ContractSummaryService(self.retriever, self.runner)

    def analyze(self, contract: dict) -> ContractAnalysisResult:
        contract_id = str(contract["id"])
        tenant_id = str(contract["tenant_id"])
        run_id = self.store.create_analysis_run(contract_id, model_name=get_settings().llm_model)
        context = ContractRetrievalContext(
            contract_id=contract_id,
            dataset_id=str(contract["dataset_id"]),
            tenant_id=str(contract["tenant_id"]),
            visibility=str(contract["visibility"]),
            allowed_roles=list(contract.get("allowed_roles") or ["public"]),
            scenario_id=str(contract["scenario_id"]),
        )
        stage = "extraction"
        try:
            logger.info("Contract analysis stage: contract_id=%s run_id=%s tenant_id=%s stage=extracting_facts", contract_id, run_id, tenant_id)
            extraction = self.extraction_service.extract(context)
            self.store.save_analysis_extraction(run_id, extraction)
            stage = "obligations"
            logger.info("Contract analysis stage: contract_id=%s run_id=%s tenant_id=%s stage=obligations", contract_id, run_id, tenant_id)
            obligations = self.obligation_service.extract(context, extraction)
            stage = "deterministic_risk"
            logger.info("Contract analysis stage: contract_id=%s run_id=%s tenant_id=%s stage=deterministic_risk", contract_id, run_id, tenant_id)
            rule_risks = evaluate_deterministic_risks(contract_id, extraction, obligations)
            stage = "semantic_risk"
            logger.info("Contract analysis stage: contract_id=%s run_id=%s tenant_id=%s stage=semantic_risk", contract_id, run_id, tenant_id)
            stage_errors: list[str] = []
            try:
                semantic_risks = self.semantic_risk_service.analyze(context, extraction, obligations)
            except Exception as exc:
                semantic_risks = []
                stage_errors.append(f"semantic_risk:{type(exc).__name__}")
                logger.warning(
                    "Contract semantic risk degraded: contract_id=%s run_id=%s tenant_id=%s error_type=%s",
                    contract_id,
                    run_id,
                    tenant_id,
                    type(exc).__name__,
                )
            risks = deduplicate_risks(contract_id, rule_risks + semantic_risks, obligations)
            obligations = apply_obligation_risk_levels(obligations, risks)
            partial = ContractAnalysisResult(extraction=extraction, obligations=obligations, risks=risks)
            stage = "summary"
            logger.info("Contract analysis stage: contract_id=%s run_id=%s tenant_id=%s stage=summary", contract_id, run_id, tenant_id)
            try:
                summary = self.summary_service.generate(context, partial)
            except Exception as exc:
                summary = []
                stage_errors.append(f"summary:{type(exc).__name__}")
                logger.warning(
                    "Contract summary degraded: contract_id=%s run_id=%s tenant_id=%s error_type=%s",
                    contract_id,
                    run_id,
                    tenant_id,
                    type(exc).__name__,
                )
            result = partial.model_copy(update={"summary": summary})
            stage = "persistence"
            self.store.save_analysis_result(
                run_id,
                contract_id,
                result,
                warning_summary=",".join(stage_errors),
            )
            logger.info(
                "Contract analysis succeeded: contract_id=%s run_id=%s tenant_id=%s obligations=%s risks=%s",
                contract_id,
                run_id,
                tenant_id,
                len(obligations),
                len(risks),
            )
            return result
        except Exception as exc:
            self.store.fail_analysis(run_id, contract_id, f"{stage}:{type(exc).__name__}")
            logger.error(
                "Contract analysis failed: contract_id=%s run_id=%s tenant_id=%s error_type=%s",
                contract_id,
                run_id,
                tenant_id,
                type(exc).__name__,
            )
            raise
