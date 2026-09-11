"""合同分析任务级 Hybrid Retrieval，复用现有 Milvus/BGE/reranker。"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from qa_core.config.settings import get_settings
from qa_core.contracts.evidence import build_contract_context
from qa_core.governance.data_scope import resolve_data_scope
from qa_core.governance.kb_versions import resolve_active_kb_version
from qa_core.retrieval.factory import get_doc_store
from qa_core.scenarios.registry import resolve_scenario


@dataclass(frozen=True)
class ContractRetrievalContext:
    contract_id: str
    dataset_id: str
    tenant_id: str
    visibility: str
    allowed_roles: list[str]
    scenario_id: str


TASK_QUERIES: dict[str, list[str]] = {
    "basic": [
        "合同名称 合同编号 合同类型 甲方 乙方 签署日期 生效日期 终止日期 合同期限",
        "合同金额 价款 总金额 币种 人民币",
    ],
    "commercial": [
        "付款 支付 价款 费用 发票 结算 预付款 尾款 付款条件 付款期限",
        "交付 发货 交货 实施 上线 交付物 交付时间 交付标准",
        "验收 验收标准 验收合格 验收期限 验收流程 默认验收",
    ],
    "legal": [
        "解除 终止 自动续约 通知期限 单方解除",
        "违约 赔偿 违约金 责任上限 免责",
        "保密 知识产权 数据安全 不可抗力 争议 管辖法院 仲裁",
    ],
    "obligation": [
        "应当 应在 负责 提交 支付 开票 发货 交付 实施 上线 验收 提供服务",
        "保证金 续约 通知 质保 售后 SLA 保险 审计 期限 履约节点 里程碑",
    ],
    "semantic_risk": [
        "付款条件是否明确 验收标准是否主观 交付标准是否明确",
        "责任是否对等 违约责任 免责范围 知识产权归属 单方终止权 数据安全义务",
        "补充协议 条款冲突 条款变更",
    ],
    "summary": [
        "合同核心交易内容 双方核心义务 付款 交付 验收 违约 解除 争议 关键时间节点",
    ],
}


class ContractChunkRetriever:
    def retrieve(self, context: ContractRetrievalContext, task: str) -> list[Document]:
        queries = TASK_QUERIES.get(task)
        if not queries:
            raise ValueError(f"未知合同分析检索任务：{task}")
        scenario = resolve_scenario(context.scenario_id)
        scope = resolve_data_scope(
            tenant_id=context.tenant_id,
            dataset_id=context.dataset_id,
            visibility=context.visibility,
            user_roles=context.allowed_roles,
        )
        result = get_doc_store(scenario.doc_collection).search_many(
            queries,
            k=get_settings().contract_analysis_retrieval_top_k,
            source_filter="contract",
            kb_version=resolve_active_kb_version(None, scenario.scenario_id),
            data_scope=scope,
            scenario_id=scenario.scenario_id,
            source_type="doc",
            rerank=True,
        )
        return build_contract_context([hit.document for hit in result.hits])
