"""答案置信度公共计算模块。

检索命中的 ``score`` 只表示候选内容和查询的相关性排序，不等价于最终答案可信度。
本模块提供三段公共纯逻辑：
- ``calculate_evidence_confidence()``：生成前证据置信度；
- ``calculate_generation_confidence()``：LLM 生成后答案支撑度；
- ``combine_answer_confidence()``：最终 ``answer_confidence`` 合并。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


LEVEL_LABELS = {
    "high": "高",
    "medium": "中",
    "low": "低",
}

_CITATION_RE = re.compile(r"\[(\d+)\]")
_REFERENCE_SECTION_RE = re.compile(r"(?:^|\n)\s*(?:参考来源|来源参考|references?)\s*[:：]", re.IGNORECASE)
_CLAIM_SPLIT_RE = re.compile(r"[\r\n。！？!?；;]+")
_CJK_SEGMENT_RE = re.compile(r"[\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{1,}")

__all__ = [
    "AnswerConfidence",
    "calculate_evidence_confidence",
    "calculate_generation_confidence",
    "combine_answer_confidence",
    "confidence_level",
    "evaluate_generated_answer",
    "faq_exact_match",
    "finalize_generated_answer_confidence",
    "mark_answer_confidence_not_applicable",
    "normalize_retrieval_score",
    "record_answer_confidence",
    "record_evidence_confidence",
]


@dataclass(frozen=True)
class AnswerConfidence:
    """生成前证据置信度结果。

    score 是 [0, 1] 区间的工程评分；level 是便于前端展示的粗粒度等级。

    调用顺序：RAG 管线 -> calculate_evidence_confidence() -> AnswerConfidence。
    """

    score: float
    level: str
    reasons: list[str]
    signals: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """转换为 API / Trace 可直接序列化的结构。

        返回：
            dict: 含 score（四舍五入到 0.01）/ level / label / reasons /
            signals 的可序列化字典。

        调用顺序：RAG 管线 -> AnswerConfidence.as_dict()。
        """
        return {
            "score": round(self.score, 2),
            "level": self.level,
            "label": LEVEL_LABELS[self.level],
            "reasons": list(self.reasons),
            "signals": dict(self.signals),
        }


def calculate_evidence_confidence(
    *,
    hit_type: str,
    retrieval_top_score: float,
    context_count: int,
    source_count: int,
    intent_rule_score: float,
    query: str,
    raw_query: str,
    rewritten_query: str | None,
    deterministic_route: bool = False,
    faq_exact_match: bool = False,
    intent_rule_candidate_score: float | None = None,
) -> AnswerConfidence:
    """计算生成前的证据置信度，将多个可解释信号合并为一个 [0, 1] 工程评分。（★★★ 核心）

    通俗理解：这个函数不是在算“答案一定对不对”，而是在算“这次拿到的
    证据够不够扎实，能不能更放心地把答案展示给用户”。它的实际判断方式是：
      - 标准 FAQ 题完全命中，证据最强，从 0.95 起步。
      - 普通 RAG 主要看“搜到的内容和问题像不像”，所以检索分的权重最大。
      - 真正放入 Prompt 的资料片段和独立来源越多，说明证据更厚，会加分；但是各自封顶，
        避免重复资料把分数刷高。
      - 如果系统是靠“那这个呢”这类追问改写才能理解问题，或者意图网关自身拿不准，会扣一点分。
      - 普通 RAG 最终没有选中任何上下文时，最高只能是 0.35：有候选不等于
        有可供生成答案的证据。

    算分例子：用户问“新人入职流程怎么走？”。这次是普通 RAG，检索最高分是 0.80，
    最终放入 Prompt 的资料有 3 条，来自 2 个不同来源，意图网关最终分是 0.84，且没有使用
    追问改写。分数就是：

        0.20（基础分） + 0.80 * 0.45（检索） + 3 * 0.06（上下文）
        + 2 * 0.04（来源） + 0.84 * 0.12（意图） = 0.9208，最终四舍五入为 0.92。

    因为这次不是依赖历史改写，且意图分不低于 0.70，所以不会触发扣分；0.92 对应 ``high``
    等级。

    分数解读：≥0.82 表示生成前的证据较扎实，0.55~0.81 表示可以尝试回答但应更谨慎，
    低于 0.55 表示资料不足、检索较弱或路由不稳。这个分数不能替代 Recall@K、MRR
    或人工复核，也不能单独代表 LLM 最终生成文本的事实正确率。

    核心思想：
      这个函数回答的问题是——“在调用 LLM 之前，当前检索证据有多大的支撑能力？”
      它评估的是**生成条件是否可靠**，不是**生成结果的统计正确率**。

      打分依据五类信号：
      - 回答路径（确定性路由 > FAQ 精确命中 > FAQ 相似 > 文档 RAG > 信息不足）
      - 检索相关性（排序后第一条候选的 top-1 检索分，归一化后贡献最大权重）
      - 证据厚度（上下文条数 + 来源个数，但封顶，防止堆砌候选刷分）
      - 意图稳定性（意图决策分，路由不确定时扣分）
      - 历史依赖（是否依靠了追问改写来补全问题）

      最终 ``score`` 是一个**生成前工程指标**，用于决定是否具备进入 LLM 生成的证据基础，
      也用于前端诊断、Trace 排查和 Bad Case 分桶。生成完成后，主流程还会调用
      ``finalize_generated_answer_confidence()``，把答案中的引用覆盖、引用合法性和
      答案与上下文的词面支撑度纳入最终分数。

    top-1 检索分的含义：
      检索阶段会返回多条候选，并按相关性从高到低排序。``retrieval_top_score`` 只取排序
      第一条候选的分数，而不是把所有候选分数求平均，也不是取进入 Prompt 的文档数量。
      例如候选分数为 ``0.91、0.78、0.62``，top-1 检索分就是 ``0.91``。它代表“当前
      最强的一条证据与问题有多匹配”。之所以给它最大的权重，是因为最强证据通常决定
      这次回答有没有明确落点；如果第一条证据都不相关，仅靠增加几条弱相关资料不应该
      把答案置信度抬得很高。需要注意的是，top-1 仍然只说明“最相关”，不能证明
      LLM 一定正确，所以生成后还要进行独立核验。

    分支逻辑（从高到低的信任阶梯）：
      - **确定性路由**（问候/越界/转人工）：默认高置信，意图分只做小幅微调。
      - **FAQ 精确命中**：直接给 0.95 满档，不参与其他信号折扣——标准问题一字不差
        地匹配是最强的证据，不应被无关信号拉低。
      - **FAQ 相似命中**：以检索相似度为主力，意图分做辅助。
      - **信息不足 (insufficient_context)**：基线很低，弱召回只能轻微抬分，
        不可能越过中高置信区间。
      - **通用文档 RAG**：检索相关性权重最高，上下文和来源提供有限增益且各自封顶，
        意图分占小部分。

    稳定性折扣（在基础分之上扣减）：
      - **追问改写扣分**：如果当前问题经过了历史改写（说明原始提问信息不完整），
        扣除 0.05~0.08。上下文越多扣越少，因为更多证据可以弥补改写误差。
      - **低意图扣分**：意图决策分 < 0.70 且非确定路由时，扣 0.04。
        反映”路由本身就有点拿不准”，确定性路由不重复扣。
      - **空上下文封顶**：RAG 路径但最终没有上下文入选时，分数强制 ≤ 0.35。
        这是兜底安全网——有候选 ≠ 有可用证据。

    输出等级（供前端展示）：
      - ``high``（≥0.82）：放心展示。
      - ``medium``（≥0.55）：可展示，建议附带核实提示。
      - ``low``（<0.55）：信息不足或路由不稳，应谨慎处理。

    参数：
        hit_type: 回答路径类型，常见值：``"rag"``、``"faq_direct"``、
            ``”insufficient_context”``。确定性回答通过 ``deterministic_route`` 标记。
        retrieval_top_score: 检索/重排后按降序排列的第一条候选相关性分，也就是 top-1 分。
            可大于 1，内部平滑归一化。
        context_count: 最终放入 Prompt 的上下文片段数（不是原始召回数）。
        source_count: 最终返回给用户的去重来源数量。
        intent_rule_score: 意图决策网关最终采用的分数，参与基础分和低意图扣分。
        query: 当前用于业务处理的归一化问题。
        raw_query: 用户原始输入，用于判断追问改写。
        rewritten_query: 改写后的问题，无改写时传 ``None``。
        deterministic_route: 是否为确定性路由（问候/越界/转人工）。
        faq_exact_match: FAQ 是否与标准问题精确匹配。
        intent_rule_candidate_score: 意图规则候选分，仅输出到诊断信号；
            未传入时回退为 ``intent_rule_score``。

    返回：
        AnswerConfidence: 包含 score（[0,1] 工程分）、level（high/medium/low）、
        reasons（触发了哪些路径/折扣）和 signals（所有原始/派生信号的快照）。

    调用顺序：RAG 管线 -> record_evidence_confidence() -> calculate_evidence_confidence()。
    """
    # 检索分只衡量“候选是否相关”，不能直接当作最终答案置信度。
    # retrieval_top_score 是排序后的第一条候选分数：它代表最强证据的相关性，
    # 不是所有候选的平均分，也不是“最终进入 Prompt 的文档数量”。
    # 先统一到 [0, 1]，并保留原始分数供 Trace 诊断。
    normalized_score = normalize_retrieval_score(retrieval_top_score)
    decision_score = _clamp(intent_rule_score)
    rule_candidate_score = _clamp(intent_rule_candidate_score if intent_rule_candidate_score is not None else intent_rule_score)
    history_rewrite_used = _history_rewrite_used(
        query=query,
        raw_query=raw_query,
        rewritten_query=rewritten_query,
    )

    # 评分拆成两段，便于讲义和跟敲代码阅读：
    # 1. 先按回答路径计算基础分；
    # 2. 再按追问改写、低意图分、空上下文等风险统一扣分/封顶。
    score, reasons = _base_evidence_score(
        hit_type=hit_type,
        normalized_score=normalized_score,
        context_count=context_count,
        source_count=source_count,
        decision_score=decision_score,
        deterministic_route=deterministic_route,
        faq_exact_match=faq_exact_match,
    )
    score, reasons = _apply_evidence_risk_adjustments(
        score=score,
        reasons=reasons,
        hit_type=hit_type,
        context_count=context_count,
        decision_score=decision_score,
        deterministic_route=deterministic_route,
        history_rewrite_used=history_rewrite_used,
    )

    # 最后统一做边界收口和展示精度处理，确保 API、前端和 Trace 使用同一数值。
    final_score = round(_clamp(score), 2)
    return AnswerConfidence(
        score=final_score,
        level=confidence_level(final_score),
        reasons=reasons,
        signals={
            "hit_type": hit_type,
            "retrieval_top_score": retrieval_top_score,
            "normalized_retrieval_score": round(normalized_score, 2),
            "context_count": context_count,
            "source_count": source_count,
            "intent_rule_score": round(rule_candidate_score, 2),
            "intent_decision_score": round(decision_score, 2),
            "history_rewrite_used": history_rewrite_used,
            "faq_exact_match": faq_exact_match,
            "deterministic_route": deterministic_route,
        },
    )


def _base_evidence_score(
    *,
    hit_type: str,
    normalized_score: float,
    context_count: int,
    source_count: int,
    decision_score: float,
    deterministic_route: bool,
    faq_exact_match: bool,
) -> tuple[float, list[str]]:
    """按回答路径计算生成前基础分。

    参数：
        hit_type: 回答路径类型（rag / faq_direct / insufficient_context）。
        normalized_score: 归一化后的 top-1 检索分（[0,1]）。
        context_count: 最终放入 Prompt 的上下文片段数。
        source_count: 去重来源数量。
        decision_score: 意图决策分（[0,1]）。
        deterministic_route: 是否为确定性路由。
        faq_exact_match: FAQ 是否精确匹配标准问题。

    返回：
        tuple[float, list[str]]: (基础分, 本路径触发的原因列表)。

    调用顺序：RAG 管线 -> calculate_evidence_confidence() -> _base_evidence_score()。
    """
    # 原因：确定性路由（问候/越界/转人工）不依赖检索证据，直接以
    # 0.78 + 0.18*decision_score 的高基线起步，意图分只做小幅微调。
    if deterministic_route:
        return 0.78 + 0.18 * decision_score, ["deterministic_route"]
    # 原因：标准问题一字不差的精确匹配是最强证据，直接给 0.95 满档，
    # 不参与其他信号折扣，避免被无关信号拉低。
    if hit_type == "faq_direct" and faq_exact_match:
        return 0.95, ["faq_exact_match"]
    # 原因：FAQ 相似命中以检索相似度为主力（0.55 起步），意图分做辅助微调。
    if hit_type == "faq_direct":
        return 0.55 + 0.35 * normalized_score + 0.08 * decision_score, ["faq_score_direct"]
    # 原因：信息不足路径基线极低，弱召回只能轻微抬分，不可能越过中高置信区间。
    if hit_type == "insufficient_context":
        return 0.12 + 0.15 * normalized_score, ["insufficient_context"]
    # 原因：通用文档 RAG 中检索相关性权重最大（0.45），上下文和来源提供有限
    # 增益且各自封顶（min(…,4)/min(…,3)），防止堆砌候选把分数刷高。
    return (
        0.20
        + 0.45 * normalized_score
        + min(context_count, 4) * 0.06
        + min(source_count, 3) * 0.04
        + 0.12 * decision_score,
        ["rag_with_context" if context_count else "rag_without_context"],
    )


def _apply_evidence_risk_adjustments(
    *,
    score: float,
    reasons: list[str],
    hit_type: str,
    context_count: int,
    decision_score: float,
    deterministic_route: bool,
    history_rewrite_used: bool,
) -> tuple[float, list[str]]:
    """应用生成前风险扣分和安全封顶。

    参数：
        score: _base_evidence_score() 产出的基础分。
        reasons: 已有原因列表，扣分触发的条目会追加到其中。
        hit_type: 回答路径类型。
        context_count: 最终放入 Prompt 的上下文片段数。
        decision_score: 意图决策分。
        deterministic_route: 是否为确定性路由。
        history_rewrite_used: 是否依赖追问改写补全问题。

    返回：
        tuple[float, list[str]]: (扣分/封顶后的分数, 追加后的原因列表)。

    调用顺序：RAG 管线 -> calculate_evidence_confidence() -> _apply_evidence_risk_adjustments()。
    """
    adjusted_score = score
    adjusted_reasons = list(reasons)
    # 原因：追问改写说明原始提问信息不完整，扣 0.05~0.08；上下文越多扣越少，
    # 因为更多证据可以弥补改写误差。
    if history_rewrite_used:
        adjusted_score -= 0.05 if context_count >= 2 else 0.08
        adjusted_reasons.append("history_rewrite_used")
    # 原因：意图网关自身拿不准（<0.70）说明路由不稳定，扣 0.04；
    # 确定性路由已给高基线，不重复扣。
    if decision_score < 0.70 and not deterministic_route:
        adjusted_score -= 0.04
        adjusted_reasons.append("low_intent_decision_score")
    # 原因：RAG 路径但没有上下文入选时强制 <=0.35——有候选不等于有可用证据，
    # 这是兜底安全网，防止空上下文仍显示中高置信。
    if hit_type == "rag" and context_count == 0:
        adjusted_score = min(adjusted_score, 0.35)
        adjusted_reasons.append("no_selected_context")
    return adjusted_score, adjusted_reasons


def record_evidence_confidence(
    context: Any,
    *,
    hit_type: str,
    retrieval_top_score: float,
    context_count: int,
    source_count: int,
    deterministic_route: bool = False,
    faq_exact_match: bool = False,
) -> dict[str, Any]:
    """计算并写入生成前证据置信度，作为后续最终置信度合并的基础。

    参数：
        context: RAG 查询上下文，读取 query/raw_query/rewritten_query 与
            intent_payload（confidence、rule_score），并写入
            answer_confidence 和 retrieval_info。
        hit_type: 回答路径类型。
        retrieval_top_score: top-1 检索相关性分。
        context_count: 最终放入 Prompt 的上下文片段数。
        source_count: 去重来源数量。
        deterministic_route: 是否为确定性路由。
        faq_exact_match: FAQ 是否精确匹配标准问题。

    返回：
        dict: 写入后的置信度字典（含 evidence_confidence 摘要与
        generation_verification 占位）。

    调用顺序：RAG 管线 -> record_evidence_confidence() -> calculate_evidence_confidence()。
    """
    confidence = calculate_evidence_confidence(
        hit_type=hit_type,
        retrieval_top_score=retrieval_top_score,
        context_count=context_count,
        source_count=source_count,
        intent_rule_score=float(context.intent_payload["confidence"]) if context.intent_payload else 0.6,
        intent_rule_candidate_score=float(context.intent_payload["rule_score"]) if context.intent_payload else 0.6,
        query=context.query,
        raw_query=context.raw_query,
        rewritten_query=context.rewritten_query,
        deterministic_route=deterministic_route,
        faq_exact_match=faq_exact_match,
    ).as_dict()
    # 生成前先明确标记 pending：
    # RAG 主链路后续会在 LLM 输出完成后更新为 verified/partial/failed；
    # FAQ 直出、确定性回答和信息不足分支则会标记为 not_applicable。
    confidence["evidence_confidence"] = _confidence_summary(confidence["score"])
    confidence["generation_verification"] = {
        "status": "pending",
        "score": None,
        "reasons": [],
        "signals": {},
    }
    context.answer_confidence = confidence
    context.retrieval_info["answer_confidence"] = confidence
    return confidence


def record_answer_confidence(
    context: Any,
    *,
    hit_type: str,
    retrieval_top_score: float,
    context_count: int,
    source_count: int,
    deterministic_route: bool = False,
    faq_exact_match: bool = False,
) -> dict[str, Any]:
    """兼容旧名称：写入生成前证据置信度。新代码优先用 record_evidence_confidence()。

    参数：
        context: RAG 查询上下文。
        hit_type: 回答路径类型。
        retrieval_top_score: top-1 检索相关性分。
        context_count: 最终放入 Prompt 的上下文片段数。
        source_count: 去重来源数量。
        deterministic_route: 是否为确定性路由。
        faq_exact_match: FAQ 是否精确匹配标准问题。

    返回：
        dict: 同 record_evidence_confidence() 的写入结果。

    调用顺序：RAG 管线 -> record_answer_confidence() -> record_evidence_confidence()。
    """
    return record_evidence_confidence(
        context,
        hit_type=hit_type,
        retrieval_top_score=retrieval_top_score,
        context_count=context_count,
        source_count=source_count,
        deterministic_route=deterministic_route,
        faq_exact_match=faq_exact_match,
    )


def finalize_generated_answer_confidence(
    context: Any,
    *,
    answer: str,
    context_docs: list[Document],
) -> dict[str, Any]:
    """Stage 6：在 LLM 生成完成后核验答案，并更新最终答案置信度。

    生成前的 ``calculate_evidence_confidence()`` 只能判断“证据是否足以支撑生成”，
    无法知道模型是否真的使用了这些证据。因此 RAG 主流程在引用补强之后调用本函数，
    检查最终文本的几个可观测属性：

    1. 答案是否为空；
    2. 事实段落是否带有行内引用；
    3. 引用编号是否落在当前上下文文档范围内；
    4. 答案中的词组和关键数字是否能在上下文中找到词面支撑。

    这里使用的是低延迟、可解释的确定性核验，不声称完成了语义蕴含判断。
    最终分数采用保守合并：``min(evidence_confidence, generation_verification)``。
    原因是“证据很强但答案没有引用/明显偏离上下文”时，不能继续保留高置信度。

    参数：
        context: RAG 查询上下文，读取已写入的 answer_confidence，
            核验后更新 answer_confidence 与 retrieval_info。
        answer: LLM 生成并完成引用补强的最终答案文本。
        context_docs: 进入 Prompt 的上下文文档，用于引用合法性判断
            与词面支撑度计算。

    返回：
        dict: 合并后的最终置信度字典；尚未写入生成前置信度时返回空 dict。

    调用顺序：RAG 管线 -> enforce_answer_citations() ->
    finalize_generated_answer_confidence() -> finish_success()。
    """
    current = dict(context.answer_confidence or {})
    if not current:
        return {}

    verification = calculate_generation_confidence(answer=answer, context_docs=context_docs)
    finalized = combine_answer_confidence(current, verification)
    context.answer_confidence = finalized
    context.retrieval_info["answer_confidence"] = finalized
    return finalized


def combine_answer_confidence(
    evidence_confidence: dict[str, Any],
    generation_confidence: dict[str, Any],
) -> dict[str, Any]:
    """合并生成前证据评分和生成后答案核验，产出最终答案置信度。

    合并策略刻意保守：只要生成后核验给出了有效分数，最终分数就取
    ``min(evidence_score, generation_score)``。这样可以避免“检索很强，但 LLM
    没有引用、引用非法或答案明显偏离上下文”时仍显示高置信。

    参数：
        evidence_confidence: record_evidence_confidence() 写入的生成前
            置信度字典（score、reasons、signals）。
        generation_confidence: calculate_generation_confidence() 产出的
            核验结果；status 为 not_applicable 或 score 为 None 时
            退化为只采用证据分。

    返回：
        dict: 最终答案置信度，含合并分数、等级以及合并后的 signals/reasons，
        并保留 evidence_confidence / generation_verification 两个子结构。

    调用顺序：RAG 管线 -> finalize_generated_answer_confidence() -> combine_answer_confidence()。
    """
    evidence_score = _clamp(float(evidence_confidence.get("score") or 0.0))
    generation_score = generation_confidence.get("score")
    if generation_confidence.get("status") == "not_applicable" or generation_score is None:
        final_score = evidence_score
    else:
        final_score = min(evidence_score, _clamp(float(generation_score)))

    final_level = confidence_level(final_score)
    signals = dict(evidence_confidence.get("signals") or {})
    generation_signals = dict(generation_confidence.get("signals") or {})
    signals.update(
        {
            "evidence_confidence_score": round(evidence_score, 2),
            "generation_verification_score": (
                round(float(generation_score), 2) if generation_score is not None else None
            ),
            "generation_verification_status": generation_confidence.get("status"),
            **generation_signals,
        }
    )
    return {
        "score": round(_clamp(final_score), 2),
        "level": final_level,
        "label": LEVEL_LABELS[final_level],
        "reasons": _merge_reasons(
            evidence_confidence.get("reasons") or [],
            generation_confidence.get("reasons") or [],
        ),
        "signals": signals,
        "evidence_confidence": _confidence_summary(evidence_score),
        "generation_verification": generation_confidence,
    }


def mark_answer_confidence_not_applicable(context: Any, *, reason: str) -> dict[str, Any]:
    """标记没有经过 LLM 生成的答案分支，不虚构生成后核验结果。

    FAQ 直出、确定性路由和 ``insufficient_context`` 都没有经过最终 LLM 生成，
    这些分支应保留证据置信度，但把生成核验标记为 ``not_applicable``。

    参数：
        context: RAG 查询上下文，更新其 answer_confidence 与 retrieval_info。
        reason: 标记原因，写入 generation_verification.reasons 供 Trace 排查。

    返回：
        dict: 更新后的置信度字典；尚无生成前置信度时返回空 dict。

    调用顺序：RAG 管线 -> _finish_with_single_answer() ->
    mark_answer_confidence_not_applicable()。
    """
    current = dict(context.answer_confidence or {})
    if not current:
        return {}
    score = _clamp(float(current.get("score") or 0.0))
    current["evidence_confidence"] = _confidence_summary(score)
    current["generation_verification"] = {
        "status": "not_applicable",
        "score": None,
        "reasons": [reason],
        "signals": {"generation_attempted": False},
    }
    signals = dict(current.get("signals") or {})
    signals.update(
        {
            "evidence_confidence_score": round(score, 2),
            "generation_verification_score": None,
            "generation_verification_status": "not_applicable",
            "generation_attempted": False,
        }
    )
    current["signals"] = signals
    context.answer_confidence = current
    context.retrieval_info["answer_confidence"] = current
    return current


def calculate_generation_confidence(
    *,
    answer: str,
    context_docs: list[Document],
) -> dict[str, Any]:
    """对生成答案做低延迟的引用和词面支撑核验。

    该函数故意不把“出现了来源编号”直接等同于“事实已被证明”：
    - 只出现在“参考来源”尾部的编号不算行内引用；
    - 超出上下文范围的编号会被标记为非法；
    - 每个事实段落会计算与上下文的词面重合度；
    - 词面重合度只是启发式信号，不替代 NLI、LLM Judge 或人工复核。

    参数：
        answer: LLM 生成、完成引用补强后的答案文本。
        context_docs: 进入 Prompt 的上下文文档，文档数量即合法引用编号上限，
            也是词面支撑度计算的对照语料。

    返回：
        dict: 包含 score、status、reasons 和 signals 的可序列化核验结果。

    调用顺序：RAG 管线 -> finalize_generated_answer_confidence() -> calculate_generation_confidence()。
    """
    clean_answer = answer.strip()
    if not clean_answer:
        return _generation_confidence_result(
            score=0.0,
            status="failed",
            reasons=["answer_empty"],
            signals=_generation_signals(answer_char_count=0),
        )

    if not context_docs:
        return _generation_confidence_result(
            score=0.0,
            status="not_applicable",
            reasons=["no_context_for_generation_verification"],
            signals=_generation_signals(answer_char_count=len(clean_answer)),
        )

    answer_body = _answer_body_without_reference_section(clean_answer)
    claims = _extract_claim_units(answer_body)
    if not claims and answer_body:
        claims = [answer_body]

    evidence = _inspect_generated_claims(claims=claims, context_docs=context_docs)
    score = _score_generation_grounding(
        answer_body=answer_body,
        claim_count=evidence["claim_count"],
        citation_coverage=evidence["citation_coverage"],
        context_overlap=evidence["context_overlap"],
        invalid_citation_count=len(evidence["invalid_numbers"]),
    )
    score = round(_clamp(score), 2)

    status, reasons = _generation_status_and_reasons(
        score=score,
        citation_coverage=evidence["citation_coverage"],
        context_overlap=evidence["context_overlap"],
        invalid_citation_count=len(evidence["invalid_numbers"]),
    )
    return _generation_confidence_result(
        score=score,
        status=status,
        reasons=reasons,
        signals=_generation_signals(
            answer_char_count=len(clean_answer),
            claim_count=evidence["claim_count"],
            cited_claim_count=evidence["cited_claim_count"],
            citation_coverage=evidence["citation_coverage"],
            context_overlap=evidence["context_overlap"],
            valid_citation_numbers=sorted(evidence["valid_numbers"]),
            invalid_citation_numbers=sorted(evidence["invalid_numbers"]),
            inline_citation_count=len(_CITATION_RE.findall(answer_body)),
        ),
    )


def _generation_confidence_result(
    *,
    score: float,
    status: str,
    reasons: list[str],
    signals: dict[str, Any],
) -> dict[str, Any]:
    """构造生成后核验的统一返回结构。

    参数：
        score: 核验分（[0,1]）。
        status: 核验状态，取值 verified / partial / failed / not_applicable。
        reasons: 核验原因列表。
        signals: 供前端和 Trace 展示的信号字典。

    返回：
        dict: 统一四字段核验结构 {score, status, reasons, signals}。

    调用顺序：calculate_generation_confidence() -> _generation_confidence_result()。
    """
    return {
        "score": score,
        "status": status,
        "reasons": reasons,
        "signals": signals,
    }


def evaluate_generated_answer(
    *,
    answer: str,
    context_docs: list[Document],
) -> dict[str, Any]:
    """兼容旧名称：新代码优先调用 calculate_generation_confidence()。

    参数：
        answer: LLM 生成的答案文本。
        context_docs: 进入 Prompt 的上下文文档。

    返回：
        dict: 与 calculate_generation_confidence() 相同的核验结果。

    调用顺序：RAG 管线 -> evaluate_generated_answer() -> calculate_generation_confidence()。
    """
    return calculate_generation_confidence(answer=answer, context_docs=context_docs)


def faq_exact_match(query: str, doc: Document | None) -> bool:
    """判断 FAQ 命中是否为标准问题精确匹配。

    参数：
        query: 归一化后的用户问题。
        doc: FAQ 候选文档，从 metadata 读取 standard_question 作为比对基准。

    返回：
        bool: 查询与标准问题完全一致时为 True；doc 为 None 时为 False。

    调用顺序：RAG 管线 -> faq_exact_match()。
    """
    if doc is None:
        return False
    metadata = doc.metadata
    standard_question = str(metadata.get("standard_question") or metadata.get("question") or doc.page_content).strip()
    return query.strip() == standard_question


def normalize_retrieval_score(score: float) -> float:
    """把检索/重排分数压到 [0, 1]，仅用于置信度派生，不改变排序。

    Milvus/LangChain 返回值在本项目里按”越大越相关”使用；CrossEncoder 有时会返回
    大于 1 的 logits，因此这里用平滑压缩而不是直接截断（直接截断会丢失区分度：
    logit=5 和 logit=50 在截断后都是 1.0，但平滑压缩仍能区分出差异）。

    参数：
        score: 检索/重排后的原始相关性分，可大于 1。

    返回：
        float: [0,1] 区间的归一化分数；score <= 0 时返回 0.0。

    调用顺序：RAG 管线 -> normalize_retrieval_score()。
    """
    if score <= 0:
        return 0.0
    if score <= 1:
        return score
    return 1 - (1 / (1 + score))


def confidence_level(score: float) -> str:
    """将连续分数映射为前端展示等级。

    参数：
        score: [0,1] 区间的置信度分数。

    返回：
        str: "high"（>=0.82）/ "medium"（>=0.55）/ "low"（<0.55）。

    调用顺序：RAG 管线 -> confidence_level()。
    """
    if score >= 0.82:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _history_rewrite_used(*, query: str, raw_query: str, rewritten_query: str | None) -> bool:
    """判断当前回答是否依赖历史追问改写。

    参数：
        query: 当前用于业务处理的归一化问题。
        raw_query: 用户原始输入。
        rewritten_query: 改写后的问题，无改写时传 ``None``。

    返回：
        bool: 改写后问题既不同于当前问题也不同于原始问题时为 True，
        即本次回答确实借助了历史改写补全。

    调用顺序：RAG 管线 -> calculate_evidence_confidence() -> _history_rewrite_used()。
    """
    rewritten = (rewritten_query or "").strip()
    if not rewritten:
        return False
    return rewritten not in {query.strip(), raw_query.strip()}


def _clamp(value: float) -> float:
    """将数值限制在 [0, 1] 区间。

    参数：
        value: 任意数值。

    返回：
        float: 截断到 [0,1] 后的数值。

    调用顺序：RAG 管线 -> _clamp()。
    """
    return max(0.0, min(1.0, value))


def _confidence_summary(score: float) -> dict[str, Any]:
    """构造证据置信度摘要，避免前端重复解释同一个 score。

    参数：
        score: 置信度分数（允许越界，内部先 clamp 再取两位小数）。

    返回：
        dict: {score, level, label} 三元摘要，供 API 与 Trace 复用。

    调用顺序：RAG 管线 -> _confidence_summary()。
    """
    normalized = round(_clamp(score), 2)
    level = confidence_level(normalized)
    return {
        "score": normalized,
        "level": level,
        "label": LEVEL_LABELS[level],
    }


def _merge_reasons(*groups: list[str]) -> list[str]:
    """合并 reasons 并保持首次出现顺序。

    参数：
        *groups: 多组原因列表（如生成前原因 + 生成后原因）。

    返回：
        list[str]: 去重且保持首次出现顺序的原因列表，保证前端展示顺序稳定。

    调用顺序：combine_answer_confidence() -> _merge_reasons()。
    """
    merged: list[str] = []
    for group in groups:
        for reason in group:
            if reason not in merged:
                merged.append(reason)
    return merged


def _answer_body_without_reference_section(answer: str) -> str:
    """移除末尾自动补充的来源列表，只核验答案事实正文。

    原因：答案尾部常带"参考来源"清单，其中的编号不属于行内引用，
    若参与核验会把引用覆盖统计虚高，因此核验前先剔除。

    参数：
        answer: 完整答案文本。

    返回：
        str: 去掉来源清单后的正文；无来源清单时原样返回。

    调用顺序：calculate_generation_confidence() -> _answer_body_without_reference_section()。
    """
    match = _REFERENCE_SECTION_RE.search(answer)
    return answer[: match.start()].strip() if match else answer.strip()


def _inspect_generated_claims(
    *,
    claims: list[str],
    context_docs: list[Document],
) -> dict[str, Any]:
    """逐条核验事实单元的引用编号和上下文词面重合度。

    参数：
        claims: 从答案正文拆分出的事实单元列表。
        context_docs: 进入 Prompt 的上下文文档，文档数量即合法引用编号上限。

    返回：
        dict: 含 claim_count / cited_claim_count / citation_coverage /
        context_overlap / valid_numbers / invalid_numbers 的核验汇总。

    调用顺序：calculate_generation_confidence() -> _inspect_generated_claims()。
    """
    max_citation_number = len(context_docs)
    valid_numbers: set[int] = set()
    invalid_numbers: set[int] = set()
    cited_claim_count = 0
    overlap_values: list[float] = []
    for claim in claims:
        numbers = {int(value) for value in _CITATION_RE.findall(claim)}
        # 原因：引用编号与上下文文档一一对应（1..N），越界编号视为非法引用，
        # 是"模型编造来源编号"的可观测信号，供后续封顶扣分。
        valid = {number for number in numbers if 1 <= number <= max_citation_number}
        invalid = numbers - valid
        valid_numbers.update(valid)
        invalid_numbers.update(invalid)
        if valid:
            cited_claim_count += 1
        overlap_values.append(_context_overlap(claim, context_docs))

    claim_count = len(claims)
    return {
        "claim_count": claim_count,
        "cited_claim_count": cited_claim_count,
        "citation_coverage": cited_claim_count / claim_count if claim_count else 0.0,
        "context_overlap": sum(overlap_values) / len(overlap_values) if overlap_values else 0.0,
        "valid_numbers": valid_numbers,
        "invalid_numbers": invalid_numbers,
    }


def _score_generation_grounding(
    *,
    answer_body: str,
    claim_count: int,
    citation_coverage: float,
    context_overlap: float,
    invalid_citation_count: int,
) -> float:
    """按引用覆盖、上下文支撑和答案长度计算生成后核验分。

    参数：
        answer_body: 去掉来源清单后的答案正文。
        claim_count: 事实单元数量。
        citation_coverage: 带行内引用的事实单元占比（[0,1]）。
        context_overlap: 与上下文的词面支撑比例（[0,1]）。
        invalid_citation_count: 非法引用编号数量。

    返回：
        float: [0,1] 区间的生成后核验分。

    调用顺序：calculate_generation_confidence() -> _score_generation_grounding()。
    """
    # 原因：引用覆盖权重最高（0.55），词面支撑次之（0.40），长度只占 0.05——
    # 防止"写得多"成为高置信的理由；citation_signal 从 0.35 起步，
    # 完全无引用时仍反映"证据不足"而不是归零。
    answer_length_signal = min(len(answer_body) / 120.0, 1.0)
    citation_signal = 0.35 + 0.65 * citation_coverage
    support_signal = 0.25 + 0.75 * context_overlap
    score = 0.55 * citation_signal + 0.40 * support_signal + 0.05 * answer_length_signal
    if invalid_citation_count:
        # 非法引用说明模型可能编造了不存在的来源编号，按数量施加封顶扣分。
        score -= min(0.20, invalid_citation_count * 0.05)
    if claim_count > 1 and citation_coverage == 0:
        # 多段事实完全没有行内引用时，不能仅凭词面相似度判为高可信。
        score = min(score, 0.65)
    return score


def _generation_status_and_reasons(
    *,
    score: float,
    citation_coverage: float,
    context_overlap: float,
    invalid_citation_count: int,
) -> tuple[str, list[str]]:
    """把生成后核验分映射为状态和可解释原因。

    参数：
        score: 生成后核验分。
        citation_coverage: 带行内引用的事实单元占比。
        context_overlap: 与上下文的词面支撑比例。
        invalid_citation_count: 非法引用编号数量。

    返回：
        tuple[str, list[str]]: (核验状态, 原因列表)；状态取值
        verified / partial / failed。

    调用顺序：calculate_generation_confidence() -> _generation_status_and_reasons()。
    """
    reasons: list[str] = []
    # 原因：三个独立风险信号各自记录——引用覆盖不足、上下文支撑不足、
    # 非法引用，只要其一存在就不可能判定为 verified。
    if citation_coverage < 0.50:
        reasons.append("low_inline_citation_coverage")
    if context_overlap < 0.45:
        reasons.append("low_context_overlap")
    if invalid_citation_count:
        reasons.append("invalid_citation_reference")
    if score >= 0.82 and citation_coverage >= 0.50 and context_overlap >= 0.45 and invalid_citation_count == 0:
        reasons.append("generation_grounded")
        return "verified", reasons
    if score >= 0.55:
        return "partial", reasons
    return "failed", reasons


def _generation_signals(
    *,
    answer_char_count: int,
    claim_count: int = 0,
    cited_claim_count: int = 0,
    citation_coverage: float = 0.0,
    context_overlap: float = 0.0,
    valid_citation_numbers: list[int] | None = None,
    invalid_citation_numbers: list[int] | None = None,
    inline_citation_count: int = 0,
) -> dict[str, Any]:
    """构造前端和 Trace 展示用的生成后核验信号。

    参数：
        answer_char_count: 答案字符数。
        claim_count: 事实单元数量。
        cited_claim_count: 带行内引用的事实单元数量。
        citation_coverage: 引用覆盖（[0,1]）。
        context_overlap: 词面支撑比例（[0,1]）。
        valid_citation_numbers: 合法引用编号列表。
        invalid_citation_numbers: 非法引用编号列表。
        inline_citation_count: 行内引用出现次数。

    返回：
        dict: 序列化信号字典，reference_section_excluded 恒为 True，
        表示核验前已剔除来源清单。

    调用顺序：calculate_generation_confidence() -> _generation_signals()。
    """
    return {
        "answer_char_count": answer_char_count,
        "claim_count": claim_count,
        "cited_claim_count": cited_claim_count,
        "citation_coverage": round(citation_coverage, 2),
        "context_overlap": round(context_overlap, 2),
        "valid_citation_numbers": valid_citation_numbers or [],
        "invalid_citation_numbers": invalid_citation_numbers or [],
        "inline_citation_count": inline_citation_count,
        "reference_section_excluded": True,
    }


def _extract_claim_units(answer: str) -> list[str]:
    """按句号、分号和换行拆分事实单元，过滤标题和过短片段。

    参数：
        answer: 去掉来源清单后的答案正文。

    返回：
        list[str]: 事实单元列表；过短片段（<5 字）与孤立引用编号会被过滤。

    调用顺序：calculate_generation_confidence() -> _extract_claim_units()。
    """
    claims: list[str] = []
    for raw_unit in _CLAIM_SPLIT_RE.split(answer):
        # 原因：去掉"1. / - / •"等列表前缀，避免序号被误当作事实单元内容。
        unit = re.sub(r"^\s*(?:[-*•]|\d+[.)、])\s*", "", raw_unit).strip()
        if claims and _CITATION_RE.fullmatch(unit):
            # 原因：引用编号自成一段时（如换行后单独一行"[3]"），归并到上一个
            # 事实单元末尾，保证引用覆盖按"事实单元"而非"编号行"统计。
            claims[-1] = f"{claims[-1]} {unit}"
            continue
        plain_unit = _CITATION_RE.sub("", unit).strip()
        # 原因：短于 5 字的片段多为标题或残留标点，不构成可核验的事实单元。
        if len(plain_unit) >= 5:
            claims.append(unit)
    return claims


def _context_overlap(claim: str, context_docs: list[Document]) -> float:
    """计算答案事实单元与上下文的词面支撑比例。

    中文没有天然空格分词，因此连续中文片段按相邻双字组拆分；英文、数字和文件名
    按完整词片段处理。该比例只用于低成本风险提示，不代表语义蕴含概率。

    参数：
        claim: 单个事实单元（引用编号已剔除）。
        context_docs: 进入 Prompt 的上下文文档，拼接为对照语料。

    返回：
        float: [0,1] 词面支撑比例；claim 或上下文无 token 时返回 0.0。

    调用顺序：calculate_generation_confidence() -> _inspect_generated_claims() -> _context_overlap()。
    """
    claim_tokens = _text_tokens(_CITATION_RE.sub("", claim))
    if not claim_tokens:
        return 0.0
    context_text = "\n".join(doc.page_content for doc in context_docs)
    context_tokens = _text_tokens(context_text)
    if not context_tokens:
        return 0.0
    return len(claim_tokens & context_tokens) / len(claim_tokens)


def _text_tokens(text: str) -> set[str]:
    """提取可用于答案/上下文词面比较的中英文 token。

    参数：
        text: 待提取 token 的文本。

    返回：
        set[str]: token 集合；中文按相邻双字组切分（单字按原字保留），
        英文/数字/文件名按完整词片段小写保留。

    调用顺序：_context_overlap() -> _text_tokens()。
    """
    tokens: set[str] = set()
    for segment in _CJK_SEGMENT_RE.findall(text):
        if len(segment) == 1:
            tokens.add(segment)
            continue
        tokens.update(segment[index : index + 2] for index in range(len(segment) - 1))
    tokens.update(
        token.lower()
        for token in _WORD_RE.findall(text)
        if len(token) >= 2
    )
    return tokens
