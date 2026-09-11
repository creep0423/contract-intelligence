"""合同 Structured Output 调用适配器，统一复用非流式 LLM 客户端。"""

from __future__ import annotations

from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from qa_core.llm.client import get_chat_model


T = TypeVar("T", bound=BaseModel)


class StructuredTaskRunner:
    def invoke(self, schema: type[T], *, system_prompt: str, user_prompt: str) -> T:
        llm = get_chat_model(streaming=False)
        structured_llm = llm.with_structured_output(schema, method="function_calling", include_raw=True)
        response = structured_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        parsed = response.get("parsed") if isinstance(response, dict) else response
        parsing_error = response.get("parsing_error") if isinstance(response, dict) else None
        if parsing_error or parsed is None:
            # 上游 ValidationError 可能携带模型原始输出，其中可能包含合同正文。
            # 对外只传播稳定错误类型，阶段日志通过 contract_id/run_id 排查。
            raise RuntimeError("合同结构化输出解析失败") from None
        return parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
