"""合同 Structured Output 调用适配器，统一复用非流式 LLM 客户端。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from qa_core.llm.client import get_chat_model


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class StructuredValidationIssue:
    """不包含模型原始值的 Pydantic 校验诊断。"""

    path: tuple[str, ...]
    error_type: str
    received_type: str


class StructuredOutputError(RuntimeError):
    """结构化输出边界的稳定、安全异常。"""

    def __init__(
        self,
        reason: str,
        *,
        validation_issues: tuple[StructuredValidationIssue, ...] = (),
    ) -> None:
        self.reason = reason
        self.validation_issues = validation_issues
        super().__init__(f"合同结构化输出解析失败（{reason}）")


class StructuredTaskRunner:
    def invoke(self, schema: type[T], *, system_prompt: str, user_prompt: str) -> T:
        llm = get_chat_model(streaming=False)
        structured_llm = llm.with_structured_output(schema, method="function_calling", include_raw=True)
        response = structured_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        parsed = response.get("parsed") if isinstance(response, dict) else response
        parsing_error = response.get("parsing_error") if isinstance(response, dict) else None
        if parsing_error is not None:
            # LangChain parser 异常可能携带模型原始输出，其中可能包含合同正文。
            raise StructuredOutputError("LANGCHAIN_PARSING_ERROR") from None
        if parsed is not None:
            return parsed if isinstance(parsed, schema) else _validate(schema, parsed)

        recovered = _recover_single_trailing_brace(schema, response)
        if recovered is not None:
            return recovered

        raw = response.get("raw") if isinstance(response, dict) else None
        if getattr(raw, "invalid_tool_calls", None):
            raise StructuredOutputError("TOOL_CALL_ARGUMENT_PARSE_FAILED") from None
        raise StructuredOutputError("PARSED_VALUE_MISSING") from None


def _validate(schema: type[T], payload: Any) -> T:
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        issues = tuple(
            StructuredValidationIssue(
                path=tuple(str(part) for part in error.get("loc", ())),
                error_type=str(error.get("type", "validation_error")),
                received_type=type(error.get("input")).__name__,
            )
            for error in exc.errors(include_url=False)
        )
        raise StructuredOutputError(
            "PYDANTIC_VALIDATION_FAILED",
            validation_issues=issues,
        ) from None


def _recover_single_trailing_brace(schema: type[T], response: Any) -> T | None:
    """兼容 qwen-plus 偶发在完整 tool arguments 后多返回一个 ``}``。

    这里只接受一个可独立解码、且唯一剩余字符恰为右花括号的对象。其他截断、
    拼接、多余文本或一般 JSON 错误继续失败，避免把兼容处理扩展成模糊修复器。
    """

    if not isinstance(response, Mapping):
        return None
    raw = response.get("raw")
    if getattr(raw, "tool_calls", None):
        return None
    invalid_calls = list(getattr(raw, "invalid_tool_calls", None) or [])
    if len(invalid_calls) != 1 or not isinstance(invalid_calls[0], Mapping):
        return None
    invalid_call = invalid_calls[0]
    if invalid_call.get("name") != schema.__name__:
        return None
    arguments = invalid_call.get("args")
    if not isinstance(arguments, str):
        return None

    candidate = arguments.strip()
    try:
        payload, end = json.JSONDecoder().raw_decode(candidate)
    except (TypeError, ValueError):
        return None
    if candidate[end:] != "}" or not isinstance(payload, Mapping):
        return None
    return _validate(schema, payload)
