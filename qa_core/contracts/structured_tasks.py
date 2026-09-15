"""合同 Structured Output 的统一可靠性边界。

只记录传输与结构元数据；模型原文、tool arguments 和业务字段值不得进入日志或遥测。
"""

from __future__ import annotations

import json
import re
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from qa_core.config.logging_config import get_logger
from qa_core.llm.client import get_chat_model


T = TypeVar("T", bound=BaseModel)
logger = get_logger(__name__)

PROVIDER_EMPTY_OUTPUT = "PROVIDER_EMPTY_OUTPUT"
TOOL_CALL_ARGUMENT_PARSE_FAILED = "TOOL_CALL_ARGUMENT_PARSE_FAILED"
TRAILING_STRUCTURAL_CORRUPTION = "TRAILING_STRUCTURAL_CORRUPTION"
TRUNCATED_JSON = "TRUNCATED_JSON"
INTERNAL_JSON_SYNTAX_ERROR = "INTERNAL_JSON_SYNTAX_ERROR"
MULTIPLE_JSON_VALUES = "MULTIPLE_JSON_VALUES"
WRONG_TOOL_SCHEMA = "WRONG_TOOL_SCHEMA"
MULTIPLE_AMBIGUOUS_TOOL_CALLS = "MULTIPLE_AMBIGUOUS_TOOL_CALLS"
PYDANTIC_VALIDATION_FAILED = "PYDANTIC_VALIDATION_FAILED"
UNKNOWN_STRUCTURED_OUTPUT_FAILURE = "UNKNOWN_STRUCTURED_OUTPUT_FAILURE"
PROVIDER_INVOCATION_ERROR = "PROVIDER_INVOCATION_ERROR"

_RETRYABLE_CATEGORIES = frozenset(
    {
        PROVIDER_EMPTY_OUTPUT,
        TOOL_CALL_ARGUMENT_PARSE_FAILED,
        TRUNCATED_JSON,
        INTERNAL_JSON_SYNTAX_ERROR,
    }
)


@dataclass(frozen=True)
class StructuredValidationIssue:
    """不包含模型原始值的 Pydantic 校验诊断。"""

    path: tuple[str, ...]
    error_type: str
    received_type: str


@dataclass(frozen=True)
class StructuredAttemptDiagnostic:
    """单次调用的安全结构元数据，不保存任何语义载荷。"""

    task_name: str
    schema_name: str
    attempt_number: int
    result: str
    error_category: str | None = None
    response_type: str = "unknown"
    parsed_present: bool = False
    parsing_error_type: str | None = None
    valid_tool_calls_count: int = 0
    invalid_tool_calls_count: int = 0
    tool_name: str | None = None
    args_type: str | None = None
    args_length: int | None = None
    json_error_type: str | None = None
    json_error_position: int | None = None
    complete_first_object: bool = False
    suffix_length: int = 0
    suffix_class: str = "NONE"
    validation_issues: tuple[StructuredValidationIssue, ...] = ()
    bounded_recovery_invoked: bool = False
    retry_trigger_category: str | None = None
    terminal: bool = False


@dataclass(frozen=True)
class StructuredTelemetrySnapshot:
    structured_calls: int
    total_attempts: int
    normal_successes: int
    first_attempt_successes: int
    bounded_recovery_invocations: int
    bounded_recoveries: int
    provider_parse_retries: int
    retry_successes: int
    unrecoverable_failures: int
    pydantic_failures: int


class StructuredOutputError(RuntimeError):
    """结构化输出边界的稳定、安全异常。"""

    def __init__(
        self,
        reason: str,
        *,
        validation_issues: tuple[StructuredValidationIssue, ...] = (),
        diagnostic: StructuredAttemptDiagnostic | None = None,
    ) -> None:
        self.reason = reason
        self.validation_issues = validation_issues
        self.diagnostic = diagnostic
        super().__init__(f"合同结构化输出解析失败（{reason}）")


class StructuredTaskRunner:
    """所有合同结构化任务共用的规范化、校验、恢复和固定重试边界。"""

    def __init__(self, *, telemetry_limit: int = 1000) -> None:
        self._telemetry: deque[StructuredAttemptDiagnostic] = deque(maxlen=telemetry_limit)
        self._telemetry_lock = threading.Lock()

    def invoke(
        self,
        schema: type[T],
        *,
        system_prompt: str,
        user_prompt: str,
        task_name: str | None = None,
    ) -> T:
        resolved_task_name = _safe_identifier(task_name or schema.__name__, fallback="UNSAFE_TASK_NAME")
        llm = get_chat_model(streaming=False)
        structured_llm = llm.with_structured_output(
            schema,
            method="function_calling",
            include_raw=True,
        )
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        for attempt_number in (1, 2):
            try:
                response = structured_llm.invoke(messages)
            except Exception as exc:
                self._record(
                    _diagnostic(
                        task_name=resolved_task_name,
                        schema=schema,
                        attempt_number=attempt_number,
                        result="FAILURE",
                        error_category=PROVIDER_INVOCATION_ERROR,
                        response_type="NO_RESPONSE",
                        parsing_error_type=type(exc).__name__,
                        terminal=True,
                    )
                )
                raise
            try:
                result, diagnostic = _normalize_response(
                    schema,
                    response,
                    task_name=resolved_task_name,
                    attempt_number=attempt_number,
                )
            except StructuredOutputError as exc:
                diagnostic = exc.diagnostic or _diagnostic(
                    task_name=resolved_task_name,
                    schema=schema,
                    attempt_number=attempt_number,
                    result="FAILURE",
                    error_category=exc.reason,
                    validation_issues=exc.validation_issues,
                )
                retryable = attempt_number == 1 and exc.reason in _RETRYABLE_CATEGORIES
                diagnostic = replace(
                    diagnostic,
                    retry_trigger_category=exc.reason if retryable else None,
                    terminal=not retryable,
                )
                self._record(diagnostic)
                if retryable:
                    continue
                raise StructuredOutputError(
                    exc.reason,
                    validation_issues=exc.validation_issues,
                    diagnostic=diagnostic,
                ) from None
            self._record(diagnostic)
            return result

        raise AssertionError("bounded structured-output retry loop exhausted unexpectedly")

    def telemetry_events(self) -> tuple[StructuredAttemptDiagnostic, ...]:
        with self._telemetry_lock:
            return tuple(self._telemetry)

    def telemetry_snapshot(self) -> StructuredTelemetrySnapshot:
        events = self.telemetry_events()
        return StructuredTelemetrySnapshot(
            structured_calls=sum(event.attempt_number == 1 for event in events),
            total_attempts=len(events),
            normal_successes=sum(event.result == "NORMAL_SUCCESS" for event in events),
            first_attempt_successes=sum(
                event.attempt_number == 1 and event.result in {"NORMAL_SUCCESS", "RECOVERED"}
                for event in events
            ),
            bounded_recovery_invocations=sum(
                event.bounded_recovery_invoked for event in events
            ),
            bounded_recoveries=sum(event.result == "RECOVERED" for event in events),
            provider_parse_retries=sum(event.retry_trigger_category is not None for event in events),
            retry_successes=sum(
                event.attempt_number == 2 and event.result in {"NORMAL_SUCCESS", "RECOVERED"}
                for event in events
            ),
            unrecoverable_failures=sum(event.result == "FAILURE" and event.terminal for event in events),
            pydantic_failures=sum(
                event.error_category == PYDANTIC_VALIDATION_FAILED for event in events
            ),
        )

    def clear_telemetry(self) -> None:
        with self._telemetry_lock:
            self._telemetry.clear()

    def _record(self, diagnostic: StructuredAttemptDiagnostic) -> None:
        with self._telemetry_lock:
            self._telemetry.append(diagnostic)
        logger.info(
            "Contract structured task: task_name=%s schema_name=%s attempt_number=%s "
            "result=%s error_category=%s response_type=%s parsed_present=%s "
            "parsing_error_type=%s valid_tool_calls_count=%s invalid_tool_calls_count=%s "
            "tool_name=%s args_type=%s args_length=%s json_error_type=%s "
            "json_error_position=%s complete_first_object=%s suffix_length=%s "
            "suffix_class=%s validation_issue_paths_types=%s "
            "bounded_recovery_invoked=%s retry_trigger_category=%s terminal=%s",
            diagnostic.task_name,
            diagnostic.schema_name,
            diagnostic.attempt_number,
            diagnostic.result,
            diagnostic.error_category,
            diagnostic.response_type,
            diagnostic.parsed_present,
            diagnostic.parsing_error_type,
            diagnostic.valid_tool_calls_count,
            diagnostic.invalid_tool_calls_count,
            diagnostic.tool_name,
            diagnostic.args_type,
            diagnostic.args_length,
            diagnostic.json_error_type,
            diagnostic.json_error_position,
            diagnostic.complete_first_object,
            diagnostic.suffix_length,
            diagnostic.suffix_class,
            tuple((issue.path, issue.error_type) for issue in diagnostic.validation_issues),
            diagnostic.bounded_recovery_invoked,
            diagnostic.retry_trigger_category,
            diagnostic.terminal,
        )


def _normalize_response(
    schema: type[T],
    response: Any,
    *,
    task_name: str,
    attempt_number: int,
) -> tuple[T, StructuredAttemptDiagnostic]:
    response_type = type(response).__name__
    if not isinstance(response, Mapping):
        try:
            result = response if isinstance(response, schema) else _validate(schema, response)
        except StructuredOutputError as exc:
            _raise_with_diagnostic(
                exc.reason,
                validation_issues=exc.validation_issues,
                task_name=task_name,
                schema=schema,
                attempt_number=attempt_number,
                response_type=response_type,
                parsed_present=response is not None,
            )
        return result, _diagnostic(
            task_name=task_name,
            schema=schema,
            attempt_number=attempt_number,
            result="NORMAL_SUCCESS",
            response_type=response_type,
            parsed_present=True,
        )

    parsed = response.get("parsed")
    parsing_error = response.get("parsing_error")
    parsing_error_type = type(parsing_error).__name__ if parsing_error is not None else None
    raw = response.get("raw")
    valid_calls = list(getattr(raw, "tool_calls", None) or [])
    invalid_calls = list(getattr(raw, "invalid_tool_calls", None) or [])
    base = dict(
        task_name=task_name,
        schema=schema,
        attempt_number=attempt_number,
        response_type=response_type,
        parsed_present=parsed is not None,
        parsing_error_type=parsing_error_type,
        valid_tool_calls_count=len(valid_calls),
        invalid_tool_calls_count=len(invalid_calls),
        tool_name=_single_safe_tool_name(schema, valid_calls, invalid_calls),
    )

    if len(valid_calls) + len(invalid_calls) > 1:
        _raise_with_diagnostic(MULTIPLE_AMBIGUOUS_TOOL_CALLS, **base)

    single_call = (valid_calls + invalid_calls)[0] if valid_calls or invalid_calls else None
    if single_call is not None and _call_value(single_call, "name") != schema.__name__:
        _raise_with_diagnostic(WRONG_TOOL_SCHEMA, **base)

    if parsed is not None:
        try:
            result = parsed if isinstance(parsed, schema) else _validate(schema, parsed)
        except StructuredOutputError as exc:
            _raise_with_diagnostic(exc.reason, validation_issues=exc.validation_issues, **base)
        return result, _diagnostic(result="NORMAL_SUCCESS", **base)

    if len(invalid_calls) == 1:
        return _normalize_invalid_call(schema, invalid_calls[0], base)

    if len(valid_calls) == 1:
        call = valid_calls[0]
        tool_name = _call_value(call, "name")
        args = _call_value(call, "args")
        call_base = {
            **base,
            "tool_name": schema.__name__ if tool_name == schema.__name__ else "UNEXPECTED_TOOL",
            "args_type": type(args).__name__,
        }
        if tool_name != schema.__name__:
            _raise_with_diagnostic(WRONG_TOOL_SCHEMA, **call_base)
        if not isinstance(args, Mapping):
            _raise_with_diagnostic(UNKNOWN_STRUCTURED_OUTPUT_FAILURE, **call_base)
        try:
            result = _validate(schema, args)
        except StructuredOutputError as exc:
            _raise_with_diagnostic(
                exc.reason,
                validation_issues=exc.validation_issues,
                **call_base,
            )
        return result, _diagnostic(result="NORMAL_SUCCESS", **call_base)

    if isinstance(parsing_error, ValidationError):
        issues = _validation_issues(parsing_error)
        _raise_with_diagnostic(PYDANTIC_VALIDATION_FAILED, validation_issues=issues, **base)
    if raw is None or not getattr(raw, "content", None):
        _raise_with_diagnostic(PROVIDER_EMPTY_OUTPUT, **base)
    _raise_with_diagnostic(UNKNOWN_STRUCTURED_OUTPUT_FAILURE, **base)


def _normalize_invalid_call(
    schema: type[T],
    invalid_call: Any,
    base: dict[str, Any],
) -> tuple[T, StructuredAttemptDiagnostic]:
    tool_name = _call_value(invalid_call, "name")
    arguments = _call_value(invalid_call, "args")
    details = {
        **base,
        "tool_name": schema.__name__ if tool_name == schema.__name__ else "UNEXPECTED_TOOL",
        "args_type": type(arguments).__name__,
        "args_length": len(arguments) if isinstance(arguments, str) else None,
        "bounded_recovery_invoked": False,
    }
    if tool_name != schema.__name__:
        _raise_with_diagnostic(WRONG_TOOL_SCHEMA, **details)
    if not isinstance(arguments, str):
        _raise_with_diagnostic(TOOL_CALL_ARGUMENT_PARSE_FAILED, **details)
    details["bounded_recovery_invoked"] = True
    candidate = arguments.lstrip()
    if not candidate:
        _raise_with_diagnostic(PROVIDER_EMPTY_OUTPUT, **details)

    try:
        payload, end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError as exc:
        category = TRUNCATED_JSON if _looks_truncated(exc, candidate) else INTERNAL_JSON_SYNTAX_ERROR
        _raise_with_diagnostic(
            category,
            json_error_type=type(exc).__name__,
            json_error_position=exc.pos,
            **details,
        )

    suffix = candidate[end:]
    compact_suffix = "".join(character for character in suffix if not character.isspace())
    details.update(
        complete_first_object=True,
        suffix_length=len(compact_suffix),
        suffix_class=_classify_suffix(compact_suffix),
    )
    if not isinstance(payload, Mapping):
        _raise_with_diagnostic(PYDANTIC_VALIDATION_FAILED, **details)
    if not compact_suffix:
        _raise_with_diagnostic(TOOL_CALL_ARGUMENT_PARSE_FAILED, **details)
    if any(character != "}" for character in compact_suffix):
        category = MULTIPLE_JSON_VALUES if _contains_second_json_value(compact_suffix) else TOOL_CALL_ARGUMENT_PARSE_FAILED
        _raise_with_diagnostic(category, **details)

    try:
        result = _validate(schema, payload)
    except StructuredOutputError as exc:
        _raise_with_diagnostic(
            exc.reason,
            validation_issues=exc.validation_issues,
            **details,
        )
    return result, _diagnostic(
        result="RECOVERED",
        error_category=TRAILING_STRUCTURAL_CORRUPTION,
        **details,
    )


def _validate(schema: type[T], payload: Any) -> T:
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        issues = _validation_issues(exc)
        raise StructuredOutputError(PYDANTIC_VALIDATION_FAILED, validation_issues=issues) from None


def _validation_issues(exc: ValidationError) -> tuple[StructuredValidationIssue, ...]:
    return tuple(
        StructuredValidationIssue(
            path=tuple(str(part) for part in error.get("loc", ())),
            error_type=str(error.get("type", "validation_error")),
            received_type=type(error.get("input")).__name__,
        )
        for error in exc.errors(include_url=False)
    )


def _raise_with_diagnostic(
    reason: str,
    *,
    validation_issues: tuple[StructuredValidationIssue, ...] = (),
    **diagnostic_fields: Any,
) -> None:
    diagnostic = _diagnostic(
        result="FAILURE",
        error_category=reason,
        validation_issues=validation_issues,
        **diagnostic_fields,
    )
    raise StructuredOutputError(
        reason,
        validation_issues=validation_issues,
        diagnostic=diagnostic,
    ) from None


def _diagnostic(
    *,
    task_name: str,
    schema: type[BaseModel],
    attempt_number: int,
    result: str,
    error_category: str | None = None,
    **fields: Any,
) -> StructuredAttemptDiagnostic:
    return StructuredAttemptDiagnostic(
        task_name=task_name,
        schema_name=schema.__name__,
        attempt_number=attempt_number,
        result=result,
        error_category=error_category,
        **fields,
    )


def _call_value(call: Any, key: str) -> Any:
    return call.get(key) if isinstance(call, Mapping) else getattr(call, key, None)


def _safe_identifier(value: str, *, fallback: str) -> str:
    return value if len(value) <= 128 and re.fullmatch(r"[A-Za-z0-9_.:-]+", value) else fallback


def _single_safe_tool_name(
    schema: type[BaseModel],
    valid_calls: list[Any],
    invalid_calls: list[Any],
) -> str | None:
    calls = valid_calls + invalid_calls
    if len(calls) != 1:
        return None
    value = _call_value(calls[0], "name")
    return schema.__name__ if value == schema.__name__ else "UNEXPECTED_TOOL"


def _looks_truncated(exc: json.JSONDecodeError, candidate: str) -> bool:
    return exc.pos >= max(0, len(candidate) - 2) or exc.msg.startswith("Unterminated string")


def _contains_second_json_value(suffix: str) -> bool:
    try:
        json.JSONDecoder().raw_decode(suffix)
    except (TypeError, ValueError):
        return False
    return True


def _classify_suffix(suffix: str) -> str:
    if not suffix:
        return "NONE"
    if all(character == "}" for character in suffix):
        return "REDUNDANT_CLOSING_BRACES"
    if _contains_second_json_value(suffix):
        return "MULTIPLE_JSON_VALUES"
    return "TEXT_OR_INVALID"
