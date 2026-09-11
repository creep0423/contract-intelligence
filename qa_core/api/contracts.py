"""合同工作台 HTTP/WebSocket API，所有资源先经过合同 DataScope 校验。"""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from qa_core.api.chat import _send_stream_events
from qa_core.api.dependencies import check_rate_limit, client_key, enforce_http_rate_limit, resolve_contract_access, resolve_websocket_contract_access
from qa_core.api.service_context import QueryServiceContext
from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings
from qa_core.contracts.errors import ContractNotFoundError
from qa_core.contracts.schemas import (
    ContractAccessContext,
    ContractDocumentType,
    ContractEventCreate,
    ContractEventUpdate,
    ContractStatusUpdate,
    ObligationStatusUpdate,
    RiskStatusUpdate,
)
from qa_core.contracts.service import get_contract_service
from qa_core.memory.history import get_history_store
from qa_core.pipeline.events import user_facing_error_message


router = APIRouter(prefix="/api/contracts", tags=["contracts"])
logger = get_logger(__name__)


async def _read_upload_limited(file: UploadFile, *, limit: int | None = None) -> bytes:
    """分块读取并在超过限制后立即终止，避免无界读入内存。"""
    limit = limit or get_settings().contract_max_upload_bytes
    content = bytearray()
    while chunk := await file.read(min(1024 * 1024, limit + 1)):
        content.extend(chunk)
        if len(content) > limit:
            raise ValueError(f"合同文件不能超过 {limit // (1024 * 1024)}MB")
    return bytes(content)


def _roles(value: str | None, access: ContractAccessContext) -> list[str]:
    roles = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return roles or access.user_roles


@router.post("")
async def upload_contract(
    file: UploadFile = File(...),
    contract_name: str | None = Form(default=None, max_length=255),
    contract_type: str | None = Form(default=None, max_length=64),
    document_type: ContractDocumentType = Form(default=ContractDocumentType.MASTER_CONTRACT),
    document_version: str = Form(default="1", max_length=64),
    allowed_roles: str | None = Form(default=None, max_length=2048),
    auto_analyze: bool = Form(default=True),
    access: ContractAccessContext = Depends(resolve_contract_access),
    _: None = Depends(enforce_http_rate_limit),
):
    content = await _read_upload_limited(file)
    return await asyncio.to_thread(
        get_contract_service().upload_and_process,
        file_name=file.filename or "contract",
        content=content,
        access=access,
        contract_name=contract_name,
        contract_type=contract_type,
        document_type=document_type,
        document_version=document_version,
        allowed_roles=_roles(allowed_roles, access),
        auto_analyze=auto_analyze,
        content_type=file.content_type,
    )


@router.post("/{contract_id}/documents")
async def add_contract_document(
    contract_id: str,
    file: UploadFile = File(...),
    document_type: ContractDocumentType = Form(default=ContractDocumentType.SUPPLEMENTARY_AGREEMENT),
    document_version: str = Form(default="1", max_length=64),
    reanalyze: bool = Form(default=True),
    access: ContractAccessContext = Depends(resolve_contract_access),
    _: None = Depends(enforce_http_rate_limit),
):
    return await asyncio.to_thread(
        get_contract_service().add_document,
        contract_id,
        file_name=file.filename or "contract",
        content=await _read_upload_limited(file),
        access=access,
        document_type=document_type,
        document_version=document_version,
        reanalyze=reanalyze,
        content_type=file.content_type,
    )


@router.get("")
def list_contracts(
    risk_level: str | None = None,
    status: str | None = None,
    page: int = 1,
    page_size: int = 50,
    access: ContractAccessContext = Depends(resolve_contract_access),
):
    return get_contract_service().list_contracts_page(
        access,
        risk_level=risk_level,
        status=status,
        page=page,
        page_size=page_size,
    )


@router.get("/{contract_id}")
def get_contract(contract_id: str, access: ContractAccessContext = Depends(resolve_contract_access)):
    return get_contract_service().get_detail(contract_id, access)


@router.patch("/{contract_id}")
def update_contract_status(
    contract_id: str,
    payload: ContractStatusUpdate,
    access: ContractAccessContext = Depends(resolve_contract_access),
):
    return get_contract_service().update_contract_status(contract_id, payload.status, access)


@router.post("/{contract_id}/analyze")
async def analyze_contract(
    contract_id: str,
    access: ContractAccessContext = Depends(resolve_contract_access),
    _: None = Depends(enforce_http_rate_limit),
):
    return await asyncio.to_thread(get_contract_service().analyze, contract_id, access)


@router.get("/{contract_id}/summary")
def get_contract_summary(contract_id: str, access: ContractAccessContext = Depends(resolve_contract_access)):
    detail = get_contract_service().get_detail(contract_id, access)
    analysis = detail.get("analysis") or {}
    return {
        "contract_id": contract_id,
        "summary": analysis.get("summary") or [],
        "analysis_status": analysis.get("status"),
        "stage_status": analysis.get("stage_status") or {},
        "stage_errors": analysis.get("stage_errors") or [],
        "is_degraded": bool(analysis.get("is_degraded")),
    }


@router.get("/{contract_id}/obligations")
def get_contract_obligations(
    contract_id: str,
    page: int = 1,
    page_size: int = 50,
    access: ContractAccessContext = Depends(resolve_contract_access),
):
    return get_contract_service().list_obligations_page(contract_id, access, page=page, page_size=page_size)


@router.patch("/{contract_id}/obligations/{obligation_id}")
def update_obligation(
    contract_id: str,
    obligation_id: str,
    payload: ObligationStatusUpdate,
    access: ContractAccessContext = Depends(resolve_contract_access),
):
    return get_contract_service().update_obligation_status(
        contract_id,
        obligation_id,
        status=payload.status,
        actual_completed_at=payload.actual_completed_at,
        note=payload.note,
        access=access,
    )


@router.get("/{contract_id}/risks")
def get_contract_risks(
    contract_id: str,
    risk_level: str | None = None,
    page: int = 1,
    page_size: int = 50,
    access: ContractAccessContext = Depends(resolve_contract_access),
):
    return get_contract_service().list_risks_page(
        contract_id,
        access,
        risk_level=risk_level,
        page=page,
        page_size=page_size,
    )


@router.get("/{contract_id}/analysis-runs")
def get_contract_analysis_runs(
    contract_id: str,
    page: int = 1,
    page_size: int = 50,
    access: ContractAccessContext = Depends(resolve_contract_access),
):
    return get_contract_service().list_analysis_runs_page(contract_id, access, page=page, page_size=page_size)


@router.patch("/{contract_id}/risks/{risk_id}")
def update_contract_risk(
    contract_id: str,
    risk_id: str,
    payload: RiskStatusUpdate,
    access: ContractAccessContext = Depends(resolve_contract_access),
):
    return get_contract_service().update_risk_status(contract_id, risk_id, payload.status, access)


@router.get("/{contract_id}/timeline")
def get_contract_timeline(contract_id: str, access: ContractAccessContext = Depends(resolve_contract_access)):
    detail = get_contract_service().get_detail(contract_id, access)
    return {"contract_id": contract_id, "timeline": detail["timeline"], "workday_calculation_note": detail["workday_calculation_note"]}


@router.post("/{contract_id}/events")
def create_contract_event(
    contract_id: str,
    payload: ContractEventCreate,
    access: ContractAccessContext = Depends(resolve_contract_access),
):
    return get_contract_service().add_event(contract_id, payload, access)


@router.patch("/{contract_id}/events/{event_id}")
def update_contract_event(
    contract_id: str,
    event_id: str,
    payload: ContractEventUpdate,
    access: ContractAccessContext = Depends(resolve_contract_access),
):
    return get_contract_service().update_event(contract_id, event_id, payload, access)


@router.get("/{contract_id}/history/{session_id}")
def get_contract_history(
    contract_id: str,
    session_id: str,
    access: ContractAccessContext = Depends(resolve_contract_access),
):
    get_contract_service().get_detail(contract_id, access)
    expected_prefix = f"contract:{contract_id}:"
    if not session_id.startswith(expected_prefix):
        raise ContractNotFoundError()
    return {"session_id": session_id, "history": get_history_store().as_pairs(session_id, limit=get_settings().max_history_messages)}


@router.websocket("/{contract_id}/stream")
async def stream_contract_qa(websocket: WebSocket, contract_id: str):
    await websocket.accept()
    try:
        access = resolve_websocket_contract_access(websocket)
    except Exception:
        await websocket.send_json({"type": "error", "error": "合同身份校验失败"})
        await websocket.close(code=1008)
        return
    try:
        contract = get_contract_service().store.get_contract(contract_id, access)
        if contract is None:
            await websocket.send_json({"type": "error", "error": "合同不存在或无权访问"})
            await websocket.close(code=1008)
            return
        while True:
            raw_payload = await websocket.receive_text()
            if not check_rate_limit(client_key(websocket)):
                await websocket.send_json({"type": "error", "error": "请求过于频繁，请稍后再试。"})
                continue
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "error": "无效的 JSON 数据"})
                continue
            query = str(payload.get("query") or "").strip()
            if not query:
                await websocket.send_json({"type": "error", "error": "查询内容不能为空"})
                continue
            if len(query) > 4000:
                await websocket.send_json({"type": "error", "error": "查询内容过长"})
                continue
            requested_session = str(payload.get("session_id") or "")
            session_id = (
                requested_session
                if requested_session.startswith(f"contract:{contract_id}:") and len(requested_session) <= 191
                else f"contract:{contract_id}:{uuid.uuid4()}"
            )
            conflict_response = get_contract_service().conflict_qa_response(contract_id, query, access)
            if conflict_response:
                await websocket.send_json({"type": "token", "token": conflict_response["answer"], "session_id": session_id})
                await websocket.send_json(
                    {
                        "type": "end",
                        "session_id": session_id,
                        "is_complete": True,
                        "hit_type": "contract_conflict",
                        "sources": conflict_response["sources"],
                    }
                )
                continue
            context = QueryServiceContext(
                query=query,
                source_filter="contract",
                session_id=session_id,
                kb_version=None,
                scenario_id=str(contract["scenario_id"]),
                tenant_id=str(contract["tenant_id"]),
                dataset_id=str(contract["dataset_id"]),
                visibility=str(contract["visibility"]),
                user_role=None,
                user_roles=list(access.user_roles),
            )
            if not await _send_stream_events(websocket, context):
                return
    except WebSocketDisconnect:
        logger.info("Contract WebSocket disconnected: contract_id=%s", contract_id)
    except Exception as exc:
        logger.exception("Contract WebSocket failed: contract_id=%s", contract_id)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "error": user_facing_error_message(exc)})
