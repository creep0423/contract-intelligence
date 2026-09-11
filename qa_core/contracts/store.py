"""合同控制面 MySQL Store，沿用项目 SQLAlchemy 同步访问与显式 DDL 风格。"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

from sqlalchemy import text

from qa_core.common import utc_now
from qa_core.config.settings import get_settings
from qa_core.contracts.errors import ContractConflictError, ContractNotFoundError
from qa_core.contracts.risk import overall_risk_level, risk_fingerprint
from qa_core.contracts.schemas import (
    AnalysisStatus,
    ContractAccessContext,
    ContractAnalysisResult,
    ContractStatus,
    ProcessingStatus,
    RiskSource,
)
from qa_core.memory.base import _MySqlStore


CONTRACTS_TABLE = "contracts"
CONTRACT_DOCUMENTS_TABLE = "contract_documents"
CONTRACT_ANALYSES_TABLE = "contract_analyses"
CONTRACT_OBLIGATIONS_TABLE = "contract_obligations"
CONTRACT_RISKS_TABLE = "contract_risks"
CONTRACT_EVENTS_TABLE = "contract_events"


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _json_load(value: Any, default: Any) -> Any:
    try:
        return json.loads(str(value or ""))
    except (json.JSONDecodeError, TypeError):
        return default


def _versioned_obligation_id(analysis_id: str, business_id: str | None, index: int) -> str:
    """义务主键属于分析版本；business_id 仍用于跨版本风险指纹。"""
    stable_part = str(business_id or index)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"contract-analysis:{analysis_id}:obligation:{stable_part}"))


def _row_payload(row) -> dict[str, Any]:
    payload = dict(row)
    list_json = {"allowed_roles_json", "chunk_ids_json", "summary_json"}
    for key in (
        "allowed_roles_json", "chunk_ids_json", "extraction_json", "summary_json",
        "time_rule_json", "evidence_json",
    ):
        if key in payload:
            payload[key.removesuffix("_json")] = _json_load(payload.pop(key), [] if key in list_json else {})
    for key in (
        "created_at", "updated_at", "started_at", "finished_at", "actual_completed_at",
        "event_date",
    ):
        if key in payload and payload[key] is not None:
            payload[key] = str(payload[key])
    if "confidence" in payload and payload["confidence"] is not None:
        payload["confidence"] = float(payload["confidence"])
    if "needs_review" in payload:
        payload["needs_review"] = bool(payload["needs_review"])
    if "risk_type" in payload and isinstance(payload.get("evidence"), dict) and "primary" in payload["evidence"]:
        envelope = payload["evidence"]
        payload["evidence"] = envelope.get("primary") or {}
        payload["supporting_evidence"] = envelope.get("supporting") or []
    return payload


def _public_contract_payload(row) -> dict[str, Any]:
    payload = _row_payload(row)
    if payload.get("error_message"):
        payload["error_message"] = "合同处理失败，请查看服务日志"
    return payload


ANALYSIS_STAGE_ORDER = ("extraction", "obligations", "deterministic_risk", "semantic_risk", "summary", "persistence")
OPTIONAL_ANALYSIS_STAGES = {"semantic_risk", "summary"}


def _analysis_payload(row) -> dict[str, Any]:
    """为 API 提供明确阶段语义，不改变已有分析表结构。"""
    payload = _row_payload(row)
    raw_error = str(payload.get("error_summary") or "")
    stage_errors: list[dict[str, str]] = []
    for item in raw_error.split(","):
        if ":" not in item:
            continue
        stage, error_type = item.split(":", 1)
        if stage in ANALYSIS_STAGE_ORDER and re.fullmatch(r"[A-Za-z0-9_]+", error_type):
            stage_errors.append({"stage": stage, "error_type": error_type})
    failed_stages = {item["stage"] for item in stage_errors}
    status = str(payload.get("status") or "")
    stage_status: dict[str, str] = {}
    if status == "succeeded":
        stage_status = {stage: ("failed" if stage in failed_stages else "succeeded") for stage in ANALYSIS_STAGE_ORDER}
    elif status == "failed":
        failed_stage = stage_errors[0]["stage"] if stage_errors else None
        reached_failure = False
        for stage in ANALYSIS_STAGE_ORDER:
            if stage == failed_stage:
                stage_status[stage] = "failed"
                reached_failure = True
            else:
                stage_status[stage] = "not_started" if reached_failure else "succeeded"
    else:
        stage_status = {stage: "running" if stage == "extraction" else "not_started" for stage in ANALYSIS_STAGE_ORDER}
    payload["stage_status"] = stage_status
    payload["stage_errors"] = stage_errors
    payload["is_degraded"] = bool(status == "succeeded" and failed_stages.intersection(OPTIONAL_ANALYSIS_STAGES))
    return payload


def can_access_contract(contract: dict[str, Any], access: ContractAccessContext) -> bool:
    """统一校验合同控制面权限；向量检索还会使用同一 tenant/dataset/role 范围。"""
    if str(contract.get("tenant_id") or "") != access.tenant_id:
        return False
    visibility = str(contract.get("visibility") or "public")
    if "admin" not in access.user_roles and visibility != "public" and access.visibility != visibility:
        return False
    allowed = set(contract.get("allowed_roles") or [])
    return "admin" in access.user_roles or not allowed or bool(allowed.intersection(access.user_roles))


def _scope_filter(access: ContractAccessContext, *, alias: str = "") -> tuple[list[str], dict[str, Any]]:
    """生成合同控制面 SQL scope；不先读取资源再在 Python 中做权限判断。"""
    prefix = f"{alias}." if alias else ""
    clauses = [f"{prefix}tenant_id=:scope_tenant_id"]
    params: dict[str, Any] = {"scope_tenant_id": access.tenant_id}
    if "admin" in access.user_roles:
        return clauses, params
    clauses.append(f"({prefix}visibility='public' OR {prefix}visibility=:scope_visibility)")
    params["scope_visibility"] = access.visibility
    role_checks: list[str] = []
    for index, role in enumerate(access.user_roles):
        key = f"scope_role_{index}"
        role_checks.append(f"JSON_CONTAINS({prefix}allowed_roles_json, :{key})")
        params[key] = _json_dump(role)
    clauses.append(f"(JSON_LENGTH({prefix}allowed_roles_json)=0 OR {' OR '.join(role_checks)})")
    return clauses, params


class ContractStore(_MySqlStore):
    def create_contract(
        self,
        *,
        contract_id: str,
        name: str,
        contract_type: str | None,
        scenario_id: str,
        dataset_id: str,
        access: ContractAccessContext,
        allowed_roles: list[str],
    ) -> dict[str, Any]:
        now = utc_now()
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""INSERT INTO {CONTRACTS_TABLE}
                    (id, tenant_id, owner_user_id, contract_name, contract_type, status, processing_status,
                     overall_risk_level, scenario_id, dataset_id, visibility, allowed_roles_json,
                     created_at, updated_at)
                    VALUES (:id, :tenant_id, :owner_user_id, :contract_name, :contract_type, :status,
                            :processing_status, 'low', :scenario_id, :dataset_id, :visibility,
                            :allowed_roles_json, :created_at, :updated_at)"""
                ),
                {
                    "id": contract_id,
                    "tenant_id": access.tenant_id,
                    "owner_user_id": access.user_id,
                    "contract_name": name,
                    "contract_type": contract_type,
                    "status": ContractStatus.ACTIVE.value,
                    "processing_status": ProcessingStatus.UPLOADED.value,
                    "scenario_id": scenario_id,
                    "dataset_id": dataset_id,
                    "visibility": access.visibility,
                    "allowed_roles_json": _json_dump(allowed_roles),
                    "created_at": now,
                    "updated_at": now,
                },
            )
        return self.get_contract(contract_id, access) or {}

    def create_document(
        self,
        *,
        document_id: str,
        contract_id: str,
        file_name: str,
        file_path: str,
        file_type: str,
        parser_backend: str,
        document_type: str,
        document_version: str,
    ) -> None:
        now = utc_now()
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""INSERT INTO {CONTRACT_DOCUMENTS_TABLE}
                    (id, contract_id, file_name, file_path, file_type, parser_backend, parse_status,
                     document_type, document_version, page_count, chunk_count, chunk_ids_json,
                     created_at, updated_at)
                    VALUES (:id, :contract_id, :file_name, :file_path, :file_type, :parser_backend,
                            'uploaded', :document_type, :document_version, 0, 0, '[]', :created_at, :updated_at)"""
                ),
                {
                    "id": document_id,
                    "contract_id": contract_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "file_type": file_type,
                    "parser_backend": parser_backend,
                    "document_type": document_type,
                    "document_version": document_version,
                    "created_at": now,
                    "updated_at": now,
                },
            )

    def update_document_parsed(self, document_id: str, *, doc_id: str, page_count: int, chunk_ids: list[str]) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""UPDATE {CONTRACT_DOCUMENTS_TABLE}
                    SET doc_id=:doc_id, parse_status='parsed', page_count=:page_count,
                        chunk_count=:chunk_count, chunk_ids_json=:chunk_ids_json, updated_at=:updated_at
                    WHERE id=:id"""
                ),
                {
                    "id": document_id,
                    "doc_id": doc_id,
                    "page_count": page_count,
                    "chunk_count": len(chunk_ids),
                    "chunk_ids_json": _json_dump(chunk_ids),
                    "updated_at": utc_now(),
                },
            )

    def update_document_failed(self, document_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(f"UPDATE {CONTRACT_DOCUMENTS_TABLE} SET parse_status='failed', updated_at=:updated_at WHERE id=:id"),
                {"id": document_id, "updated_at": utc_now()},
            )

    def update_processing_status(
        self,
        contract_id: str,
        status: ProcessingStatus,
        *,
        error_message: str = "",
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""UPDATE {CONTRACTS_TABLE}
                    SET processing_status=:status, error_message=:error, updated_at=:updated_at WHERE id=:id"""
                ),
                {"id": contract_id, "status": status.value, "error": error_message[:1000], "updated_at": utc_now()},
            )

    def update_contract_facts(self, contract_id: str, result: ContractAnalysisResult) -> None:
        basic = result.extraction.basic_info
        updates = {
            "contract_name": basic.contract_name.extracted_value,
            "contract_number": basic.contract_number.extracted_value,
            "contract_type": basic.contract_type.extracted_value,
            "overall_risk_level": overall_risk_level(result.risks),
        }
        fields = ["overall_risk_level=:overall_risk_level", "processing_status='ready'", "error_message=''", "updated_at=:updated_at"]
        params = {"id": contract_id, "updated_at": utc_now(), **updates}
        for field in ("contract_name", "contract_number", "contract_type"):
            if updates[field]:
                fields.append(f"{field}=:{field}")
        with self.engine.begin() as conn:
            conn.execute(text(f"UPDATE {CONTRACTS_TABLE} SET {', '.join(fields)} WHERE id=:id"), params)

    def get_contract(self, contract_id: str, access: ContractAccessContext) -> dict[str, Any] | None:
        scope, params = _scope_filter(access)
        with self.engine.begin() as conn:
            row = conn.execute(
                text(f"SELECT * FROM {CONTRACTS_TABLE} WHERE id=:id AND {' AND '.join(scope)}"),
                {"id": contract_id, **params},
            ).mappings().first()
        return _public_contract_payload(row) if row else None

    def list_contracts(
        self,
        access: ContractAccessContext,
        *,
        risk_level: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        filters, params = _scope_filter(access)
        params.update({"limit": max(1, min(limit, 200)), "offset": max(0, offset)})
        if risk_level:
            filters.append("overall_risk_level=:risk_level")
            params["risk_level"] = risk_level
        if status:
            filters.append("status=:status")
            params["status"] = status
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(f"SELECT * FROM {CONTRACTS_TABLE} WHERE {' AND '.join(filters)} ORDER BY updated_at DESC LIMIT :limit OFFSET :offset"),
                params,
            ).mappings().all()
        return [_public_contract_payload(row) for row in rows]

    def count_contracts(self, access: ContractAccessContext, *, risk_level: str | None = None, status: str | None = None) -> int:
        filters, params = _scope_filter(access)
        if risk_level:
            filters.append("overall_risk_level=:risk_level")
            params["risk_level"] = risk_level
        if status:
            filters.append("status=:status")
            params["status"] = status
        with self.engine.begin() as conn:
            return int(conn.execute(text(f"SELECT COUNT(*) FROM {CONTRACTS_TABLE} WHERE {' AND '.join(filters)}"), params).scalar_one())

    def list_documents(self, contract_id: str) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(f"SELECT * FROM {CONTRACT_DOCUMENTS_TABLE} WHERE contract_id=:id ORDER BY created_at"),
                {"id": contract_id},
            ).mappings().all()
        documents = [_row_payload(row) for row in rows]
        for document in documents:
            document.pop("file_path", None)
        return documents

    def create_analysis_run(self, contract_id: str, *, model_name: str) -> str:
        analysis_id = str(uuid.uuid4())
        now = utc_now()
        with self.engine.begin() as conn:
            locked = conn.execute(
                text(f"SELECT id FROM {CONTRACTS_TABLE} WHERE id=:id FOR UPDATE"),
                {"id": contract_id},
            ).first()
            if not locked:
                raise ContractNotFoundError()
            running_rows = conn.execute(
                text(f"SELECT id, started_at FROM {CONTRACT_ANALYSES_TABLE} WHERE contract_id=:id AND status='running'"),
                {"id": contract_id},
            ).mappings().all()
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=get_settings().contract_analysis_stale_seconds)
            active_running: list[str] = []
            for running in running_rows:
                try:
                    started = datetime.fromisoformat(str(running.get("started_at") or "").replace("Z", "+00:00"))
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=timezone.utc)
                    is_stale = started.astimezone(timezone.utc) <= cutoff
                except ValueError:
                    is_stale = True
                if not is_stale:
                    active_running.append(str(running["id"]))
                    continue
                conn.execute(
                    text(f"UPDATE {CONTRACT_ANALYSES_TABLE} SET status='failed', error_summary='stale_recovery:TimeoutError', finished_at=:finished WHERE id=:id AND status='running'"),
                    {"id": running["id"], "finished": now},
                )
            if active_running:
                raise ContractConflictError("该合同已有正在运行的分析，请勿重复提交")
            conn.execute(
                text(
                    f"""INSERT INTO {CONTRACT_ANALYSES_TABLE}
                    (id, contract_id, status, analysis_version, model_name, extraction_json,
                     summary_json, started_at, created_at)
                    VALUES (:id, :contract_id, :status, 'contract-analysis-v2', :model_name,
                            '{{}}', '[]', :started_at, :created_at)"""
                ),
                {"id": analysis_id, "contract_id": contract_id, "status": AnalysisStatus.RUNNING.value, "model_name": model_name, "started_at": now, "created_at": now},
            )
            conn.execute(
                text(f"UPDATE {CONTRACTS_TABLE} SET processing_status='analyzing', error_message='', updated_at=:updated WHERE id=:id"),
                {"id": contract_id, "updated": now},
            )
        return analysis_id

    def fail_analysis(self, analysis_id: str, contract_id: str, error_summary: str) -> None:
        now = utc_now()
        with self.engine.begin() as conn:
            changed = conn.execute(
                text(f"UPDATE {CONTRACT_ANALYSES_TABLE} SET status='failed', error_summary=:error, finished_at=:finished WHERE id=:id AND status='running'"),
                {"id": analysis_id, "error": error_summary[:1000], "finished": now},
            ).rowcount
            # 陈旧运行可能已被恢复流程置为 FAILED，并已有新运行接管。
            # 此时旧进程迟到的异常不能覆盖新运行的合同状态。
            if changed != 1:
                return
            has_success = bool(conn.execute(
                text(f"SELECT 1 FROM {CONTRACT_ANALYSES_TABLE} WHERE contract_id=:id AND status='succeeded' LIMIT 1"),
                {"id": contract_id},
            ).first())
            has_running = bool(conn.execute(
                text(f"SELECT 1 FROM {CONTRACT_ANALYSES_TABLE} WHERE contract_id=:id AND status='running' LIMIT 1"),
                {"id": contract_id},
            ).first())
            conn.execute(
                text(f"UPDATE {CONTRACTS_TABLE} SET processing_status=:status, error_message=:error, updated_at=:updated WHERE id=:id"),
                {
                    "id": contract_id,
                    "status": "analyzing" if has_running else "ready" if has_success else "failed",
                    "error": "" if has_running else error_summary[:1000],
                    "updated": now,
                },
            )

    def save_analysis_extraction(self, analysis_id: str, extraction: Any) -> None:
        """在义务和风险分析前先固化已校验证据的事实抽取结果。"""
        payload = extraction.model_dump(mode="json") if hasattr(extraction, "model_dump") else extraction
        with self.engine.begin() as conn:
            conn.execute(
                text(f"UPDATE {CONTRACT_ANALYSES_TABLE} SET extraction_json=:payload WHERE id=:id AND status='running'"),
                {"id": analysis_id, "payload": _json_dump(payload)},
            )

    def save_analysis_result(self, analysis_id: str, contract_id: str, result: ContractAnalysisResult, *, warning_summary: str = "") -> None:
        now = utc_now()
        payload = result.model_dump(mode="json")
        obligation_business_ids = [str(item.id or "") for item in result.obligations]
        obligation_ids = [
            _versioned_obligation_id(analysis_id, item.id, index)
            for index, item in enumerate(result.obligations)
        ]
        obligation_id_map = {
            business_id: obligation_ids[index]
            for index, business_id in enumerate(obligation_business_ids)
            if business_id
        }
        with self.engine.begin() as conn:
            previous_id = conn.execute(
                text(f"SELECT id FROM {CONTRACT_ANALYSES_TABLE} WHERE contract_id=:id AND status='succeeded' ORDER BY finished_at DESC LIMIT 1"),
                {"id": contract_id},
            ).scalar()
            previous_statuses: dict[str, str] = {}
            if previous_id:
                previous_statuses = {
                    str(row["risk_fingerprint"]): str(row["status"])
                    for row in conn.execute(
                        text(f"SELECT risk_fingerprint, status FROM {CONTRACT_RISKS_TABLE} WHERE analysis_id=:id AND status<>'open'"),
                        {"id": previous_id},
                    ).mappings()
                }
            changed = conn.execute(
                text(
                    f"""UPDATE {CONTRACT_ANALYSES_TABLE}
                    SET status='succeeded', extraction_json=:extraction, summary_json=:summary,
                        finished_at=:finished_at, error_summary=:warning WHERE id=:id AND status='running'"""
                ),
                {"id": analysis_id, "extraction": _json_dump(payload["extraction"]), "summary": _json_dump(payload["summary"]), "finished_at": now, "warning": warning_summary[:1000] or None},
            ).rowcount
            if changed != 1:
                raise ContractConflictError("分析运行已失效，结果未写入")
            for index, item in enumerate(payload["obligations"]):
                obligation_id = obligation_ids[index]
                rule = item.get("time_rule") or {}
                evidence = item.get("source_evidence") or {}
                conn.execute(
                    text(
                        f"""INSERT INTO {CONTRACT_OBLIGATIONS_TABLE}
                        (id, contract_id, analysis_id, item_index, obligation_type, title, description,
                         responsible_party, beneficiary_party, trigger_type, trigger_event_type,
                         trigger_event_id, trigger_description, time_rule_json, planned_date, amount,
                         currency, percentage, status, risk_level, evidence_json, source_page,
                         source_chunk_id, confidence, needs_review, created_at, updated_at)
                        VALUES (:id, :contract_id, :analysis_id, :item_index, :obligation_type, :title,
                                :description, :responsible_party, :beneficiary_party, :trigger_type,
                                :trigger_event_type, :trigger_event_id, :trigger_description,
                                :time_rule_json, :planned_date, :amount, :currency, :percentage, :status,
                                :risk_level, :evidence_json, :source_page, :source_chunk_id, :confidence,
                                :needs_review, :created_at, :updated_at)"""
                    ),
                    {
                        "id": obligation_id, "contract_id": contract_id, "analysis_id": analysis_id,
                        "item_index": index, "obligation_type": item["obligation_type"], "title": item["title"],
                        "description": item.get("description", ""), "responsible_party": item.get("responsible_party"),
                        "beneficiary_party": item.get("beneficiary_party"), "trigger_type": item.get("trigger_type"),
                        "trigger_event_type": item.get("trigger_event_type"), "trigger_event_id": item.get("trigger_event_id"),
                        "trigger_description": item.get("trigger_description"), "time_rule_json": _json_dump(rule),
                        "planned_date": item.get("planned_date"), "amount": item.get("amount"), "currency": item.get("currency"),
                        "percentage": item.get("percentage"), "status": item["status"], "risk_level": item["risk_level"],
                        "evidence_json": _json_dump(evidence), "source_page": evidence.get("page_number"),
                        "source_chunk_id": evidence.get("chunk_id"), "confidence": item.get("confidence", 0),
                        "needs_review": bool(item.get("needs_review")), "created_at": now, "updated_at": now,
                    },
                )
            self._insert_risks(
                conn,
                analysis_id,
                contract_id,
                result.risks,
                obligation_ids,
                now,
                previous_statuses,
                obligation_id_map,
            )
        self.update_contract_facts(contract_id, result)

    def _insert_risks(
        self,
        conn,
        analysis_id: str,
        contract_id: str,
        risks,
        obligation_ids: list[str],
        now: str,
        previous_statuses: dict[str, str] | None = None,
        obligation_id_map: dict[str, str] | None = None,
    ) -> None:
        for risk in risks:
            item = risk.model_dump(mode="json") if hasattr(risk, "model_dump") else dict(risk)
            evidence = item.get("source_evidence") or {}
            evidence_payload: dict[str, Any] = evidence
            if item.get("supporting_evidence"):
                evidence_payload = {"primary": evidence, "supporting": item["supporting_evidence"]}
            related_index = item.get("related_obligation_index")
            related_id = item.get("related_obligation_id")
            if isinstance(related_index, int) and 0 <= related_index < len(obligation_ids):
                related_id = obligation_ids[related_index]
            elif related_id:
                related_id = (obligation_id_map or {}).get(str(related_id), related_id)
            fingerprint = "|".join(risk_fingerprint(contract_id, risk))
            carried_status = (previous_statuses or {}).get(fingerprint, item["status"])
            conn.execute(
                text(
                    f"""INSERT INTO {CONTRACT_RISKS_TABLE}
                    (id, contract_id, analysis_id, risk_fingerprint, risk_type, risk_name, risk_level,
                     risk_source, status, severity, likelihood, urgency, description, impact, reason,
                     suggestion, evidence_json, source_page, source_chunk_id, related_obligation_id,
                     confidence, needs_review, created_at)
                    VALUES (:id, :contract_id, :analysis_id, :risk_fingerprint, :risk_type, :risk_name,
                            :risk_level, :risk_source, :status, :severity, :likelihood, :urgency,
                            :description, :impact, :reason, :suggestion, :evidence_json, :source_page,
                            :source_chunk_id, :related_obligation_id, :confidence, :needs_review, :created_at)"""
                ),
                {
                    "id": str(uuid.uuid4()), "contract_id": contract_id, "analysis_id": analysis_id,
                    "risk_fingerprint": fingerprint, "risk_type": item["risk_type"], "risk_name": item["risk_name"],
                    "risk_level": item["risk_level"], "risk_source": item["risk_source"], "status": carried_status,
                    "severity": item["severity"], "likelihood": item["likelihood"], "urgency": item["urgency"],
                    "description": item["description"], "impact": item.get("impact"), "reason": item["reason"],
                    "suggestion": item["suggestion"], "evidence_json": _json_dump(evidence_payload),
                    "source_page": evidence.get("page_number"), "source_chunk_id": evidence.get("chunk_id"),
                    "related_obligation_id": related_id, "confidence": item.get("confidence", 0),
                    "needs_review": bool(item.get("needs_review")), "created_at": now,
                },
            )

    def latest_analysis(self, contract_id: str) -> dict[str, Any] | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                text(f"SELECT * FROM {CONTRACT_ANALYSES_TABLE} WHERE contract_id=:id AND status='succeeded' ORDER BY finished_at DESC LIMIT 1"),
                {"id": contract_id},
            ).mappings().first()
        return _analysis_payload(row) if row else None

    def list_analysis_runs(self, contract_id: str, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(f"SELECT id, status, analysis_version, model_name, started_at, finished_at, error_summary, created_at FROM {CONTRACT_ANALYSES_TABLE} WHERE contract_id=:id ORDER BY created_at DESC LIMIT :limit OFFSET :offset"),
                {"id": contract_id, "limit": max(1, min(limit, 200)), "offset": max(0, offset)},
            ).mappings().all()
        runs = [_analysis_payload(row) for row in rows]
        for run in runs:
            error = str(run.get("error_summary") or "")
            if error and not re.fullmatch(r"[a-z_]+:[A-Za-z0-9_]+(?:,[a-z_]+:[A-Za-z0-9_]+)*", error):
                run["error_summary"] = "分析失败，请查看服务日志"
        return runs

    def count_analysis_runs(self, contract_id: str) -> int:
        with self.engine.begin() as conn:
            return int(conn.execute(text(f"SELECT COUNT(*) FROM {CONTRACT_ANALYSES_TABLE} WHERE contract_id=:id"), {"id": contract_id}).scalar_one())

    def _latest_analysis_id(self, contract_id: str) -> str | None:
        analysis = self.latest_analysis(contract_id)
        return str(analysis["id"]) if analysis else None

    def list_obligations(self, contract_id: str, *, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        analysis_id = self._latest_analysis_id(contract_id)
        if not analysis_id:
            return []
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(f"SELECT * FROM {CONTRACT_OBLIGATIONS_TABLE} WHERE analysis_id=:id ORDER BY item_index LIMIT :limit OFFSET :offset"),
                {"id": analysis_id, "limit": max(1, min(limit, 500)), "offset": max(0, offset)},
            ).mappings().all()
        return [_row_payload(row) for row in rows]

    def count_obligations(self, contract_id: str) -> int:
        analysis_id = self._latest_analysis_id(contract_id)
        if not analysis_id:
            return 0
        with self.engine.begin() as conn:
            return int(conn.execute(text(f"SELECT COUNT(*) FROM {CONTRACT_OBLIGATIONS_TABLE} WHERE analysis_id=:id"), {"id": analysis_id}).scalar_one())

    def list_risks(self, contract_id: str, *, risk_level: str | None = None, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        analysis_id = self._latest_analysis_id(contract_id)
        if not analysis_id:
            return []
        filters = ["analysis_id=:analysis_id"]
        params: dict[str, Any] = {"analysis_id": analysis_id, "limit": max(1, min(limit, 500)), "offset": max(0, offset)}
        if risk_level:
            filters.append("risk_level=:risk_level")
            params["risk_level"] = risk_level
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(f"SELECT * FROM {CONTRACT_RISKS_TABLE} WHERE {' AND '.join(filters)} ORDER BY FIELD(risk_level, 'critical','high','medium','low'), created_at LIMIT :limit OFFSET :offset"),
                params,
            ).mappings().all()
        return [_row_payload(row) for row in rows]

    def count_risks(self, contract_id: str, *, risk_level: str | None = None) -> int:
        analysis_id = self._latest_analysis_id(contract_id)
        if not analysis_id:
            return 0
        filters = ["analysis_id=:analysis_id"]
        params: dict[str, Any] = {"analysis_id": analysis_id}
        if risk_level:
            filters.append("risk_level=:risk_level")
            params["risk_level"] = risk_level
        with self.engine.begin() as conn:
            return int(conn.execute(text(f"SELECT COUNT(*) FROM {CONTRACT_RISKS_TABLE} WHERE {' AND '.join(filters)}"), params).scalar_one())

    def update_obligation(self, obligation_id: str, contract_id: str, values: dict[str, Any]) -> bool:
        allowed = {"status", "actual_completed_at", "note", "planned_date", "trigger_event_id", "time_rule_json", "risk_level"}
        safe_values = {key: value for key, value in values.items() if key in allowed}
        if not safe_values:
            return False
        fields = [f"{key}=:{key}" for key in safe_values] + ["updated_at=:updated_at"]
        params = {**safe_values, "id": obligation_id, "contract_id": contract_id, "updated_at": utc_now()}
        with self.engine.begin() as conn:
            result = conn.execute(
                text(f"UPDATE {CONTRACT_OBLIGATIONS_TABLE} SET {', '.join(fields)} WHERE id=:id AND contract_id=:contract_id"),
                params,
            )
        return bool(result.rowcount)

    def update_obligations_batch(self, contract_id: str, rows: list[dict[str, Any]]) -> None:
        """事件重算批量落库，避免每条义务单独开启事务。"""
        if not rows:
            return
        now = utc_now()
        payload = [
            {
                "id": str(item["id"]),
                "contract_id": contract_id,
                "status": str(item["status"]),
                "planned_date": item.get("planned_date"),
                "trigger_event_id": item.get("trigger_event_id"),
                "time_rule_json": item.get("time_rule_json") or "{}",
                "risk_level": str(item.get("risk_level") or "low"),
                "updated_at": now,
            }
            for item in rows
        ]
        with self.engine.begin() as conn:
            conn.execute(
                text(f"""UPDATE {CONTRACT_OBLIGATIONS_TABLE}
                SET status=:status, planned_date=:planned_date, trigger_event_id=:trigger_event_id,
                    time_rule_json=:time_rule_json, risk_level=:risk_level, updated_at=:updated_at
                WHERE id=:id AND contract_id=:contract_id"""),
                payload,
            )

    def update_risk_status(self, risk_id: str, contract_id: str, status: str) -> bool:
        analysis_id = self._latest_analysis_id(contract_id)
        if not analysis_id:
            return False
        with self.engine.begin() as conn:
            result = conn.execute(
                text(f"UPDATE {CONTRACT_RISKS_TABLE} SET status=:status WHERE id=:id AND contract_id=:contract_id AND analysis_id=:analysis_id"),
                {"id": risk_id, "contract_id": contract_id, "analysis_id": analysis_id, "status": status},
            )
        return bool(result.rowcount)

    def update_contract_status(self, contract_id: str, status: str) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(
                text(f"UPDATE {CONTRACTS_TABLE} SET status=:status, updated_at=:updated_at WHERE id=:id"),
                {"id": contract_id, "status": status, "updated_at": utc_now()},
            )
        return bool(result.rowcount)

    def replace_rule_risks(self, contract_id: str, risks: list[Any]) -> None:
        analysis_id = self._latest_analysis_id(contract_id)
        if not analysis_id:
            return
        fingerprints = ["|".join(risk_fingerprint(contract_id, risk)) for risk in risks]
        fingerprint_params = {f"fingerprint_{index}": value for index, value in enumerate(fingerprints)}
        matching_hybrid = ""
        if fingerprint_params:
            placeholders = ",".join(f":{key}" for key in fingerprint_params)
            matching_hybrid = f" OR (risk_source='hybrid' AND risk_fingerprint IN ({placeholders}))"
        with self.engine.begin() as conn:
            previous_statuses = {
                str(row["risk_fingerprint"]): str(row["status"])
                for row in conn.execute(
                    text(f"SELECT risk_fingerprint, status FROM {CONTRACT_RISKS_TABLE} WHERE analysis_id=:id AND (risk_source='rule'{matching_hybrid}) AND status<>'open'"),
                    {"id": analysis_id, **fingerprint_params},
                ).mappings()
            }
            conn.execute(
                text(f"DELETE FROM {CONTRACT_RISKS_TABLE} WHERE analysis_id=:id AND (risk_source='rule'{matching_hybrid})"),
                {"id": analysis_id, **fingerprint_params},
            )
            self._insert_risks(conn, analysis_id, contract_id, risks, [], utc_now(), previous_statuses)
        levels = self.list_risks(contract_id)
        with self.engine.begin() as conn:
            conn.execute(
                text(f"UPDATE {CONTRACTS_TABLE} SET overall_risk_level=:level, updated_at=:updated WHERE id=:id"),
                {"id": contract_id, "level": overall_risk_level(levels), "updated": utc_now()},
            )

    def add_event(
        self,
        contract_id: str,
        *,
        event_type: str,
        event_date: str,
        related_obligation_id: str | None,
        description: str,
        source: str,
        user_id: str,
    ) -> str:
        event_id = str(uuid.uuid4())
        with self.engine.begin() as conn:
            locked = conn.execute(text(f"SELECT id FROM {CONTRACTS_TABLE} WHERE id=:id FOR UPDATE"), {"id": contract_id}).first()
            if not locked:
                raise ContractNotFoundError()
            if event_type != "custom":
                existing = conn.execute(
                    text(f"""SELECT id FROM {CONTRACT_EVENTS_TABLE}
                    WHERE contract_id=:contract_id AND event_type=:event_type AND event_date=:event_date
                      AND ((related_obligation_id IS NULL AND :related_id IS NULL) OR related_obligation_id=:related_id)
                    LIMIT 1"""),
                    {"contract_id": contract_id, "event_type": event_type, "event_date": event_date, "related_id": related_obligation_id},
                ).scalar()
                if existing:
                    return str(existing)
            conn.execute(
                text(
                    f"""INSERT INTO {CONTRACT_EVENTS_TABLE}
                    (id, contract_id, event_type, event_date, related_obligation_id, description,
                     source, created_by, created_at)
                    VALUES (:id, :contract_id, :event_type, :event_date, :related_obligation_id,
                            :description, :source, :created_by, :created_at)"""
                ),
                {
                    "id": event_id, "contract_id": contract_id, "event_type": event_type,
                    "event_date": event_date, "related_obligation_id": related_obligation_id,
                    "description": description, "source": source, "created_by": user_id,
                    "created_at": utc_now(),
                },
            )
        return event_id

    def update_event(self, event_id: str, contract_id: str, values: dict[str, Any]) -> bool:
        allowed = {"event_type", "event_date", "related_obligation_id", "description", "source"}
        safe_values = {key: value for key, value in values.items() if key in allowed and value is not None}
        if not safe_values:
            return False
        fields = [f"{key}=:{key}" for key in safe_values]
        with self.engine.begin() as conn:
            result = conn.execute(
                text(f"UPDATE {CONTRACT_EVENTS_TABLE} SET {', '.join(fields)} WHERE id=:id AND contract_id=:contract_id"),
                {**safe_values, "id": event_id, "contract_id": contract_id},
            )
        return bool(result.rowcount)

    def list_events(self, contract_id: str) -> list[dict[str, Any]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(f"SELECT * FROM {CONTRACT_EVENTS_TABLE} WHERE contract_id=:id ORDER BY event_date, created_at"),
                {"id": contract_id},
            ).mappings().all()
        return [_row_payload(row) for row in rows]

    def detail(self, contract_id: str, access: ContractAccessContext) -> dict[str, Any] | None:
        contract = self.get_contract(contract_id, access)
        if contract is None:
            return None
        contract["documents"] = self.list_documents(contract_id)
        contract["analysis"] = self.latest_analysis(contract_id)
        contract["analysis_runs"] = self.list_analysis_runs(contract_id)
        contract["obligations"] = self.list_obligations(contract_id)
        contract["risks"] = self.list_risks(contract_id)
        contract["events"] = self.list_events(contract_id)
        return contract


@lru_cache(maxsize=1)
def get_contract_store() -> ContractStore:
    return ContractStore()
