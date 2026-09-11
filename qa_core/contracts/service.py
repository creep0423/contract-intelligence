"""合同上传/索引、分析入口、事件重算、详情与时间轴应用服务。"""

from __future__ import annotations

import io
import re
import uuid
import zipfile
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from langchain_core.documents import Document

from qa_core.config.logging_config import get_logger
from qa_core.config.settings import PROJECT_ROOT, get_settings
from qa_core.contracts.analysis import ContractAnalysisService
from qa_core.contracts.errors import ContractNotFoundError
from qa_core.contracts.obligations import recalculate_obligations, transition_obligation_status
from qa_core.contracts.risk import RISK_LEVEL_SCORE, evaluate_deterministic_risks, transition_risk_status
from qa_core.contracts.schemas import (
    ContractAccessContext,
    ContractAnalysisResult,
    ContractDocumentType,
    ContractEventCreate,
    ContractEventUpdate,
    ContractExtractionResult,
    ContractObligation,
    ObligationStatus,
    ProcessingStatus,
    RiskLevel,
    RiskStatus,
    ContractStatus,
    TimeRule,
)
from qa_core.contracts.store import ContractStore, get_contract_store
from qa_core.contracts.time_rules import parse_date
from qa_core.governance.chunk_versions import ChunkVersionIndex
from qa_core.governance.data_scope import resolve_data_scope
from qa_core.governance.kb_versions import get_kb_version_store, resolve_active_kb_version
from qa_core.indexing.chunking import split_contract_documents
from qa_core.indexing.document_loaders import SUPPORTED_DOCUMENT_SUFFIXES, load_file
from qa_core.indexing.document_normalizer import normalize_documents
from qa_core.retrieval.factory import get_doc_store
from qa_core.scenarios.registry import resolve_scenario


logger = get_logger(__name__)
CONTRACT_SCENARIO_ID = "tender_contract_risk"
CONTRACT_SOURCE = "contract"
CONTRACT_STORAGE_ROOT = PROJECT_ROOT / "data" / "contracts"
OOXML_SUFFIXES = {".docx", ".pptx", ".xlsx"}
OOXML_REQUIRED_PARTS = {
    ".docx": "word/document.xml",
    ".pptx": "ppt/presentation.xml",
    ".xlsx": "xl/workbook.xml",
}
OLE_SUFFIXES = {".doc", ".ppt", ".xls"}
TEXT_SUFFIXES = {".txt", ".md", ".csv", ".html", ".htm"}
UPLOAD_MIME_TYPES: dict[str, set[str]] = {
    ".pdf": {"application/pdf"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/zip"},
    ".doc": {"application/msword"},
    ".pptx": {"application/vnd.openxmlformats-officedocument.presentationml.presentation", "application/zip"},
    ".ppt": {"application/vnd.ms-powerpoint"},
    ".xlsx": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/zip"},
    ".xls": {"application/vnd.ms-excel"},
    ".txt": {"text/plain"},
    ".md": {"text/plain", "text/markdown"},
    ".csv": {"text/plain", "text/csv", "application/vnd.ms-excel"},
    ".html": {"text/html", "text/plain"},
    ".htm": {"text/html", "text/plain"},
}
CONFLICT_QUERY_TOPICS: tuple[tuple[str, ...], ...] = (
    ("付款", "支付", "尾款", "预付款", "价款", "发票"),
    ("交付", "交货", "发货", "上线"),
    ("验收",),
    ("解除", "终止"),
    ("续约",),
    ("违约", "赔偿", "责任"),
)


def contract_dataset_id(contract_id: str) -> str:
    return f"contract:{contract_id}"


def _safe_path_token(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", str(value or "")).strip("._")
    return cleaned[:120] or fallback


def validate_contract_upload(file_name: str, content: bytes, content_type: str | None = None) -> str:
    """校验后缀、明显伪造格式和可执行文件签名，返回安全展示文件名。"""
    if not content:
        raise ValueError("合同文件不能为空")
    client_name = Path(str(file_name or "")).name
    suffix = Path(client_name).suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
        raise ValueError(f"不支持的合同文件类型：{suffix or '无后缀'}")
    safe_stem = _safe_path_token(Path(client_name).stem, "contract")[:100]
    safe_name = f"{safe_stem}{suffix}"
    if content.startswith((b"MZ", b"\x7fELF")):
        raise ValueError("合同文件内容与允许的文档类型不匹配")
    if suffix == ".pdf" and not content.startswith(b"%PDF-"):
        raise ValueError("PDF 文件签名无效")
    if suffix in OOXML_SUFFIXES:
        if not content.startswith(b"PK"):
            raise ValueError("Office 文件签名无效")
        settings = get_settings()
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                members = archive.infolist()
                if len(members) > settings.contract_max_archive_entries:
                    raise ValueError("Office 文件包含过多压缩条目")
                total_uncompressed = 0
                total_compressed = 0
                names: set[str] = set()
                for member in members:
                    normalized_name = member.filename.replace("\\", "/")
                    path = PurePosixPath(normalized_name)
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError("Office 文件包含不安全的压缩路径")
                    if member.flag_bits & 0x1:
                        raise ValueError("不支持加密的 Office 文件")
                    names.add(normalized_name)
                    total_uncompressed += member.file_size
                    total_compressed += member.compress_size
                if total_uncompressed > settings.contract_max_archive_uncompressed_bytes:
                    raise ValueError("Office 文件解压后体积超过限制")
                if total_uncompressed and total_uncompressed > max(1, total_compressed) * settings.contract_max_archive_ratio:
                    raise ValueError("Office 文件压缩比异常")
                if "[Content_Types].xml" not in names or OOXML_REQUIRED_PARTS[suffix] not in names:
                    raise ValueError("Office 文件结构与后缀不匹配")
        except zipfile.BadZipFile as exc:
            raise ValueError("Office 文件压缩结构无效") from exc
    if suffix in OLE_SUFFIXES and not content.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        raise ValueError("旧版 Office 文件签名无效")
    if suffix in TEXT_SUFFIXES:
        if b"\x00" in content:
            raise ValueError("文本合同包含非法二进制内容")
        try:
            content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("文本合同必须使用 UTF-8 编码") from exc
    declared = str(content_type or "").split(";", 1)[0].strip().lower()
    if declared in {"application/x-msdownload", "application/x-executable"}:
        raise ValueError("合同文件 Content-Type 不受支持")
    if declared and declared != "application/octet-stream" and declared not in UPLOAD_MIME_TYPES.get(suffix, set()):
        raise ValueError("合同文件 Content-Type 与后缀不匹配")
    return safe_name


class ContractService:
    def __init__(self, store: ContractStore | None = None) -> None:
        self.store = store or get_contract_store()
        self.settings = get_settings()
        self.analysis_service = ContractAnalysisService(self.store)

    def upload_and_process(
        self,
        *,
        file_name: str,
        content: bytes,
        access: ContractAccessContext,
        contract_name: str | None = None,
        contract_type: str | None = None,
        document_type: ContractDocumentType = ContractDocumentType.MASTER_CONTRACT,
        document_version: str = "1",
        allowed_roles: list[str] | None = None,
        auto_analyze: bool = True,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        if len(content) > self.settings.contract_max_upload_bytes:
            raise ValueError(f"合同文件不能超过 {self.settings.contract_max_upload_bytes // (1024 * 1024)}MB")
        safe_name = validate_contract_upload(file_name, content, content_type)
        scenario = resolve_scenario(CONTRACT_SCENARIO_ID)
        contract_id = str(uuid.uuid4())
        roles = list(dict.fromkeys(allowed_roles or access.user_roles or ["public"]))
        if len(roles) > 32 or any(len(role) > 64 or not re.fullmatch(r"[A-Za-z0-9._:-]+", role) for role in roles):
            raise ValueError("合同访问角色格式无效")
        if "admin" not in access.user_roles and not set(roles).intersection(access.user_roles):
            roles.extend(role for role in access.user_roles if role not in roles)
        contract_dir = CONTRACT_STORAGE_ROOT / _safe_path_token(access.tenant_id, "default") / contract_id
        contract_dir.mkdir(parents=True, exist_ok=False)
        file_path = contract_dir / f"{uuid.uuid4()}_{safe_name}"
        try:
            file_path.write_bytes(content)
            self.store.create_contract(
                contract_id=contract_id,
                name=(contract_name or Path(safe_name).stem).strip(),
                contract_type=contract_type,
                scenario_id=scenario.scenario_id,
                dataset_id=contract_dataset_id(contract_id),
                access=access,
                allowed_roles=roles,
            )
        except Exception:
            file_path.unlink(missing_ok=True)
            try:
                contract_dir.rmdir()
            except OSError:
                pass
            raise
        document_indexed = False
        try:
            self._add_document(
                contract_id=contract_id,
                file_path=file_path,
                original_file_name=safe_name,
                document_type=document_type,
                document_version=document_version,
                contract=self._require_contract(contract_id, access),
            )
            document_indexed = True
            if auto_analyze:
                self.analysis_service.analyze(self._require_contract(contract_id, access))
        except Exception as exc:
            # 文档阶段失败由这里落合同状态；分析阶段会在自身事务中决定是 FAILED，
            # 还是因存在旧成功版本而继续保持 READY。不得在外层覆盖该决策。
            if not document_indexed:
                self.store.update_processing_status(contract_id, ProcessingStatus.FAILED, error_message="合同处理失败，请查看服务日志")
            logger.error("Contract upload processing failed: contract_id=%s error_type=%s", contract_id, type(exc).__name__)
            raise
        logger.info("Contract upload completed: contract_id=%s tenant_id=%s", contract_id, access.tenant_id)
        return self.get_detail(contract_id, access)

    def add_document(
        self,
        contract_id: str,
        *,
        file_name: str,
        content: bytes,
        access: ContractAccessContext,
        document_type: ContractDocumentType,
        document_version: str,
        reanalyze: bool = True,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        contract = self._require_contract(contract_id, access)
        if len(content) > self.settings.contract_max_upload_bytes:
            raise ValueError(f"合同文件不能超过 {self.settings.contract_max_upload_bytes // (1024 * 1024)}MB")
        safe_name = validate_contract_upload(file_name, content, content_type)
        document_id = str(uuid.uuid4())
        target = CONTRACT_STORAGE_ROOT / _safe_path_token(access.tenant_id, "default") / contract_id / f"{document_id}_{safe_name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        document_indexed = False
        try:
            self._add_document(
                contract_id=contract_id,
                file_path=target,
                original_file_name=safe_name,
                document_type=document_type,
                document_version=document_version,
                contract=contract,
                document_id=document_id,
            )
            document_indexed = True
            if reanalyze:
                self.analysis_service.analyze(contract)
        except Exception as exc:
            if not document_indexed:
                self.store.update_processing_status(contract_id, ProcessingStatus.FAILED, error_message="合同文档处理失败，请查看服务日志")
            logger.error(
                "Contract document processing failed: contract_id=%s document_id=%s error_type=%s",
                contract_id,
                document_id,
                type(exc).__name__,
            )
            raise
        return self.get_detail(contract_id, access)

    def _add_document(
        self,
        *,
        contract_id: str,
        file_path: Path,
        original_file_name: str,
        document_type: ContractDocumentType,
        document_version: str,
        contract: dict[str, Any],
        document_id: str | None = None,
    ) -> list[Document]:
        document_id = document_id or str(uuid.uuid4())
        try:
            self.store.create_document(
                document_id=document_id,
                contract_id=contract_id,
                file_name=original_file_name,
                file_path=str(file_path),
                file_type=Path(original_file_name).suffix.lower(),
                parser_backend=self.settings.document_parser_backend,
                document_type=document_type.value,
                document_version=document_version,
            )
        except Exception:
            file_path.unlink(missing_ok=True)
            raise
        self.store.update_processing_status(contract_id, ProcessingStatus.PARSING)
        chunk_ids: list[str] = []
        doc_store = None
        try:
            scenario = resolve_scenario(str(contract["scenario_id"]))
            kb_version = resolve_active_kb_version(None, scenario.scenario_id)
            version = get_kb_version_store(scenario.scenario_id).resolve_version(kb_version)
            raw_documents = load_file(file_path)
            if not raw_documents or not any(doc.page_content.strip() for doc in raw_documents):
                raise ValueError("合同解析结果为空；扫描件请先完成 OCR 复核后再上传")
            scope = resolve_data_scope(
                tenant_id=str(contract["tenant_id"]),
                dataset_id=str(contract["dataset_id"]),
                visibility=str(contract["visibility"]),
                user_roles=list(contract.get("allowed_roles") or ["public"]),
            )
            normalized = normalize_documents(
                raw_documents,
                file_path,
                CONTRACT_SOURCE,
                kb_version,
                scenario.scenario_id,
                version.version_seq,
                scope,
                list(contract.get("allowed_roles") or ["public"]),
            )
            for doc in normalized:
                doc.metadata.update(
                    {
                        "contract_id": contract_id,
                        "contract_name": contract["contract_name"],
                        "contract_type": contract.get("contract_type") or "",
                        "contract_document_id": document_id,
                        "contract_document_name": original_file_name,
                        "document_version": document_version,
                        "document_type": document_type.value,
                        "is_contract_document": True,
                    }
                )
            chunks, chunk_ids = split_contract_documents(normalized)
            if not chunks:
                raise ValueError("合同未生成可检索条款")
            doc_store = get_doc_store(scenario.doc_collection)
            doc_store.add_documents(chunks, ids=chunk_ids)
            ChunkVersionIndex().upsert_chunks(
                chunk_ids,
                scenario_id=scenario.scenario_id,
                source=CONTRACT_SOURCE,
                kb_version=kb_version,
                valid_from_seq=version.version_seq,
                file_path=str(file_path.resolve()),
            )
            self.store.update_document_parsed(
                document_id,
                doc_id=str(normalized[0].metadata.get("doc_id") or ""),
                page_count=len(raw_documents),
                chunk_ids=chunk_ids,
            )
            self.store.update_processing_status(contract_id, ProcessingStatus.PARSED)
            logger.info("Contract document parsed and indexed: contract_id=%s document_id=%s pages=%s chunks=%s", contract_id, document_id, len(raw_documents), len(chunks))
            return chunks
        except Exception:
            if doc_store is not None and chunk_ids:
                try:
                    doc_store.delete_ids(chunk_ids)
                except Exception:
                    logger.error(
                        "Contract vector compensation failed: contract_id=%s document_id=%s chunk_count=%s",
                        contract_id,
                        document_id,
                        len(chunk_ids),
                    )
            try:
                self.store.update_document_failed(document_id)
            except Exception:
                logger.error(
                    "Contract document failure status update failed: contract_id=%s document_id=%s",
                    contract_id,
                    document_id,
                )
            raise

    def analyze(self, contract_id: str, access: ContractAccessContext) -> dict[str, Any]:
        self.analysis_service.analyze(self._require_contract(contract_id, access))
        return self.get_detail(contract_id, access)

    def conflict_qa_response(
        self,
        contract_id: str,
        query: str,
        access: ContractAccessContext,
    ) -> dict[str, Any] | None:
        """对已落库的跨文档冲突给出确定性答复，禁止 LLM 擅自判定版本效力。"""
        self._require_contract(contract_id, access)
        normalized_query = str(query or "").lower()
        risks = [item for item in self.store.list_risks(contract_id) if item.get("risk_type") == "potential_conflict"]
        selected: dict[str, Any] | None = None
        for risk in risks:
            searchable = " ".join(
                str(risk.get(key) or "")
                for key in ("risk_name", "description", "reason")
            ).lower()
            if "冲突" in normalized_query or "差异" in normalized_query:
                selected = risk
                break
            if any(
                any(keyword in normalized_query for keyword in topic)
                and any(keyword in searchable for keyword in topic)
                for topic in CONFLICT_QUERY_TOPICS
            ):
                selected = risk
                break
        if selected is None:
            return None
        evidence_items = [selected.get("evidence")] + list(selected.get("supporting_evidence") or [])
        evidence_items = [item for item in evidence_items if isinstance(item, dict) and item.get("quote")]
        if len({str(item.get("document_id") or "") for item in evidence_items}) < 2:
            return None
        documents = {str(item["id"]): item for item in self.store.list_documents(contract_id)}
        fact_lines: list[str] = []
        sources: list[dict[str, Any]] = []
        for index, evidence in enumerate(evidence_items, start=1):
            document = documents.get(str(evidence.get("document_id") or ""), {})
            name = str(evidence.get("document_name") or document.get("file_name") or f"合同文件 {index}")
            fact_lines.append(f"- {name}：{evidence['quote']} [{index}]")
            sources.append(
                {
                    "document_id": evidence.get("document_id"),
                    "document_name": name,
                    "document_type": document.get("document_type"),
                    "document_version": document.get("document_version"),
                    "contract_id": contract_id,
                    "page": evidence.get("page_number"),
                    "chunk_id": evidence.get("chunk_id"),
                    "source_text": evidence["quote"],
                    "citation": name,
                    "metadata": {
                        "section": evidence.get("section"),
                        "clause_no": evidence.get("clause_no"),
                        "document_type": document.get("document_type"),
                        "document_version": document.get("document_version"),
                    },
                }
            )
        answer = (
            "【合同事实】\n"
            + "\n".join(fact_lines)
            + "\n\n【风险判断】\n上述合同文件对同一事项存在约定差异。当前系统没有版本优先或法律效力判定规则，不能自动认定其中一份约定优先。"
            + "\n\n【处理建议】\n请由企业法务结合签署时间、适用范围及双方真实意思确认补充协议的适用关系，并据此更新履约计划。"
        )
        return {"answer": answer, "sources": sources}

    def add_event(
        self,
        contract_id: str,
        payload: ContractEventCreate,
        access: ContractAccessContext,
    ) -> dict[str, Any]:
        detail = self.get_detail(contract_id, access)
        if parse_date(payload.event_date) is None:
            raise ValueError("事件日期格式无效")
        if payload.related_obligation_id and payload.related_obligation_id not in {str(item["id"]) for item in detail["obligations"]}:
            raise ContractNotFoundError()
        self.store.add_event(
            contract_id,
            event_type=payload.event_type.value,
            event_date=payload.event_date,
            related_obligation_id=payload.related_obligation_id,
            description=payload.description,
            source=payload.source,
            user_id=access.user_id,
        )
        self.recalculate_contract_obligations(contract_id, access)
        return self.get_detail(contract_id, access)

    def update_event(
        self,
        contract_id: str,
        event_id: str,
        payload: ContractEventUpdate,
        access: ContractAccessContext,
    ) -> dict[str, Any]:
        detail = self.get_detail(contract_id, access)
        if event_id not in {str(item["id"]) for item in detail["events"]}:
            raise ContractNotFoundError()
        if payload.event_date is not None and parse_date(payload.event_date) is None:
            raise ValueError("事件日期格式无效")
        values = payload.model_dump(mode="json", exclude_none=True)
        related_id = values.get("related_obligation_id")
        if related_id and related_id not in {str(item["id"]) for item in detail["obligations"]}:
            raise ContractNotFoundError()
        if not self.store.update_event(event_id, contract_id, values):
            raise ContractNotFoundError()
        self.recalculate_contract_obligations(contract_id, access)
        return self.get_detail(contract_id, access)

    def update_risk_status(
        self,
        contract_id: str,
        risk_id: str,
        status: RiskStatus,
        access: ContractAccessContext,
    ) -> dict[str, Any]:
        detail = self.get_detail(contract_id, access)
        current = next((item for item in detail["risks"] if str(item["id"]) == risk_id), None)
        if current is None:
            raise ContractNotFoundError()
        transition_risk_status(RiskStatus(str(current["status"])), status)
        if not self.store.update_risk_status(risk_id, contract_id, status.value):
            raise ContractNotFoundError()
        return self.get_detail(contract_id, access)

    def update_contract_status(
        self,
        contract_id: str,
        status: ContractStatus,
        access: ContractAccessContext,
    ) -> dict[str, Any]:
        self._require_contract(contract_id, access)
        self.store.update_contract_status(contract_id, status.value)
        return self.get_detail(contract_id, access)

    def recalculate_contract_obligations(
        self,
        contract_id: str,
        access: ContractAccessContext,
    ) -> list[ContractObligation]:
        detail = self.store.detail(contract_id, access)
        if detail is None:
            raise ContractNotFoundError()
        analysis = detail.get("analysis") or {}
        extraction = ContractExtractionResult.model_validate(analysis.get("extraction") or {})
        obligations = [self._obligation_from_row(row) for row in detail.get("obligations") or []]
        resolved = recalculate_obligations(obligations, extraction, detail.get("events") or [])
        rule_risks = evaluate_deterministic_risks(contract_id, extraction, resolved)
        self.store.update_obligations_batch(
            contract_id,
            [
                {
                    "id": item.id,
                    "status": item.status.value,
                    "planned_date": item.planned_date,
                    "trigger_event_id": item.trigger_event_id,
                    "time_rule_json": item.time_rule.model_dump_json(),
                    "risk_level": item.risk_level.value,
                }
                for item in resolved
            ],
        )
        self.store.replace_rule_risks(contract_id, rule_risks)
        related_levels: dict[str, RiskLevel] = {}
        for risk in self.store.list_risks(contract_id):
            obligation_id = str(risk.get("related_obligation_id") or "")
            if not obligation_id:
                continue
            level = RiskLevel(str(risk.get("risk_level") or "low"))
            if RISK_LEVEL_SCORE[level] > RISK_LEVEL_SCORE[related_levels.get(obligation_id, RiskLevel.LOW)]:
                related_levels[obligation_id] = level
        resolved = [item.model_copy(update={"risk_level": related_levels.get(str(item.id), RiskLevel.LOW)}) for item in resolved]
        self.store.update_obligations_batch(
            contract_id,
            [
                {
                    "id": item.id,
                    "status": item.status.value,
                    "planned_date": item.planned_date,
                    "trigger_event_id": item.trigger_event_id,
                    "time_rule_json": item.time_rule.model_dump_json(),
                    "risk_level": item.risk_level.value,
                }
                for item in resolved
            ],
        )
        logger.info("Contract obligations recalculated: contract_id=%s obligations=%s", contract_id, len(resolved))
        return resolved

    def update_obligation_status(
        self,
        contract_id: str,
        obligation_id: str,
        *,
        status: ObligationStatus,
        actual_completed_at: str | None,
        note: str | None,
        access: ContractAccessContext,
    ) -> dict[str, Any]:
        detail = self.get_detail(contract_id, access)
        current = next((item for item in detail["obligations"] if str(item["id"]) == obligation_id), None)
        if current is None:
            raise ContractNotFoundError()
        transition_obligation_status(ObligationStatus(str(current["status"])), status)
        updated = self.store.update_obligation(
            obligation_id,
            contract_id,
            {"status": status.value, "actual_completed_at": actual_completed_at, "note": note},
        )
        if not updated:
            raise ContractNotFoundError()
        self.recalculate_contract_obligations(contract_id, access)
        return self.get_detail(contract_id, access)

    def _obligation_from_row(self, row: dict[str, Any]) -> ContractObligation:
        payload = dict(row)
        payload["source_evidence"] = payload.pop("evidence", None) or None
        payload["time_rule"] = payload.pop("time_rule", {})
        return ContractObligation.model_validate(payload)

    def get_detail(self, contract_id: str, access: ContractAccessContext) -> dict[str, Any]:
        detail = self.store.detail(contract_id, access)
        if detail is None:
            raise ContractNotFoundError()
        detail["timeline"] = self._timeline(detail)
        detail["risk_dashboard"] = self._dashboard(detail)
        detail["workday_calculation_note"] = "工作日当前仅跳过周六、周日，未纳入法定节假日调整。"
        return detail

    def _timeline(self, detail: dict[str, Any]) -> list[dict[str, Any]]:
        analysis = detail.get("analysis") or {}
        extraction = (analysis.get("extraction") or {}).get("basic_info") or {}
        nodes: list[dict[str, Any]] = []
        for key, label in (("signing_date", "合同签署"), ("effective_date", "合同生效"), ("termination_date", "合同终止")):
            item = extraction.get(key) or {}
            if item.get("extracted_value"):
                nodes.append({"type": "contract", "name": label, "date": item["extracted_value"], "display_date": item["extracted_value"], "status": "completed" if key != "termination_date" else "pending", "source_evidence": item.get("source_evidence")})
        for event in detail.get("events") or []:
            nodes.append({"type": "event", "id": event["id"], "name": event["event_type"], "date": event["event_date"], "display_date": event["event_date"], "status": "completed", "description": event.get("description")})
        for row in detail.get("obligations") or []:
            nodes.append({"type": "obligation", "id": row["id"], "name": row["title"], "date": row.get("planned_date"), "display_date": row.get("planned_date") or (row.get("time_rule") or {}).get("original_text") or "待触发", "status": row["status"], "responsible_party": row.get("responsible_party"), "obligation_type": row["obligation_type"], "time_rule": row.get("time_rule"), "source_evidence": row.get("evidence")})
        return sorted(nodes, key=lambda item: (not bool(item.get("date")), str(item.get("date") or item.get("display_date") or "")))

    def _dashboard(self, detail: dict[str, Any]) -> dict[str, int]:
        risks = detail.get("risks") or []
        obligations = detail.get("obligations") or []
        today = date.today()
        warning_end = today + timedelta(days=self.settings.contract_upcoming_days)
        return {
            "critical_count": sum(item.get("risk_level") == "critical" for item in risks),
            "high_count": sum(item.get("risk_level") == "high" for item in risks),
            "medium_count": sum(item.get("risk_level") == "medium" for item in risks),
            "low_count": sum(item.get("risk_level") == "low" for item in risks),
            "overdue_obligation_count": sum(item.get("status") == "overdue" for item in obligations),
            "upcoming_obligation_count": sum(
                item.get("status") == "pending"
                and (planned := parse_date(item.get("planned_date"))) is not None
                and today <= planned <= warning_end
                for item in obligations
            ),
        }

    def _require_contract(self, contract_id: str, access: ContractAccessContext) -> dict[str, Any]:
        contract = self.store.get_contract(contract_id, access)
        if contract is None:
            raise ContractNotFoundError()
        return contract

    @staticmethod
    def _page(page: int, page_size: int) -> tuple[int, int, int, int]:
        safe_page = max(1, min(page, 10_000))
        safe_size = max(1, min(page_size, 200))
        return safe_page, safe_size, (safe_page - 1) * safe_size, safe_size

    def list_contracts_page(self, access: ContractAccessContext, *, risk_level: str | None = None, status: str | None = None, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        page, page_size, offset, limit = self._page(page, page_size)
        items = self.store.list_contracts(access, risk_level=risk_level, status=status, limit=limit, offset=offset)
        total = self.store.count_contracts(access, risk_level=risk_level, status=status)
        return {"contracts": items, "pagination": {"page": page, "page_size": page_size, "total": total}}

    def list_obligations_page(self, contract_id: str, access: ContractAccessContext, *, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        self._require_contract(contract_id, access)
        page, page_size, offset, limit = self._page(page, page_size)
        return {
            "contract_id": contract_id,
            "obligations": self.store.list_obligations(contract_id, limit=limit, offset=offset),
            "pagination": {"page": page, "page_size": page_size, "total": self.store.count_obligations(contract_id)},
        }

    def list_risks_page(self, contract_id: str, access: ContractAccessContext, *, risk_level: str | None = None, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        detail = self.get_detail(contract_id, access)
        page, page_size, offset, limit = self._page(page, page_size)
        return {
            "contract_id": contract_id,
            "risks": self.store.list_risks(contract_id, risk_level=risk_level, limit=limit, offset=offset),
            "dashboard": detail["risk_dashboard"],
            "pagination": {"page": page, "page_size": page_size, "total": self.store.count_risks(contract_id, risk_level=risk_level)},
        }

    def list_analysis_runs_page(self, contract_id: str, access: ContractAccessContext, *, page: int = 1, page_size: int = 50) -> dict[str, Any]:
        self._require_contract(contract_id, access)
        page, page_size, offset, limit = self._page(page, page_size)
        return {
            "contract_id": contract_id,
            "analysis_runs": self.store.list_analysis_runs(contract_id, limit=limit, offset=offset),
            "pagination": {"page": page, "page_size": page_size, "total": self.store.count_analysis_runs(contract_id)},
        }

    def list_contracts(self, access: ContractAccessContext, *, risk_level: str | None = None, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """兼容既有内部调用；HTTP 列表使用分页版本。"""
        return self.store.list_contracts(access, risk_level=risk_level, status=status, limit=limit)


@lru_cache(maxsize=1)
def get_contract_service() -> ContractService:
    return ContractService()
