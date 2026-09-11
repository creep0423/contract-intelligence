"""Benchmark manifest、Gold annotation 与预测快照加载和一致性校验。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from qa_core.evaluation.matching import normalize_date, normalize_number, normalize_percentage, normalize_text
from qa_core.evaluation.schemas import BenchmarkCase, BenchmarkManifest, DatasetSourceType, MatchType, PredictionBundle


@dataclass(frozen=True)
class LoadedBenchmark:
    root: Path
    manifest: BenchmarkManifest
    cases: list[BenchmarkCase]


class AnnotationValidationError(ValueError):
    pass


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnnotationValidationError(f"JSON 格式错误：{path}:{exc.lineno}:{exc.colno}") from exc


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise AnnotationValidationError(f"Benchmark 路径越界：{relative}") from exc
    return candidate


def load_benchmark(manifest_path: str | Path, *, suite: str = "full") -> LoadedBenchmark:
    manifest_file = Path(manifest_path).resolve()
    root = manifest_file.parent
    try:
        manifest = BenchmarkManifest.model_validate(_read_json(manifest_file))
        raw_cases = _read_json(_safe_child(root, manifest.cases_file))
        cases = [BenchmarkCase.model_validate(item) for item in raw_cases]
    except ValidationError as exc:
        raise AnnotationValidationError(str(exc)) from exc
    selected = [case for case in cases if suite in case.suites]
    errors = validate_annotations(root, manifest, cases)
    if errors:
        raise AnnotationValidationError("\n".join(errors))
    return LoadedBenchmark(root=root, manifest=manifest, cases=selected)


def load_predictions(path: str | Path) -> PredictionBundle:
    try:
        return PredictionBundle.model_validate(_read_json(Path(path).resolve()))
    except ValidationError as exc:
        raise AnnotationValidationError(str(exc)) from exc


def validate_annotations(root: Path, manifest: BenchmarkManifest, cases: list[BenchmarkCase]) -> list[str]:
    errors: list[str] = []
    case_ids = [case.case_id for case in cases]
    for duplicate in _duplicates(case_ids):
        errors.append(f"重复 case_id：{duplicate}")
    supported = set(manifest.supported_case_types)
    for case in cases:
        unknown_tags = sorted(set(case.tags) - supported)
        if unknown_tags:
            errors.append(f"{case.case_id}: 未登记标签 {unknown_tags}")
        if not case.synthetic and any("synthetic" in item.lower() for item in case.documents):
            errors.append(f"{case.case_id}: 真实 UAT case 不得引用 synthetic 命名文件")
        document_names = {Path(item).name for item in case.documents}
        document_text_by_name: dict[str, str] = {}
        may_read_contract_documents = manifest.source_type == DatasetSourceType.SYNTHETIC and case.synthetic
        for document in case.documents:
            path = _safe_child(root, document)
            if may_read_contract_documents:
                if not path.is_file():
                    errors.append(f"{case.case_id}: 合同文件不存在 {document}")
                else:
                    document_text_by_name[path.name] = path.read_text(encoding="utf-8", errors="replace")
        source_ids = [source.source_id for source in case.sources]
        for duplicate in _duplicates(source_ids):
            errors.append(f"{case.case_id}: 重复 source_id {duplicate}")
        source_map = {source.source_id: source for source in case.sources}
        for source in case.sources:
            if not case.synthetic and not source.document_id:
                errors.append(f"{case.case_id}: Real UAT source {source.source_id} 必须提供稳定 document_id")
            if source.document_name and source.document_name not in document_names:
                errors.append(f"{case.case_id}: source {source.source_id} 引用未登记文档 {source.document_name}")
            elif may_read_contract_documents and normalize_text(source.text_snippet) not in normalize_text(
                document_text_by_name.get(str(source.document_name or ""), "")
            ):
                errors.append(f"{case.case_id}: source {source.source_id} 的 text_snippet 不存在于合同文件")
        for field in case.gold_fields:
            if field.status.value != "EXPECTED_VALUE":
                continue
            if field.match_type == MatchType.DATE and normalize_date(field.value) is None:
                errors.append(f"{case.case_id}: field {field.name} 日期格式无效")
            if field.match_type == MatchType.NUMERIC and normalize_number(field.value) is None:
                errors.append(f"{case.case_id}: field {field.name} 数值格式无效")
            if field.match_type == MatchType.PERCENTAGE and normalize_percentage(field.value) is None:
                errors.append(f"{case.case_id}: field {field.name} 百分比格式无效")
        obligation_ids = [item.gold_id for item in case.gold_obligations]
        risk_ids = [item.gold_risk_id for item in case.gold_risks]
        question_ids = [item.question_id for item in case.gold_qa]
        for label, values in (("gold obligation", obligation_ids), ("gold risk", risk_ids), ("question", question_ids)):
            for duplicate in _duplicates(values):
                errors.append(f"{case.case_id}: 重复 {label} id {duplicate}")
        all_references: list[tuple[str, list[str]]] = []
        all_references.extend((f"field:{item.name}", item.source_references) for item in case.gold_fields)
        all_references.extend((f"obligation:{item.gold_id}", item.source_references) for item in case.gold_obligations)
        all_references.extend((f"risk:{item.gold_risk_id}", item.source_references) for item in case.gold_risks)
        all_references.extend((f"qa:{item.question_id}", item.expected_sources) for item in case.gold_qa)
        all_references.extend((f"conflict:{item.conflict_id}", item.source_references) for item in case.gold_conflicts)
        for owner, references in all_references:
            for reference in references:
                if reference not in source_map:
                    errors.append(f"{case.case_id}: {owner} 引用不存在的 Gold source {reference}")
        obligation_id_set = set(obligation_ids)
        for risk in case.gold_risks:
            if risk.related_obligation_gold_id and risk.related_obligation_gold_id not in obligation_id_set:
                errors.append(
                    f"{case.case_id}: risk {risk.gold_risk_id} 引用不存在的 obligation {risk.related_obligation_gold_id}"
                )
        if not case.synthetic:
            for conflict in case.gold_conflicts:
                if not conflict.amendment or conflict.old_value is None or conflict.new_value is None:
                    errors.append(
                        f"{case.case_id}: Real UAT conflict {conflict.conflict_id} 必须提供 amendment、old_value、new_value"
                    )
    return errors


def validate_prediction_access(
    manifest: BenchmarkManifest,
    cases: list[BenchmarkCase],
    bundle: PredictionBundle,
) -> list[str]:
    """真实数据的预测必须来自已验证 DataScope；Evaluator 自身不得读取合同正文。"""
    uses_real_data = manifest.source_type != DatasetSourceType.SYNTHETIC or any(not case.synthetic for case in cases)
    if not uses_real_data:
        return []
    access = bundle.data_access
    if access is None:
        return ["Real UAT prediction bundle 缺少 data_access 授权声明"]
    if not access.trusted_upstream_verified:
        return ["Real UAT prediction bundle 未确认 trusted upstream"]
    return []


def validate_prediction_coverage(cases: list[BenchmarkCase], bundle: PredictionBundle) -> list[str]:
    errors: list[str] = []
    prediction_ids = [item.case_id for item in bundle.predictions]
    for duplicate in _duplicates(prediction_ids):
        errors.append(f"预测包含重复 case_id：{duplicate}")
    available = set(prediction_ids)
    for case in cases:
        if case.case_id not in available:
            errors.append(f"缺少 case prediction：{case.case_id}")
    return errors


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)
