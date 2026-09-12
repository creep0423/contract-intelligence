# -*- coding: utf-8 -*-
# ============================================================================
# 项目结构和代码约束守护检查
# ============================================================================
# 该脚本把当前项目已经明确下来的工程约束做成可执行检查，而不是只写在文档里。
# 适合在每次重构后运行，也适合后续接入 CI。
#
# 检查内容（11 项）：
#   1. Python 导入必须在文件头部，禁止函数内部临时导入；
#   2. 禁止 `try import A except ImportError import B` 这类隐藏兼容分支；
#   3. 禁止恢复旧版 `mysql_qa` / `rag_qa` / `legacy` 等运行入口；
#   4. 禁止代码重新引用旧链路模块；
#   5. 禁止恢复旧版 static/docs 自定义文档导出链路；
#   6. requirements.txt 必须锁定直接依赖版本，避免本地环境漂移。
#   7. 一期主链路不引入 LlamaIndex，避免两套 RAG 框架概念混用。
#   8. MySQL DDL 只能出现在 runtime_schema.sql，业务 Store 不做隐式建表。
#   9. source_filter 白名单只在入口层校验，检索底层不重复透传 valid_sources。
#   10. 直答意图只在路由层处理，检索意图分类和检索计划层不重复防御。
#   11. Pipeline 调试链路信任上下文/事件契约，不重复写空值兜底。
#
# 用法示例：
#   python scripts\check_project_guardrails.py
# ============================================================================
"""项目结构和代码约束守护检查。

该脚本把当前项目已经明确下来的工程约束做成可执行检查，而不是只写在文档里。
适合在每次重构后运行，也适合后续接入 CI。

检查内容：
1. Python 导入必须在文件头部，禁止函数内部临时导入；
2. 禁止 `try import A except ImportError import B` 这类隐藏兼容分支；
3. 禁止恢复旧版 `mysql_qa` / `rag_qa` / `legacy` 等运行入口；
4. 禁止代码重新引用旧链路模块；
5. 禁止恢复旧版 static/docs 自定义文档导出链路；
6. requirements.txt 必须锁定直接依赖版本，避免本地环境漂移；
7. 一期主链路不引入 LlamaIndex，避免两套 RAG 框架概念混用；
8. MySQL DDL 只能出现在 runtime_schema.sql，业务 Store 不做隐式建表；
9. source_filter 白名单只在入口层校验，检索底层不重复透传 valid_sources；
10. 直答意图只在路由层处理，检索意图分类和检索计划层不重复防御；
11. Pipeline 调试链路信任上下文/事件契约，不重复写空值兜底。
"""

from __future__ import annotations

# ast: Python 抽象语法树（用于检查 import 位置和 try-except-import 模式）
import ast

# csv: CSV 读取（检查 requirements.txt 格式）
import csv

# re: 正则表达式（匹配旧版模块引用）
import re

# sys: 系统功能
import sys

# tomli: TOML 解析（pyproject.toml）
import tomli as tomllib

# collections.Counter: 计数器
from collections import Counter

# dataclasses: 数据类定义
from dataclasses import dataclass

# pathlib.Path: 文件路径操作
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SCAN_DIRS = ("app.py", "qa_core", "scripts", "tests")
LEGACY_PATHS = (
    "mysql_qa",
    "rag_qa",
    "legacy",
    "api.py",
    "old_main.py",
    "new_main.py",
    "static/old_index.html",
    "static/docs",
    "static/css/doc.css",
    "static/js/vendor/mathjax",
    "scripts/build_docs.py",
    "mermaid_formatter.py",
    "convert_md_to_html.py",
    "tests/test_websocket_stream.py",
    "docker-compose.milvus.yml",
)
FORBIDDEN_ARTIFACT_PATHS = (
    "scripts/reports",
    "scripts/ocr/reports",
)
SOURCE_ARTIFACT_SCAN_DIRS = ("qa_core", "scripts", "tests")
LEGACY_IMPORT_PREFIXES = ("mysql_qa", "rag_qa", "legacy")
FORBIDDEN_ENV_TOKENS = (
    "EDURAG_USE_LEGACY_CONFIG",
    "RERANK_ENABLED",
    "INTENT_LLM_ENABLED",
    "RETRIEVAL_VARIANT_ENABLED",
)
FORBIDDEN_IMPORT_PREFIXES = (
    *LEGACY_IMPORT_PREFIXES,
    "rank_bm25",
    "RedisSearch",
    "llama_index",
)
FORBIDDEN_DIRECT_REQUIREMENTS = {
    "llama-index": "一期主链路不引入 LlamaIndex；如需说明，只放在讲义的可选扩展方案中。",
    "llama-index-core": "一期主链路不引入 LlamaIndex；如需说明，只放在讲义的可选扩展方案中。",
    "llama-index-readers-file": "一期主链路不引入 LlamaIndex；如需说明，只放在讲义的可选扩展方案中。",
    "rank-bm25": "当前主链路不使用 Python 本地 BM25；Sparse 检索统一由 Milvus BM25BuiltInFunction 承担。",
}
ALLOWED_RUNTIME_IMPORTS: dict[str, set[str]] = {
    "scripts/quality/evaluate_ragas_quality.py": {
        "qa_core.llm.client",
        "qa_core.retrieval.models",
    },
    "qa_core/memory/history.py": {
        "langchain_community.chat_message_histories",
    },
    "qa_core/intent/model_classifier.py": {
        "qa_core.config.settings",
    },
    "qa_core/llm/client.py": {
        "langchain_openai",
    },
    "qa_core/observability/langsmith_adapter.py": {
        "langsmith.run_helpers",
    },
    "qa_core/retrieval/factory.py": {
        "qa_core.retrieval.store",
        "qa_core.retrieval.milvus_compat",
        "qa_core.retrieval.models",
    },
}
PUBLIC_DOC_FORBIDDEN_FRAGMENTS = ("跟敲", "本章运行目录", "建议按下面顺序打开文件")
PUBLIC_DOC_SCAN_DIRS = ("docs",)
PUBLIC_TEXT_SCAN_DIRS = ("README.md", "CHANGELOG.md", "VERSIONING.md", "docs", "scripts", "qa_core")
FORBIDDEN_ROLE_TERMS = (
    "\u5b66\u751f",
    "\u8001\u5e08",
    "\u6559\u5b66",
    "\u8bfe\u5802",
    "\u6388\u8bfe",
    "\u8bb2\u5e08",
    "\u6559\u5e08",
)
REQUIRED_GITIGNORE_PATTERNS = (".env", "logs/", "reports/", "models/")
COMPOSE_ENV_OVERRIDE_KEYS = (
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MILVUS_URI",
    "EMBEDDING_MODEL_PATH",
    "RERANKER_MODEL_PATH",
)
SCENARIO_REQUIRED_FIELDS = (
    "scenario_id",
    "display_name",
    "industry",
    "assistant_name",
    "business_domain",
    "support_contact",
    "valid_sources",
    "faq_collection",
    "doc_collection",
)
SUPPORTED_SCENARIO_DOC_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".doc", ".ppt", ".pptx", ".csv", ".xlsx", ".xls"}
REQUIRED_SCENARIO_DOC_SUFFIXES = {".md", ".csv", ".xlsx", ".docx", ".pptx", ".pdf"}
FROZEN_SCENARIO_IDS = {
    "tender_contract_risk",
}
SCHEMA_DDL_ALLOWED_FILES: set[str] = set()
SCHEMA_BOOTSTRAP_ALLOWED_FILES = {
    "qa_core/storage/bootstrap.py",
}
SCHEMA_TEXT_EXEMPT_FILES = {
    "scripts/check_project_guardrails.py",
}
DDL_PATTERNS = (
    r"\bCREATE\s+TABLE\b",
    r"\bALTER\s+TABLE\b",
)
REQUIRED_MYSQL_SCHEMA_FILE = "qa_core/storage/runtime_schema.sql"
FORBIDDEN_MYSQL_MIGRATION_DIR = "qa_core/storage/migrations"
SOURCE_FILTER_BOUNDARY_FILES = (
    "qa_core/retrieval/filters.py",
    "qa_core/retrieval/store.py",
    "qa_core/pipeline/steps.py",
    "qa_core/pipeline/retrieval_steps.py",
)
PIPELINE_CONTEXT_CONTRACT_FRAGMENTS = {
    "context.raw_query or query": "RAGQueryContext 创建时已经写入 raw_query，调试返回不要再回退到入参 query。",
    "(context.raw_query or query)": "RAGQueryContext 创建时已经写入 raw_query，调试返回不要再回退到入参 query。",
    "raw_query = self.raw_query or self.query": "finalize_timings() 应直接使用 self.raw_query，避免重复兜底。",
    "bool(raw_query and raw_query != self.query)": "query_normalized 直接比较 raw_query 和 query 即可。",
    "question=self.raw_query or self.query": "record_trace() 使用请求创建时写入的 raw_query，不再回退到规范化 query。",
    'source_filter=self.retrieval_info.get("source_filter") or self.source_filter': "finalize_timings() 会补齐 source_filter，record_trace() 直接读取固定字段。",
    'context.retrieval_info.get("stage_timings_ms") or {}': "finalize_timings() 已写入 stage_timings_ms，调试返回直接读取即可。",
    'context.retrieval_info.get("slowest_stage") or {}': "路由和检索阶段已记录耗时，调试返回不要重复兜底 slowest_stage。",
    'route.intent.as_dict() if route.intent else {}': "终止路由必须带 intent，调试返回不要重复判断 route.intent。",
    'context.retrieval_info.get("boundary_reason") or "source_boundary"': "source 边界分支已由 detect_and_apply_boundary_answer() 写入 boundary_reason，不再兜底。",
    "doc.metadata or {}": "LangChain Document.metadata 固定为 dict，主链路不要重复兜底。",
    "dict(doc.metadata or {})": "LangChain Document.metadata 固定为 dict，直接 dict(doc.metadata)。",
    "hit.document.metadata or {}": "RetrievalHit.document 是 LangChain Document，metadata 固定为 dict。",
    "dict(hit.document.metadata or {})": "RetrievalHit.document.metadata 固定为 dict，复制时直接 dict(hit.document.metadata)。",
    "document.metadata or {}": "document_key() 接收 LangChain Document，metadata 固定为 dict。",
    "metadata: dict[str, Any] | None": "metadata 工具函数消费内部 Document.metadata/dict，不保留 None 分支。",
    "data = metadata or {}": "metadata 工具函数消费内部 dict，直接读取 metadata 字段。",
    "dict(child_doc.metadata or {})": "LangChain Document.metadata 固定为 dict，切分后直接 dict(child_doc.metadata)。",
    "dict(chunk.metadata or {})": "LangChain Document.metadata 固定为 dict，质量检测直接 dict(chunk.metadata)。",
    "(doc.metadata or {}).get": "主链路上下文统计直接读取 doc.metadata。",
    'doc.page_content or ""': "LangChain Document.page_content 固定为 str，主链路不要重复兜底。",
    'str(doc.page_content or "")': "LangChain Document.page_content 固定为 str，不需要转字符串兜底。",
    'answer or ""': "引用后处理接收内部 answer 字符串，不保留 None 兜底。",
    'str(query or "").strip()': "normalize_queries() 接收 Iterable[str]，直接 query.strip() 即可。",
    "candidate = str(item).strip()": "QueryVariants.queries 是 Pydantic 校验后的 list[str]，直接 item.strip() 即可。",
    'compact_query = (query or "").strip()': "should_try_faq_fast_path() 只接收规范化后的 context.query，直接 query.strip() 即可。",
    'raw_query = (query or "").strip()': "create_query_context() 由 QAService 传入 query: str，直接 query.strip() 即可。",
    'clean_query = (query or "").strip()': "MilvusKnowledgeStore.search() 接收 query: str，直接 query.strip() 即可。",
    'normalized = (query or "").strip().lower()': "question_category 内部接收 query: str，直接 query.strip().lower() 即可。",
    "[score for score in [faq_result.top_score, doc_result.top_score] if score is not None]": (
        "RetrievalResult.top_score 固定返回 float，上下文 top_score 直接 max(faq, doc)。"
    ),
    "retrieval_info = retrieval or {}": "end_event() 只由 finish_success() 调用，retrieval 是固定字段，不需要空值兜底。",
    "intent: dict[str, Any] | None = None": "end_event() 是内部成功收口事件，intent 必须由 RAGQueryContext 提供。",
    "retrieval: dict[str, Any] | None = None": "end_event() 是内部成功收口事件，retrieval 必须由 RAGQueryContext 提供。",
    'retrieval_info.get("stage_timings_ms", {})': "end_event() 应直接读取 retrieval 固定字段，不重复兜底 stage_timings_ms。",
    'retrieval_info.get("first_token_ms")': "end_event() 应直接读取 retrieval 固定字段，不重复兜底 first_token_ms。",
    'retrieval_info.get("slowest_stage")': "end_event() 应直接读取 retrieval 固定字段，不重复兜底 slowest_stage。",
    'retrieval.get("slowest_stage")': "finalize_timings() 已固定写入 slowest_stage，end_event() 直接读取即可。",
    'final_event.get("sources") or []': "knowledge_answer 消费内部 end_event，sources 是固定字段。",
    'final_event.get("retrieval") or {}': "knowledge_answer 消费内部 end_event，retrieval 是固定字段。",
    'final_event.get("stage_timings_ms") or {}': "knowledge_answer 消费内部 end_event，stage_timings_ms 是固定字段。",
    'final_event.get("slowest_stage") or {}': "knowledge_answer 消费内部 end_event，slowest_stage 是固定字段。",
    "retrieval_payload = retrieval or {}": "record_query_trace() 由 RAGQueryContext.record_trace() 单一路径调用，retrieval 是固定诊断字段。",
    "intent_payload = intent or {}": "record_query_trace() 由 RAGQueryContext.record_trace() 单一路径调用，intent 是固定诊断字段。",
    "intent: dict[str, Any] | None": "record_query_trace() 消费主链路固定 intent payload，不保留 None 分支。",
    "retrieval: dict[str, Any] | None": "record_query_trace() 消费主链路固定 retrieval payload，不保留 None 分支。",
    'getattr(scenario, "scenario_id"': "record_query_trace() 消费已解析的 ScenarioDefinition，应直接读取 scenario 字段。",
    'getattr(scenario, "display_name"': "record_query_trace() 消费已解析的 ScenarioDefinition，应直接读取 scenario 字段。",
    'return (text or "")[:max_chars]': "_safe_preview() 只由 record_query_trace() 传入字符串 answer，不保留 None 兜底。",
}
PIPELINE_CONTEXT_CONTRACT_FILES = (
    "qa_core/pipeline/events.py",
    "qa_core/pipeline/runtime.py",
    "qa_core/pipeline/rag.py",
    "qa_core/pipeline/steps.py",
    "qa_core/pipeline/context.py",
    "qa_core/pipeline/citations.py",
    "qa_core/pipeline/query_variants.py",
    "qa_core/retrieval/results.py",
    "qa_core/retrieval/ranking.py",
    "qa_core/retrieval/store.py",
    "qa_core/intent/question_category.py",
    "qa_core/document_metadata.py",
    "qa_core/indexing/chunking.py",
    "qa_core/indexing/document_normalizer.py",
    "qa_core/quality/chunk.py",
    "qa_core/quality/conflicts.py",
    "qa_core/observability/langsmith_adapter.py",
)
PROMPT_SELECTOR_CONTRACT_FRAGMENTS = {
    "from qa_core.config.settings import get_settings": "Prompt selector 必须消费上游已解析的 ScenarioDefinition，不读取全局 settings 做默认展示。",
    "settings = get_settings()": "Prompt selector 必须消费上游已解析的 ScenarioDefinition，不读取全局 settings 做默认展示。",
    "ScenarioDefinition | None = None": "Prompt selector 的 scenario 是主链路固定上下文，不保留 None 分支。",
    "query: str | None = None": "Prompt selector 的 query 由主链路固定传入，应是必填 str。",
    'infer_question_category(query or "")': "Prompt selector 接收必填 query: str，直接传给 infer_question_category(query)。",
    "def _scenario_prompt_context(scenario=None)": "Prompt selector 的场景上下文必须由明确 ScenarioDefinition 构造。",
    "if scenario else": "Prompt selector 不保留默认展示兜底，场景变量来自 ScenarioDefinition。",
}


@dataclass(frozen=True)
class GuardrailIssue:
    """一条守护检查问题。

    调用顺序：命令行入口 -> GuardrailIssue。
    """

    file: Path
    line: int
    message: str

    def format(self) -> str:
        """返回适合终端输出的中文问题描述。

        调用顺序：命令行入口 -> GuardrailIssue.format()。
        """
        rel = self.file.relative_to(PROJECT_ROOT)
        return f"{rel}:{self.line}: {self.message}"


def iter_python_files() -> list[Path]:
    """返回需要扫描的 Python 文件列表。

    调用顺序：命令行入口 -> iter_python_files()。
    """
    files: list[Path] = []
    for item in PYTHON_SCAN_DIRS:
        path = PROJECT_ROOT / item
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return files


def attach_parents(tree: ast.AST) -> None:
    """给 AST 节点补 parent 属性，方便判断导入是否位于模块头部。

    调用顺序：命令行入口 -> attach_parents()。
    """
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            setattr(child, "parent", parent)


def import_module_name(node: ast.Import | ast.ImportFrom) -> str:
    """提取导入语句的模块名称。

    调用顺序：命令行入口 -> import_module_name()。
    """
    if isinstance(node, ast.ImportFrom):
        return node.module or ""
    if not node.names:
        return ""
    return node.names[0].name


def node_contains_import(node: ast.AST) -> bool:
    """判断某个 AST 子树中是否包含导入语句。

    调用顺序：命令行入口 -> node_contains_import()。
    """
    return any(isinstance(child, (ast.Import, ast.ImportFrom)) for child in ast.walk(node))


def check_python_file(path: Path) -> list[GuardrailIssue]:
    """检查单个 Python 文件的导入位置、fallback 导入和旧链路引用。

    调用顺序：命令行入口 -> check_python_file()。
    """
    issues: list[GuardrailIssue] = []
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    attach_parents(tree)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            parent = getattr(node, "parent", None)
            module_name = import_module_name(node)
            if not isinstance(parent, ast.Module) and not _is_allowed_runtime_import(path, module_name):
                issues.append(GuardrailIssue(path, node.lineno, "导入必须放在文件头部，不能写在函数、方法或分支内部。"))
            if module_name.split(".")[0] in FORBIDDEN_IMPORT_PREFIXES:
                issues.append(GuardrailIssue(path, node.lineno, f"禁止引用旧链路模块：{module_name}"))
        elif isinstance(node, ast.Try):
            catches_import_error = any(
                isinstance(handler.type, ast.Name) and handler.type.id == "ImportError"
                for handler in node.handlers
                if handler.type is not None
            )
            if catches_import_error and node_contains_import(node):
                issues.append(GuardrailIssue(path, node.lineno, "禁止使用 ImportError fallback 导入；缺依赖应修正 requirements 和环境。"))

    if path.name != "check_project_guardrails.py":
        for token in FORBIDDEN_ENV_TOKENS:
            if token in source:
                issues.append(GuardrailIssue(path, 1, f"禁止在当前主链路代码中恢复旧配置或旧检索开关：{token}"))
    return issues


def check_schema_bootstrap_boundary() -> list[GuardrailIssue]:
    """检查 MySQL DDL 只存在于显式 runtime schema SQL 文件。

    调用顺序：命令行入口 -> check_schema_bootstrap_boundary()。
    """
    issues: list[GuardrailIssue] = []
    schema_path = PROJECT_ROOT / REQUIRED_MYSQL_SCHEMA_FILE
    if not schema_path.exists():
        issues.append(
            GuardrailIssue(
                schema_path,
                1,
                "MySQL schema 必须集中在 qa_core/storage/runtime_schema.sql，不能退回 Python 硬编码 DDL。",
            )
        )

    migration_dir = PROJECT_ROOT / FORBIDDEN_MYSQL_MIGRATION_DIR
    if migration_dir.exists():
        for path in sorted(migration_dir.glob("*.sql")):
            issues.append(
                GuardrailIssue(
                    path,
                    1,
                    "当前项目不维护兼容迁移文件；请把启动期表结构集中到 runtime_schema.sql。",
                )
            )

    for path in iter_python_files():
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")

        if rel not in SCHEMA_DDL_ALLOWED_FILES | SCHEMA_TEXT_EXEMPT_FILES:
            for pattern in DDL_PATTERNS:
                match = re.search(pattern, source, flags=re.IGNORECASE)
                if match:
                    issues.append(
                        GuardrailIssue(
                            path,
                            source.count("\n", 0, match.start()) + 1,
                            "MySQL DDL 只能写在 qa_core/storage/runtime_schema.sql，由 bootstrap 统一调用。",
                        )
                    )

        if rel not in SCHEMA_BOOTSTRAP_ALLOWED_FILES | SCHEMA_TEXT_EXEMPT_FILES:
            for pattern in (r"\bdef\s+ensure_tables?\s*\(", r"\.ensure_tables?\s*\("):
                match = re.search(pattern, source)
                if match:
                    issues.append(
                        GuardrailIssue(
                            path,
                            source.count("\n", 0, match.start()) + 1,
                            "业务代码不能隐式 ensure_table(s)；请在启动或脚本入口调用 bootstrap_mysql_schema()。",
                        )
                    )

        if rel not in SCHEMA_DDL_ALLOWED_FILES | SCHEMA_BOOTSTRAP_ALLOWED_FILES | SCHEMA_TEXT_EXEMPT_FILES:
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.module != "qa_core.storage.mysql_schema":
                    continue
                for alias in node.names:
                    if alias.name.startswith("create_") and "table" in alias.name:
                        issues.append(
                            GuardrailIssue(
                                path,
                                node.lineno,
                                "业务代码不能直接导入 schema 建表函数；请通过 runtime_schema.sql 和 bootstrap 统一初始化。",
                            )
                        )
    return issues


def _function_defs(tree: ast.AST, name: str) -> list[ast.FunctionDef]:
    """返回 AST 中指定名称的函数或方法定义。

    调用顺序：命令行入口 -> _function_defs()。
    """
    return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]


def _arg_names(node: ast.FunctionDef) -> set[str]:
    """返回函数的位置参数和 keyword-only 参数名。

    调用顺序：命令行入口 -> _arg_names()。
    """
    return {arg.arg for arg in (*node.args.args, *node.args.kwonlyargs)}


def _calls_name(node: ast.AST, name: str) -> list[ast.Call]:
    """返回节点内部对指定函数名或同名属性的调用。

    调用顺序：命令行入口 -> _calls_name()。
    """
    calls: list[ast.Call] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name) and func.id == name:
            calls.append(child)
        elif isinstance(func, ast.Attribute) and func.attr == name:
            calls.append(child)
    return calls


def check_source_filter_validation_boundary() -> list[GuardrailIssue]:
    """检查 source 白名单校验只停留在入口层，检索底层只构造表达式。

    调用顺序：命令行入口 -> check_source_filter_validation_boundary()。
    """
    issues: list[GuardrailIssue] = []

    filters_path = PROJECT_ROOT / "qa_core/retrieval/filters.py"
    filters_text = filters_path.read_text(encoding="utf-8")
    filters_tree = ast.parse(filters_text, filename=str(filters_path))
    for fragment in ("valid_sources: list[str] | None", "valid_sources is not None", "None 表示不校验"):
        line = _line_number(filters_text, fragment)
        if line:
            issues.append(
                GuardrailIssue(
                    filters_path,
                    line,
                    "validate_source_filter() 必须接收当前场景的明确 source 白名单，不保留 None 跳过校验分支。",
                )
            )
    for node in _function_defs(filters_tree, "build_source_expr"):
        if "valid_sources" in _arg_names(node):
            issues.append(
                GuardrailIssue(
                    filters_path,
                    node.lineno,
                    "build_source_expr() 只负责构建 Milvus expr，不再接收 valid_sources。",
                )
            )
        for call in _calls_name(node, "validate_source_filter"):
            issues.append(
                GuardrailIssue(
                    filters_path,
                    call.lineno,
                    "source_filter 白名单校验应在入口层完成，build_source_expr() 内不要重复校验。",
                )
            )

    store_path = PROJECT_ROOT / "qa_core/retrieval/store.py"
    store_tree = ast.parse(store_path.read_text(encoding="utf-8"), filename=str(store_path))
    for function_name in ("search", "search_many"):
        for node in _function_defs(store_tree, function_name):
            if "valid_sources" in _arg_names(node):
                issues.append(
                    GuardrailIssue(
                        store_path,
                        node.lineno,
                        f"MilvusHybridStore.{function_name}() 不再透传 valid_sources；入口层已经完成 source 校验。",
                    )
                )
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                if any(keyword.arg == "valid_sources" for keyword in call.keywords):
                    issues.append(
                        GuardrailIssue(
                            store_path,
                            call.lineno,
                            f"MilvusHybridStore.{function_name}() 内不要继续传递 valid_sources。",
                        )
                    )

    steps_path = PROJECT_ROOT / "qa_core/pipeline/steps.py"
    steps_tree = ast.parse(steps_path.read_text(encoding="utf-8"), filename=str(steps_path))
    decide_route_calls_validate_source_filter = False
    for node in _function_defs(steps_tree, "decide_route"):
        decide_route_calls_validate_source_filter = bool(_calls_name(node, "validate_source_filter"))
    for node in _function_defs(steps_tree, "prepare_retrieval"):
        for constant in ast.walk(node):
            if isinstance(constant, ast.Constant) and constant.value == "validate_source":
                issues.append(
                    GuardrailIssue(
                        steps_path,
                        constant.lineno,
                        "prepare_retrieval() 由 decide_route() 调用后执行，不要重复做 validate_source 阶段。",
                    )
                )
        for call in _calls_name(node, "validate_source"):
            issues.append(
                GuardrailIssue(
                    steps_path,
                    call.lineno,
                    "prepare_retrieval() 不应重复调用 validate_source。",
                )
            )
    if not decide_route_calls_validate_source_filter:
        issues.append(
            GuardrailIssue(
                steps_path,
                1,
                "decide_route() 必须直接调用 validate_source_filter() 完成入口 source 校验，不再通过 QAService 回调转发。",
            )
        )

    runtime_path = PROJECT_ROOT / "qa_core/pipeline/runtime.py"
    runtime_tree = ast.parse(runtime_path.read_text(encoding="utf-8"), filename=str(runtime_path))
    for node in _function_defs(runtime_tree, "create_query_context"):
        if "validate_source" in _arg_names(node):
            issues.append(
                GuardrailIssue(
                    runtime_path,
                    node.lineno,
                    "create_query_context() 不再接收 validate_source 回调；source 校验属于 decide_route()。",
                )
            )
    for node in ast.walk(runtime_tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "validate_source":
            issues.append(
                GuardrailIssue(
                    runtime_path,
                    node.lineno,
                    "RAGQueryContext 不应保存 validate_source 回调；请求上下文只保存状态。",
                )
            )

    rag_path = PROJECT_ROOT / "qa_core/pipeline/rag.py"
    rag_tree = ast.parse(rag_path.read_text(encoding="utf-8"), filename=str(rag_path))
    for function_name in ("stream_query", "debug_retrieval"):
        for node in _function_defs(rag_tree, function_name):
            if "validate_source" in _arg_names(node):
                issues.append(
                    GuardrailIssue(
                        rag_path,
                        node.lineno,
                        f"{function_name}() 不再接收 validate_source 回调；入口校验在 decide_route() 中完成。",
                    )
                )

    service_path = PROJECT_ROOT / "qa_core/application/service.py"
    service_tree = ast.parse(service_path.read_text(encoding="utf-8"), filename=str(service_path))
    for node in ast.walk(service_tree):
        if isinstance(node, ast.ImportFrom) and node.module == "qa_core.config.settings":
            if any(alias.name == "get_settings" for alias in node.names):
                issues.append(
                    GuardrailIssue(
                        service_path,
                        node.lineno,
                        "QAService 不直接读取全局 settings；配置由下游依赖在各自边界读取。",
                    )
                )
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == "settings"
                ):
                    issues.append(
                        GuardrailIssue(
                            service_path,
                            target.lineno,
                            "QAService 不保存未使用的 settings 状态；服务层只持有实际编排依赖。",
                        )
                    )
    for node in _function_defs(service_tree, "validate_source"):
        issues.append(
            GuardrailIssue(
                service_path,
                node.lineno,
                "QAService 不再保留 validate_source() 薄包装；source 校验由 pipeline.steps.decide_route() 直接执行。",
            )
        )

    for rel in SOURCE_FILTER_BOUNDARY_FILES:
        path = PROJECT_ROOT / rel
        text = path.read_text(encoding="utf-8")
        for fragment in (
            "valid_sources=context.scenario.valid_sources",
            "valid_sources=valid_sources",
            "context.validate_source",
            "validate_source=validate_source",
            "self.validate_source",
        ):
            line = _line_number(text, fragment)
            if line:
                issues.append(
                    GuardrailIssue(
                        path,
                        line,
                        "source 白名单不要向检索底层透传；保留入口校验即可。",
                    )
                )
    return issues


def check_intent_retrieval_boundary() -> list[GuardrailIssue]:
    """检查直答路由、检索意图和检索计划的职责边界。

    调用顺序：命令行入口 -> check_intent_retrieval_boundary()。
    """
    issues: list[GuardrailIssue] = []

    classifier_path = PROJECT_ROOT / "qa_core/intent/classifier.py"
    classifier_text = classifier_path.read_text(encoding="utf-8")
    classifier_tree = ast.parse(classifier_text, filename=str(classifier_path))
    for fragment in (
        "from qa_core.config.settings import get_settings",
        "settings = get_settings()",
        "ScenarioDefinition | None = None",
        "if scenario is None",
        "assistant_name = scenario.assistant_name if scenario else",
        "business_domain = scenario.business_domain if scenario else",
        "support_contact = scenario.support_contact if scenario else",
    ):
        line = _line_number(classifier_text, fragment)
        if line:
            issues.append(
                GuardrailIssue(
                    classifier_path,
                    line,
                    "意图分类必须消费上游已解析的 ScenarioDefinition，不保留默认场景或 settings 回退。",
                )
            )
    for node in _function_defs(classifier_tree, "classify_intent"):
        for call in _calls_name(node, "classify_direct_intent"):
            issues.append(
                GuardrailIssue(
                    classifier_path,
                    call.lineno,
                    "classify_intent() 只处理检索类意图，不应重复调用 classify_direct_intent()。",
                )
            )

    for path in iter_python_files():
        if path.name == "check_project_guardrails.py":
            continue
        source = path.read_text(encoding="utf-8")
        fragment = "direct_intent and direct_intent.direct_answer"
        line = _line_number(source, fragment)
        if line:
            issues.append(
                GuardrailIssue(
                    path,
                    line,
                    "classify_direct_intent() 返回 None 或带 direct_answer 的 IntentResult；调用处只需判断 direct_intent。",
                )
            )

    strategy_path = PROJECT_ROOT / "qa_core/retrieval/strategy.py"
    strategy_tree = ast.parse(strategy_path.read_text(encoding="utf-8"), filename=str(strategy_path))
    for node in _function_defs(strategy_tree, "build_retrieval_plan"):
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr == "direct_answer":
                issues.append(
                    GuardrailIssue(
                        strategy_path,
                        child.lineno,
                        "build_retrieval_plan() 默认只接收检索类 IntentResult，不再判断 direct_answer。",
                    )
                )
            if isinstance(child, ast.Raise):
                issues.append(
                    GuardrailIssue(
                        strategy_path,
                        child.lineno,
                        "build_retrieval_plan() 不再保留直答意图防御分支；上游 route=retrieval 才会调用它。",
                    )
                )
    return issues


def check_pipeline_context_contract() -> list[GuardrailIssue]:
    """检查 Pipeline 调试和内部适配层不重复兜底上下文固定字段。

    调用顺序：命令行入口 -> check_pipeline_context_contract()。
    """
    issues: list[GuardrailIssue] = []
    for rel in PIPELINE_CONTEXT_CONTRACT_FILES:
        path = PROJECT_ROOT / rel
        text = path.read_text(encoding="utf-8")
        for fragment, message in PIPELINE_CONTEXT_CONTRACT_FRAGMENTS.items():
            line = _line_number(text, fragment)
            if line:
                issues.append(GuardrailIssue(path, line, message))
    return issues


def check_prompt_selector_contract() -> list[GuardrailIssue]:
    """检查 Prompt selector 不保留默认场景回退。

    调用顺序：命令行入口 -> check_prompt_selector_contract()。
    """
    issues: list[GuardrailIssue] = []
    path = PROJECT_ROOT / "qa_core/prompts/selector.py"
    text = path.read_text(encoding="utf-8")
    for fragment, message in PROMPT_SELECTOR_CONTRACT_FRAGMENTS.items():
        line = _line_number(text, fragment)
        if line:
            issues.append(GuardrailIssue(path, line, message))
    return issues


def _is_allowed_runtime_import(path: Path, module_name: str) -> bool:
    """判断是否属于极少数允许的运行时导入。

    默认规则仍然要求 import 放在文件头。这里仅允许少数重依赖或可选 SDK 在真正执行
    对应路径时加载，避免 `--help`、轻量测试或状态查询阶段提前触发 BGE/torch/Milvus/LangSmith 初始化。

    调用顺序：命令行入口 -> _is_allowed_runtime_import()。
    """
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    return module_name in ALLOWED_RUNTIME_IMPORTS.get(rel, set())


def check_legacy_paths() -> list[GuardrailIssue]:
    """检查旧链路目录和旧入口是否被重新创建。

    调用顺序：命令行入口 -> check_legacy_paths()。
    """
    issues: list[GuardrailIssue] = []
    for rel_path in LEGACY_PATHS:
        path = PROJECT_ROOT / rel_path
        if path.exists():
            issues.append(GuardrailIssue(path, 1, "旧链路文件或目录不应重新出现在工程中。"))
    for rel_path in FORBIDDEN_ARTIFACT_PATHS:
        path = PROJECT_ROOT / rel_path
        if path.exists():
            issues.append(GuardrailIssue(path, 1, "运行报告不能写在 scripts 目录下；请统一输出到项目根目录 reports/。"))
    for rel_path in SOURCE_ARTIFACT_SCAN_DIRS:
        scan_root = PROJECT_ROOT / rel_path
        if not scan_root.exists():
            continue
        for path in scan_root.rglob(".venv"):
            if path.is_dir():
                issues.append(GuardrailIssue(path, 1, "源码和跟敲目录不能包含机器相关的局部虚拟环境。"))
    return issues


def check_requirements() -> list[GuardrailIssue]:
    """检查直接依赖是否满足当前项目约束。

    调用顺序：命令行入口 -> check_requirements()。
    """
    path = PROJECT_ROOT / "requirements.txt"
    issues: list[GuardrailIssue] = []
    if not path.exists():
        issues.append(GuardrailIssue(path, 1, "requirements.txt 不存在。"))
    if issues:
        return issues
    for index, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "==" not in line:
            issues.append(GuardrailIssue(path, index, "依赖必须使用 == 锁定版本。"))
        package_name = re.split(r"==|>=|<=|~=|>|<", line, maxsplit=1)[0].strip().lower()
        if package_name in FORBIDDEN_DIRECT_REQUIREMENTS:
            issues.append(GuardrailIssue(path, index, FORBIDDEN_DIRECT_REQUIREMENTS[package_name]))
    return issues


def check_secret_hygiene() -> list[GuardrailIssue]:
    """检查本地密钥和运行产物不会被默认提交。

    该检查不读取真实 `.env` 内容，只确认 `.gitignore` 会忽略敏感文件，并确认
    环境模板只保留占位符。这样既能保护本地 Key，又不会把密钥打印到终端。

    调用顺序：命令行入口 -> check_secret_hygiene()。
    """
    issues: list[GuardrailIssue] = []
    gitignore = PROJECT_ROOT / ".gitignore"
    env_examples = [
        PROJECT_ROOT / ".env.local.example",
        PROJECT_ROOT / ".env.compose.example",
    ]
    if not gitignore.exists():
        return [GuardrailIssue(gitignore, 1, ".gitignore 不存在，真实 .env、logs、reports 可能被误提交。")]
    content = gitignore.read_text(encoding="utf-8")
    for pattern in REQUIRED_GITIGNORE_PATTERNS:
        if pattern not in content:
            issues.append(GuardrailIssue(gitignore, 1, f".gitignore 必须包含敏感/运行产物规则：{pattern}"))
    for env_example in env_examples:
        if env_example.exists() and "sk-" in env_example.read_text(encoding="utf-8"):
            issues.append(GuardrailIssue(env_example, 1, f"{env_example.name} 只能写占位符，不能出现真实 Key 形态。"))
    dockerignore = PROJECT_ROOT / ".dockerignore"
    if dockerignore.exists() and "!.env.example" in dockerignore.read_text(encoding="utf-8"):
        issues.append(GuardrailIssue(dockerignore, 1, ".dockerignore 不应继续放行已删除的 .env.example。"))
    return issues


def check_env_file_contract() -> list[GuardrailIssue]:
    """检查 Docker 和本机 env 文件边界，避免两种运行视角再次混在一起。

    调用顺序：命令行入口 -> check_env_file_contract()。
    """
    issues: list[GuardrailIssue] = []
    compose_file = PROJECT_ROOT / "docker-compose.yml"
    if compose_file.exists():
        text = compose_file.read_text(encoding="utf-8")
        if "${ENV_FILE:-.env.compose}" not in text:
            issues.append(GuardrailIssue(compose_file, 1, "api.env_file 默认值必须是 .env.compose，不能回退到 .env。"))
        for key in COMPOSE_ENV_OVERRIDE_KEYS:
            if re.search(rf"^\s+{re.escape(key)}\s*:", text, flags=re.MULTILINE):
                issues.append(GuardrailIssue(compose_file, 1, f"api.environment 不应硬编码 {key}，应从 .env.compose 读取。"))

    env_files = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / ".env.local.example",
        PROJECT_ROOT / ".env.compose",
        PROJECT_ROOT / ".env.compose.example",
    ]
    parsed = {path.name: _parse_env_keys(path) for path in env_files if path.exists()}
    if ".env.local.example" in parsed and ".env" in parsed:
        if set(parsed[".env"].keys()) != set(parsed[".env.local.example"].keys()):
            issues.append(GuardrailIssue(PROJECT_ROOT / ".env", 1, ".env 的配置项必须与 .env.local.example 保持一致。"))
    if ".env.compose.example" in parsed and ".env.compose" in parsed:
        if set(parsed[".env.compose"].keys()) != set(parsed[".env.compose.example"].keys()):
            issues.append(GuardrailIssue(PROJECT_ROOT / ".env.compose", 1, ".env.compose 的配置项必须与 .env.compose.example 保持一致。"))
    return issues


def _parse_env_keys(path: Path) -> dict[str, str]:
    """读取 env 文件键值，不输出真实值。

    调用顺序：命令行入口 -> _parse_env_keys()。
    """
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _faq_text(row: dict[str, str], *names: str) -> str:
    """按兼容列名读取 FAQ 单元格。

    调用顺序：命令行入口 -> _faq_text()。
    """
    return str(next((row.get(name) for name in names if row.get(name)), "")).strip()


def check_scenario_packages() -> list[GuardrailIssue]:
    """检查场景包配置是否满足多场景项目约束。

    场景包是当前项目扩展业务背景的唯一入口。这里把常见人为错误前置拦住：
    collection 重名、source 顺序不清、FAQ 分类写错、文档目录缺失、正则不可编译。
    这些问题如果等到线上检索时才发现，排查成本会比在守卫阶段高很多。

    调用顺序：命令行入口 -> check_scenario_packages()。
    """
    issues: list[GuardrailIssue] = []
    scenario_root = PROJECT_ROOT / "scenarios"
    if not scenario_root.exists():
        return [GuardrailIssue(scenario_root, 1, "scenarios 目录不存在，无法加载合同场景配置。")]

    faq_collections: Counter[str] = Counter()
    doc_collections: Counter[str] = Counter()
    scenario_ids: Counter[str] = Counter()
    for config_path in sorted(scenario_root.glob("*/scenario.toml")):
        try:
            payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            issues.append(GuardrailIssue(config_path, getattr(exc, "lineno", 1) or 1, f"scenario.toml 解析失败：{exc}"))
            continue

        missing = [name for name in SCENARIO_REQUIRED_FIELDS if not payload.get(name)]
        for field_name in missing:
            issues.append(GuardrailIssue(config_path, 1, f"场景配置缺少必填字段：{field_name}"))
        if missing:
            continue

        scenario_id = str(payload["scenario_id"]).strip()
        scenario_ids[scenario_id] += 1
        if scenario_id not in FROZEN_SCENARIO_IDS:
            issues.append(
                GuardrailIssue(
                    config_path,
                    1,
                    f"业务场景已经冻结，不能继续新增未评审场景：{scenario_id}",
                )
            )
        if scenario_id != config_path.parent.name:
            issues.append(GuardrailIssue(config_path, 1, "scenario_id 必须与目录名一致，避免版本清单和资料目录错位。"))

        valid_sources = [str(item).strip() for item in payload.get("valid_sources", []) if str(item).strip()]
        if len(valid_sources) != len(set(valid_sources)):
            issues.append(GuardrailIssue(config_path, 1, "valid_sources 不能重复；它同时决定 source 白名单和匹配优先级。"))
        if not valid_sources:
            issues.append(GuardrailIssue(config_path, 1, "valid_sources 不能为空。"))

        source_labels = {str(key): str(value) for key, value in dict(payload.get("source_labels", {})).items()}
        source_patterns = {str(key): str(value) for key, value in dict(payload.get("source_patterns", {})).items()}
        for source in valid_sources:
            if source not in source_labels:
                issues.append(GuardrailIssue(config_path, 1, f"source_labels 缺少 {source} 的中文标签。"))
            if source not in source_patterns:
                issues.append(GuardrailIssue(config_path, 1, f"source_patterns 缺少 {source} 的推断规则。"))
        for source in set(source_labels) | set(source_patterns):
            if source not in valid_sources:
                issues.append(GuardrailIssue(config_path, 1, f"source 配置包含不在 valid_sources 中的分类：{source}"))
        for source, pattern in source_patterns.items():
            try:
                re.compile(pattern)
            except re.error as exc:
                issues.append(GuardrailIssue(config_path, 1, f"{source} 的 source_patterns 正则不可编译：{exc}"))

        faq_collection = str(payload["faq_collection"]).strip()
        doc_collection = str(payload["doc_collection"]).strip()
        faq_collections[faq_collection] += 1
        doc_collections[doc_collection] += 1
        if faq_collection == doc_collection:
            issues.append(GuardrailIssue(config_path, 1, "FAQ collection 和文档 collection 不能相同。"))

        faq_path = config_path.parent / "faq.csv"
        if not faq_path.exists():
            issues.append(GuardrailIssue(faq_path, 1, "场景 FAQ 文件不存在。"))
        else:
            issues.extend(_check_scenario_faq(faq_path, valid_sources))

        data_root = config_path.parent / "data"
        if not data_root.exists():
            issues.append(GuardrailIssue(data_root, 1, "场景 data 目录不存在。"))
        for source in valid_sources:
            source_dir = data_root / f"{source}_data"
            if not source_dir.exists():
                issues.append(GuardrailIssue(source_dir, 1, f"缺少 source 文档目录：{source}_data"))
                continue
            docs = [path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SCENARIO_DOC_SUFFIXES]
            if not docs:
                issues.append(GuardrailIssue(source_dir, 1, f"{source}_data 中没有可入库文档。"))

        suffixes = {
            path.suffix.lower()
            for path in data_root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SCENARIO_DOC_SUFFIXES
        }
        missing_suffixes = sorted(REQUIRED_SCENARIO_DOC_SUFFIXES - suffixes)
        if missing_suffixes:
            issues.append(
                GuardrailIssue(
                    data_root,
                    1,
                    f"冻结业务场景必须保留多格式样例，当前缺少：{missing_suffixes}",
                )
            )

    duplicate_scenarios = [item for item, count in scenario_ids.items() if count > 1]
    duplicate_faq_collections = [item for item, count in faq_collections.items() if count > 1]
    duplicate_doc_collections = [item for item, count in doc_collections.items() if count > 1]
    missing_frozen_scenarios = sorted(FROZEN_SCENARIO_IDS - set(scenario_ids))
    if missing_frozen_scenarios:
        issues.append(GuardrailIssue(scenario_root, 1, f"冻结场景缺失：{missing_frozen_scenarios}"))
    if duplicate_scenarios:
        issues.append(GuardrailIssue(scenario_root, 1, f"scenario_id 重复：{duplicate_scenarios}"))
    if duplicate_faq_collections:
        issues.append(GuardrailIssue(scenario_root, 1, f"FAQ collection 重复：{duplicate_faq_collections}"))
    if duplicate_doc_collections:
        issues.append(GuardrailIssue(scenario_root, 1, f"文档 collection 重复：{duplicate_doc_collections}"))
    return issues


def _check_scenario_faq(path: Path, valid_sources: list[str]) -> list[GuardrailIssue]:
    """检查单个场景 FAQ CSV 的字段、空值、重复和 source 合法性。

    调用顺序：命令行入口 -> _check_scenario_faq()。
    """
    issues: list[GuardrailIssue] = []
    questions: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return [GuardrailIssue(path, 1, "FAQ CSV 缺少表头。")]
        for row_index, row in enumerate(reader, start=2):
            question = _faq_text(row, "question", "问题")
            answer = _faq_text(row, "answer", "答案")
            source = _faq_text(row, "source", "source_filter", "业务分类", "分类")
            if not question:
                issues.append(GuardrailIssue(path, row_index, "FAQ 问题不能为空。"))
            if not answer:
                issues.append(GuardrailIssue(path, row_index, "FAQ 答案不能为空。"))
            if not source:
                issues.append(GuardrailIssue(path, row_index, "FAQ source 不能为空。"))
            elif source not in valid_sources:
                issues.append(GuardrailIssue(path, row_index, f"FAQ source 不在当前场景 valid_sources 中：{source}"))
            if question:
                questions[question] += 1
            if source:
                source_counter[source] += 1
    for question, count in questions.items():
        if count > 1:
            issues.append(GuardrailIssue(path, 1, f"FAQ 问题重复：{question}"))
    missing_sources = [source for source in valid_sources if source_counter[source] == 0]
    if missing_sources:
        issues.append(GuardrailIssue(path, 1, f"FAQ 未覆盖这些 source：{missing_sources}"))
    return issues


def check_public_docs_do_not_expose_internal_markers() -> list[GuardrailIssue]:
    """检查正式讲义和生成站点不暴露跟敲/内部课程实现标记。

    调用顺序：命令行入口 -> check_public_docs_do_not_expose_internal_markers()。
    """
    issues: list[GuardrailIssue] = []
    suffixes = {".md", ".html"}
    for rel_dir in PUBLIC_DOC_SCAN_DIRS:
        scan_root = PROJECT_ROOT / rel_dir
        if not scan_root.exists():
            continue
        for path in sorted(scan_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if path.name == "search_index.json":
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            for fragment in PUBLIC_DOC_FORBIDDEN_FRAGMENTS:
                line_no = _line_number(content, fragment)
                if line_no:
                    issues.append(GuardrailIssue(path, line_no, f"正式讲义不能暴露内部课程实现信息：{fragment}"))
    return issues


def check_public_text_has_no_role_terms() -> list[GuardrailIssue]:
    """检查仓库资料和源码不出现角色化称呼。

    调用顺序：命令行入口 -> check_public_text_has_no_role_terms()。
    """
    issues: list[GuardrailIssue] = []
    suffixes = {".md", ".py", ".toml", ".yml", ".yaml", ".html", ".js", ".css", ".txt", ".json"}
    for rel_path in PUBLIC_TEXT_SCAN_DIRS:
        root = PROJECT_ROOT / rel_path
        paths = [root] if root.is_file() else sorted(root.rglob("*")) if root.exists() else []
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if path.name in {"check_project_guardrails.py", "mermaid.min.js", "tex-mml-chtml.js"}:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            for term in FORBIDDEN_ROLE_TERMS:
                line_no = _line_number(content, term)
                if line_no:
                    issues.append(GuardrailIssue(path, line_no, "仓库资料和源码不应出现角色化称呼，请改成中性表述。"))
    return issues


def _line_number(content: str, fragment: str) -> int:
    """返回片段首次出现行号；不存在则返回 0。

    调用顺序：命令行入口 -> _line_number()。
    """
    index = content.find(fragment)
    if index < 0:
        return 0
    return content.count("\n", 0, index) + 1

def main() -> None:
    """执行全部守护检查。

    调用顺序：命令行入口 -> main()。
    """
    issues: list[GuardrailIssue] = []
    issues.extend(check_legacy_paths())
    issues.extend(check_requirements())
    issues.extend(check_secret_hygiene())
    issues.extend(check_env_file_contract())
    issues.extend(check_scenario_packages())
    issues.extend(check_public_docs_do_not_expose_internal_markers())
    issues.extend(check_public_text_has_no_role_terms())
    issues.extend(check_schema_bootstrap_boundary())
    issues.extend(check_source_filter_validation_boundary())
    issues.extend(check_intent_retrieval_boundary())
    issues.extend(check_pipeline_context_contract())
    issues.extend(check_prompt_selector_contract())
    for path in iter_python_files():
        issues.extend(check_python_file(path))

    if issues:
        print("项目守护检查失败：")
        for issue in issues:
            print(f"- {issue.format()}")
        sys.exit(1)
    print("项目守护检查通过：导入位置、旧链路、fallback 导入、依赖版本、密钥卫生、schema bootstrap、source 校验边界、意图分层边界、上下文契约和冻结场景包均符合当前约束。")


if __name__ == "__main__":
    main()
