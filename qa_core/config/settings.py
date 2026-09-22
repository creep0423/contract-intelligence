"""Contract Intelligence 运行时配置，通过进程环境变量和本机 .env 加载。"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _discover_project_root() -> Path:
    """根据产品源码与配置目录定位仓库根目录。

    调用顺序：启动配置或前置校验 -> _discover_project_root()。
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "qa_core").exists() and (parent / "scenarios").exists() and (parent / "requirements.txt").exists():
            return parent
    return current.parents[2]


PROJECT_ROOT = _discover_project_root()
MODEL_ROOT = PROJECT_ROOT / "models"
SUPPORTED_APP_ENVIRONMENTS = frozenset({"dev", "test", "uat", "staging", "prd", "prod", "production"})
TRUSTED_IDENTITY_ENVIRONMENTS = frozenset({"uat", "staging", "prd", "prod", "production"})
# 浏览器同源演示身份只允许在非生产环境启用：staging/prd/prod/production 必须走可信上游网关注入。
DEMO_IDENTITY_ENVIRONMENTS = frozenset({"dev", "test", "uat"})


def _resolve_project_relative_path(value: str) -> str:
    """把 .env 中的相对路径固定解析到仓库根目录。

    调用顺序：启动配置或前置校验 -> _resolve_project_relative_path()。
    """
    if not value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


class Settings(BaseSettings):
    """LangChain + Milvus 主链路的运行时配置。仅负责读取配置值，外部依赖由 `validate_runtime_environment()` 统一校验。

    调用顺序：启动配置或前置校验 -> Settings。
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 展示品牌与工程内部标识严格解耦：以下字段只允许用于 UI、HTML Metadata 和 OpenAPI Metadata。
    # 不得用于数据库、缓存、向量库、权限域、文件路径、API route 或 Benchmark 标识。
    product_display_name: str = Field(default="知契", validation_alias="PRODUCT_DISPLAY_NAME")
    product_english_descriptor: str = Field(
        default="Enterprise Intelligence",
        validation_alias="PRODUCT_ENGLISH_DESCRIPTOR",
    )
    contract_product_descriptor: str = Field(
        default="Contract Intelligence",
        validation_alias="CONTRACT_PRODUCT_DESCRIPTOR",
    )
    product_description: str = Field(
        default="企业合同履约与风险智能",
        validation_alias="PRODUCT_DESCRIPTION",
    )
    product_tagline: str = Field(
        default="从原文证据，到业务判断。",
        validation_alias="PRODUCT_TAGLINE",
    )
    contract_tagline: str = Field(
        default="看清条款，掌握履约，提前发现风险。",
        validation_alias="CONTRACT_TAGLINE",
    )
    product_meta_description: str = Field(
        default="面向企业合同履约、义务、风险与原文证据分析的智能工作平台。",
        validation_alias="PRODUCT_META_DESCRIPTION",
    )

    # 工程身份：保留原值，不参与展示品牌迁移。
    app_name: str = "ContractIntelligence"
    env: str = Field(default="dev", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    active_scenario_id: str = Field(default="tender_contract_risk", validation_alias="ACTIVE_SCENARIO_ID")
    scenario_config_dir: str = Field(default=str(PROJECT_ROOT / "scenarios"), validation_alias="SCENARIO_CONFIG_DIR")

    # MySQL 保存聊天历史、摘要、反馈、知识库版本控制面和入库 manifest，启动前必须可连接。
    mysql_host: str = Field(default="localhost", validation_alias="MYSQL_HOST")
    mysql_port: int = Field(default=3306, validation_alias="MYSQL_PORT")
    mysql_user: str = Field(default="root", validation_alias="MYSQL_USER")
    mysql_password: str = Field(default="", validation_alias="MYSQL_PASSWORD")
    mysql_database: str = Field(default="subjects_kg", validation_alias="MYSQL_DATABASE")

    # FAQ/文档混合检索核心，要求 Milvus 2.5+ 支持 BM25BuiltInFunction
    milvus_uri: str = Field(default="http://localhost:19530", validation_alias="MILVUS_URI")
    milvus_database: str = Field(default="", validation_alias="MILVUS_DATABASE")
    milvus_query_timeout_seconds: float = Field(
        default=15.0,
        validation_alias="MILVUS_QUERY_TIMEOUT_SECONDS",
        ge=1.0,
        le=120.0,
    )

    # V1 企业级缓存：Docker 交付环境显式开启；纯逻辑单测默认不依赖 Redis。
    cache_enabled: bool = Field(default=False, validation_alias="CACHE_ENABLED")
    redis_host: str = Field(default="localhost", validation_alias="REDIS_HOST")
    redis_port: int = Field(default=6379, validation_alias="REDIS_PORT")
    redis_db: int = Field(default=0, validation_alias="REDIS_DB")
    redis_password: str = Field(default="", validation_alias="REDIS_PASSWORD")
    redis_socket_timeout: float = Field(default=2.0, validation_alias="REDIS_SOCKET_TIMEOUT")
    cache_key_prefix: str = Field(default="kf:v1", validation_alias="CACHE_KEY_PREFIX")
    cache_namespace_l1_ttl_seconds: int = Field(default=3, validation_alias="CACHE_NAMESPACE_L1_TTL_SECONDS")
    cache_faq_ttl_seconds: int = Field(default=1800, validation_alias="CACHE_FAQ_TTL_SECONDS")
    cache_doc_ttl_seconds: int = Field(default=900, validation_alias="CACHE_DOC_TTL_SECONDS")
    cache_embedding_ttl_seconds: int = Field(default=3600, validation_alias="CACHE_EMBEDDING_TTL_SECONDS")

    # 通过 OpenAI-compatible 接口接入 DashScope，LangChain ChatOpenAI 统一调用
    llm_model: str = Field(default="qwen-plus", validation_alias="LLM_MODEL")
    llm_api_key: str = Field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
        validation_alias="DASHSCOPE_API_KEY",
    )
    llm_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias="DASHSCOPE_BASE_URL",
    )
    llm_temperature: float = Field(default=0.1, validation_alias="LLM_TEMPERATURE")
    llm_timeout: float = Field(default=60.0, validation_alias="LLM_TIMEOUT")

    # LangSmith 是可选 Trace 旁路；本地 Evaluation/Gate 不依赖它。
    langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: str = Field(default="", validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="contract-intelligence", validation_alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", validation_alias="LANGSMITH_ENDPOINT")

    # 默认使用项目 models/ 下的本地模型；外置模型目录可通过环境变量覆盖。
    embedding_model_path: str = Field(default=str(MODEL_ROOT / "bge-m3"), validation_alias="EMBEDDING_MODEL_PATH")
    reranker_model_path: str = Field(default=str(MODEL_ROOT / "bge-reranker-large"), validation_alias="RERANKER_MODEL_PATH")
    intent_model_path: str = Field(default=str(MODEL_ROOT / "bert_intent_classifier_v1"), validation_alias="INTENT_MODEL_PATH")
    intent_model_version: str = Field(default="bert-intent-v1", validation_alias="INTENT_MODEL_VERSION")
    intent_model_device: str = Field(default="cpu", validation_alias="INTENT_MODEL_DEVICE")
    intent_model_max_length: int = Field(default=64, validation_alias="INTENT_MODEL_MAX_LENGTH")
    embedding_model_version: str = Field(default="bge-m3-local-v1", validation_alias="EMBEDDING_MODEL_VERSION")
    reranker_model_version: str = Field(default="bge-reranker-large-local-v1", validation_alias="RERANKER_MODEL_VERSION")
    chunk_schema_version: str = Field(default="parent_child_validity_v2", validation_alias="CHUNK_SCHEMA_VERSION")
    document_parser_backend: str = Field(default="native", validation_alias="DOCUMENT_PARSER_BACKEND")

    customer_service_phone: str = Field(default="12345678", validation_alias="CUSTOMER_SERVICE_PHONE")
    admin_api_token: str = Field(default="", validation_alias="ADMIN_API_TOKEN")
    contract_trusted_upstream_token: str = Field(
        default="",
        validation_alias="CONTRACT_TRUSTED_UPSTREAM_TOKEN",
    )
    # 浏览器同源演示会话：只在非生产环境（dev/test/uat）显式开启时生效。
    # 开启后页面响应会下发一个服务端签名的会话票据，浏览器凭该票据以固定的演示身份访问合同接口；
    # 可信上游令牌始终留在服务端，任何情况下都不会发送到浏览器。
    contract_browser_demo_session: bool = Field(
        default=False,
        validation_alias="CONTRACT_BROWSER_DEMO_SESSION",
    )
    contract_demo_tenant_id: str = Field(default="demo-tenant", validation_alias="CONTRACT_DEMO_TENANT_ID")
    contract_demo_user_id: str = Field(default="demo-user", validation_alias="CONTRACT_DEMO_USER_ID")
    contract_demo_visibility: str = Field(default="private", validation_alias="CONTRACT_DEMO_VISIBILITY")
    contract_demo_user_roles: List[str] = Field(
        default=["legal", "public"],
        validation_alias="CONTRACT_DEMO_USER_ROLES",
    )
    contract_demo_session_ttl_seconds: int = Field(
        default=8 * 60 * 60,
        validation_alias="CONTRACT_DEMO_SESSION_TTL_SECONDS",
        ge=60,
        le=7 * 24 * 60 * 60,
    )
    api_rate_limit_per_minute: int = Field(default=120, validation_alias="API_RATE_LIMIT_PER_MINUTE")
    # 检索参数由 retrieval_strategy 动态组合使用
    # faq_top_k/doc_top_k：初次召回的候选数量，足够大才能让 reranker 从充足池中选优；
    # 但同时需要控制召回阶段的开销（向量检索 + BM25 的 I/O 耗时）
    faq_top_k: int = Field(default=20, validation_alias="FAQ_TOP_K")
    doc_top_k: int = Field(default=20, validation_alias="DOC_TOP_K")
    # rerank_top_n：CrossEncoder 精细重排后保留的段落数，太高会引入低质候选、太低会遗漏相关段落
    rerank_top_n: int = Field(default=5, validation_alias="RERANK_TOP_N")
    # final_context_top_n：最终进入 LLM 提示词的段落数——受 LLM 注意力衰减和窗口长度双重约束
    final_context_top_n: int = Field(default=4, validation_alias="FINAL_CONTEXT_TOP_N")
    # FAQ 直接命中阈值：FAQ 匹配分超过此值时跳过文档检索，直接返回 FAQ 答案，减少 LLM 调用
    faq_direct_score_threshold: float = Field(default=0.72, validation_alias="FAQ_DIRECT_SCORE_THRESHOLD")
    # 检索质量底线：重排得分低于此值的段落被丢弃，防止噪声段落污染 LLM 上下文
    rag_min_score_threshold: float = Field(default=0.2, validation_alias="RAG_MIN_SCORE_THRESHOLD")
    # 提示词总长度上限：平衡 LLM 上下文窗口限制与响应延迟——上下文越长，首 token 延迟越高
    max_prompt_context_chars: int = Field(default=6000, validation_alias="MAX_PROMPT_CONTEXT_CHARS")
    # 单篇文档截断长度：单段内容超过此值会被截断，在"保留完整语义"与"提示词预算"之间做权衡
    max_context_doc_chars: int = Field(default=1600, validation_alias="MAX_CONTEXT_DOC_CHARS")
    max_history_messages: int = Field(default=8, validation_alias="MAX_HISTORY_MESSAGES")
    history_summary_enabled: bool = Field(default=True, validation_alias="HISTORY_SUMMARY_ENABLED")
    history_summary_after_messages: int = Field(default=14, validation_alias="HISTORY_SUMMARY_AFTER_MESSAGES")
    history_recent_messages: int = Field(default=8, validation_alias="HISTORY_RECENT_MESSAGES")
    history_summary_max_chars: int = Field(default=1200, validation_alias="HISTORY_SUMMARY_MAX_CHARS")

    short_query_max_chars: int = Field(default=20, validation_alias="SHORT_QUERY_MAX_CHARS")
    faq_short_query_top_k: int = Field(default=30, validation_alias="FAQ_SHORT_QUERY_TOP_K")
    doc_complex_query_top_k: int = Field(default=24, validation_alias="DOC_COMPLEX_QUERY_TOP_K")
    retrieval_variant_max: int = Field(default=2, validation_alias="RETRIEVAL_VARIANT_MAX")
    retrieval_debug_enabled: bool = Field(default=True, validation_alias="RETRIEVAL_DEBUG_ENABLED")

    feedback_table_name: str = Field(default="qa_feedback", validation_alias="FEEDBACK_TABLE_NAME")
    chat_summary_table_name: str = Field(default="chat_session_summaries", validation_alias="CHAT_SUMMARY_TABLE_NAME")
    active_kb_version: str = Field(default="", validation_alias="ACTIVE_KB_VERSION")

    # 父子块切分参数，调整后需重新入库才能生效
    parent_chunk_size: int = Field(default=1000, validation_alias="PARENT_CHUNK_SIZE")
    child_chunk_size: int = Field(default=350, validation_alias="CHILD_CHUNK_SIZE")
    parent_overlap: int = Field(default=100, validation_alias="PARENT_OVERLAP")
    child_overlap: int = Field(default=50, validation_alias="CHILD_OVERLAP")

    # 合同领域规则参数集中配置；均有安全默认值，不要求部署环境新增变量。
    contract_review_confidence_threshold: float = Field(
        default=0.65,
        validation_alias="CONTRACT_REVIEW_CONFIDENCE_THRESHOLD",
        ge=0.0,
        le=1.0,
    )
    contract_upcoming_days: int = Field(default=7, validation_alias="CONTRACT_UPCOMING_DAYS", ge=0)
    contract_expiry_warning_days: int = Field(default=30, validation_alias="CONTRACT_EXPIRY_WARNING_DAYS", ge=0)
    contract_analysis_retrieval_top_k: int = Field(
        default=12,
        validation_alias="CONTRACT_ANALYSIS_RETRIEVAL_TOP_K",
        ge=1,
        le=100,
    )
    contract_max_upload_bytes: int = Field(
        default=25 * 1024 * 1024,
        validation_alias="CONTRACT_MAX_UPLOAD_BYTES",
        ge=1024,
    )
    contract_max_archive_uncompressed_bytes: int = Field(
        default=250 * 1024 * 1024,
        validation_alias="CONTRACT_MAX_ARCHIVE_UNCOMPRESSED_BYTES",
        ge=1024,
    )
    contract_max_archive_ratio: int = Field(
        default=100,
        validation_alias="CONTRACT_MAX_ARCHIVE_RATIO",
        ge=1,
        le=1000,
    )
    contract_max_archive_entries: int = Field(
        default=10_000,
        validation_alias="CONTRACT_MAX_ARCHIVE_ENTRIES",
        ge=1,
        le=100_000,
    )
    contract_analysis_stale_seconds: int = Field(
        default=30 * 60,
        validation_alias="CONTRACT_ANALYSIS_STALE_SECONDS",
        ge=60,
        le=24 * 60 * 60,
    )

    cors_allow_origins: List[str] = Field(default=["http://localhost:8000", "http://127.0.0.1:8000"], validation_alias="CORS_ALLOW_ORIGINS")

    @field_validator("env", mode="before")
    @classmethod
    def validate_environment_name(cls, value):
        normalized = str(value or "").strip().lower()
        if normalized not in SUPPORTED_APP_ENVIRONMENTS:
            raise ValueError("APP_ENV 必须是 dev、test、uat、staging、prd、prod 或 production")
        return normalized

    @field_validator("cors_allow_origins", "contract_demo_user_roles", mode="before")
    @classmethod
    def parse_list(cls, value):
        """解析环境变量中的 JSON 数组或逗号分隔列表配置。

        调用顺序：启动配置或前置校验 -> Settings.parse_list()。
        """
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator(
        "scenario_config_dir",
        "embedding_model_path",
        "reranker_model_path",
        "intent_model_path",
        mode="after",
    )
    @classmethod
    def resolve_project_relative_paths(cls, value: str) -> str:
        """允许 .env 使用相对路径；运行时统一解析成绝对路径。

        调用顺序：启动配置或前置校验 -> Settings.resolve_project_relative_paths()。
        """
        return _resolve_project_relative_path(value)

    @field_validator("document_parser_backend", mode="after")
    @classmethod
    def validate_document_parser_backend(cls, value: str) -> str:
        """限制文档解析后端取值，避免拼写错误导致入库行为不确定。

        调用顺序：启动配置或前置校验 -> Settings.validate_document_parser_backend()。
        """
        normalized = value.strip().lower()
        if normalized not in {"native", "docling"}:
            raise ValueError("DOCUMENT_PARSER_BACKEND 只能是 native 或 docling")
        return normalized

    @property
    def product_browser_title(self) -> str:
        """返回浏览器使用的展示标题，不参与任何内部标识构造。"""
        return f"{self.product_display_name} · {self.product_description}"

    @property
    def openapi_title(self) -> str:
        """返回 OpenAPI 的纯展示标题。"""
        return f"{self.contract_product_descriptor} API"

    @property
    def mysql_sync_uri(self) -> str:
        """构建 LangChain SQL 历史记录的 SQLAlchemy URI（charset=utf8mb4）。

        调用顺序：启动配置或前置校验 -> Settings.mysql_sync_uri()。
        """
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}?charset=utf8mb4"
        )

    @property
    def redis_url(self) -> str:
        """构建 Redis 连接 URL。

        使用 redis:// 协议格式，包含密码时自动拼接认证信息。URL 格式示例：
        - 无密码：redis://localhost:6379/0
        - 有密码：redis://:password@localhost:6379/0

        返回：
            Redis 连接 URL 字符串。

        调用顺序：启动配置 -> Settings.redis_url()。
        """
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程级配置快照。使用 lru_cache 避免重复解析，测试可用 get_settings.cache_clear() 切换环境变量。

    调用顺序：启动配置或前置校验 -> get_settings()。
    """
    return Settings()
