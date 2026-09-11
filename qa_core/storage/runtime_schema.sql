-- Runtime MySQL schema for Contract Intelligence.

CREATE TABLE IF NOT EXISTS {{KB_VERSIONS_TABLE}} (
    scenario_id VARCHAR(128) NOT NULL,
    kb_version VARCHAR(191) NOT NULL,
    version_seq BIGINT NOT NULL DEFAULT 0,
    status VARCHAR(32) NOT NULL,
    description TEXT NULL,
    created_at VARCHAR(40) NOT NULL,
    activated_at VARCHAR(40) NULL,
    archived_at VARCHAR(40) NULL,
    doc_collection VARCHAR(191) NOT NULL,
    faq_collection VARCHAR(191) NOT NULL,
    embedding_model_version VARCHAR(191) NOT NULL,
    reranker_model_version VARCHAR(191) NOT NULL,
    chunk_schema_version VARCHAR(191) NOT NULL,
    created_by VARCHAR(128) NOT NULL,
    sources_json LONGTEXT NULL,
    stats_json LONGTEXT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (scenario_id, kb_version),
    INDEX idx_kb_versions_status (scenario_id, status),
    INDEX idx_kb_versions_seq (scenario_id, version_seq),
    INDEX idx_kb_versions_created_at (scenario_id, created_at)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{KB_ACTIVE_TABLE}} (
    scenario_id VARCHAR(128) PRIMARY KEY,
    active_kb_version VARCHAR(191) NOT NULL DEFAULT '',
    previous_kb_version VARCHAR(191) NOT NULL DEFAULT '',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{KB_ACTIVATION_TABLE}} (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    scenario_id VARCHAR(128) NOT NULL,
    from_kb_version VARCHAR(191) NOT NULL DEFAULT '',
    to_kb_version VARCHAR(191) NOT NULL DEFAULT '',
    from_version_seq BIGINT NOT NULL DEFAULT 0,
    to_version_seq BIGINT NOT NULL DEFAULT 0,
    action VARCHAR(32) NOT NULL,
    reason TEXT NULL,
    activated_by VARCHAR(128) NOT NULL DEFAULT 'system',
    created_at VARCHAR(40) NOT NULL,
    INDEX idx_kb_activation_scenario_created (scenario_id, created_at),
    INDEX idx_kb_activation_to_version (scenario_id, to_kb_version)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{CACHE_NAMESPACE_TABLE}} (
    scenario_id VARCHAR(128) NOT NULL,
    tenant_id VARCHAR(128) NOT NULL DEFAULT 'default',
    dataset_id VARCHAR(128) NOT NULL DEFAULT 'default',
    cache_epoch BIGINT NOT NULL DEFAULT 1,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (scenario_id, tenant_id, dataset_id),
    INDEX idx_cache_namespace_updated_at (updated_at)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{KB_CHUNK_VERSIONS_TABLE}} (
    scenario_id VARCHAR(128) NOT NULL,
    chunk_id VARCHAR(191) NOT NULL,
    source VARCHAR(128) NOT NULL,
    kb_version VARCHAR(191) NOT NULL DEFAULT '',
    file_path TEXT NULL,
    valid_from_seq BIGINT NOT NULL DEFAULT 0,
    valid_to_seq BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (scenario_id, chunk_id),
    INDEX idx_chunk_version_source (scenario_id, source),
    INDEX idx_chunk_version_kb_version (scenario_id, kb_version),
    INDEX idx_chunk_version_validity (scenario_id, valid_from_seq, valid_to_seq)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{INDEX_MANIFEST_TABLE}} (
    manifest_key VARCHAR(64) PRIMARY KEY,
    scenario_id VARCHAR(128) NOT NULL,
    source VARCHAR(128) NOT NULL,
    path TEXT NOT NULL,
    fingerprint VARCHAR(191) NOT NULL,
    chunk_ids_json LONGTEXT NOT NULL,
    kb_version VARCHAR(191) NOT NULL,
    embedding_model_version VARCHAR(191) NOT NULL,
    chunk_schema_version VARCHAR(191) NOT NULL,
    updated_at VARCHAR(40) NOT NULL,
    row_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_manifest_scenario_version (scenario_id, kb_version),
    INDEX idx_manifest_source (scenario_id, source),
    INDEX idx_manifest_updated_at (scenario_id, updated_at)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{FEEDBACK_TABLE}} (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(191) NULL,
    scenario_id VARCHAR(128) NULL,
    tenant_id VARCHAR(128) NULL,
    dataset_id VARCHAR(128) NULL,
    question TEXT NOT NULL,
    answer MEDIUMTEXT NOT NULL,
    rating VARCHAR(32) NOT NULL,
    comment TEXT NULL,
    sources_json LONGTEXT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session_id (session_id),
    INDEX idx_scenario_id (scenario_id),
    INDEX idx_tenant_dataset (tenant_id, dataset_id),
    INDEX idx_rating (rating)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{CHAT_SUMMARY_TABLE}} (
    session_id VARCHAR(191) PRIMARY KEY,
    summary TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{CHAT_MESSAGES_TABLE}} (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(191) NOT NULL,
    message LONGTEXT NOT NULL,
    INDEX idx_chat_messages_session_id (session_id)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{CONTRACTS_TABLE}} (
    id VARCHAR(36) PRIMARY KEY,
    tenant_id VARCHAR(128) NOT NULL,
    owner_user_id VARCHAR(128) NOT NULL DEFAULT 'anonymous',
    contract_name VARCHAR(255) NOT NULL,
    contract_number VARCHAR(191) NULL,
    contract_type VARCHAR(64) NULL,
    status VARCHAR(32) NOT NULL,
    processing_status VARCHAR(32) NOT NULL,
    overall_risk_level VARCHAR(16) NOT NULL DEFAULT 'low',
    scenario_id VARCHAR(128) NOT NULL,
    dataset_id VARCHAR(128) NOT NULL,
    visibility VARCHAR(32) NOT NULL DEFAULT 'private',
    allowed_roles_json TEXT NOT NULL,
    error_message TEXT NULL,
    created_at VARCHAR(40) NOT NULL,
    updated_at VARCHAR(40) NOT NULL,
    INDEX idx_contract_tenant_updated (tenant_id, updated_at),
    INDEX idx_contract_tenant_risk (tenant_id, overall_risk_level),
    INDEX idx_contract_tenant_status (tenant_id, status)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{CONTRACT_DOCUMENTS_TABLE}} (
    id VARCHAR(36) PRIMARY KEY,
    contract_id VARCHAR(36) NOT NULL,
    doc_id VARCHAR(191) NULL,
    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_type VARCHAR(32) NOT NULL,
    parser_backend VARCHAR(32) NOT NULL,
    parse_status VARCHAR(32) NOT NULL,
    document_type VARCHAR(64) NOT NULL,
    document_version VARCHAR(64) NOT NULL,
    page_count INT NOT NULL DEFAULT 0,
    chunk_count INT NOT NULL DEFAULT 0,
    chunk_ids_json LONGTEXT NOT NULL,
    created_at VARCHAR(40) NOT NULL,
    updated_at VARCHAR(40) NOT NULL,
    INDEX idx_contract_documents_contract (contract_id)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{CONTRACT_ANALYSES_TABLE}} (
    id VARCHAR(36) PRIMARY KEY,
    contract_id VARCHAR(36) NOT NULL,
    status VARCHAR(32) NOT NULL,
    analysis_version VARCHAR(64) NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    extraction_json LONGTEXT NOT NULL,
    summary_json LONGTEXT NOT NULL,
    started_at VARCHAR(40) NOT NULL,
    finished_at VARCHAR(40) NULL,
    error_summary TEXT NULL,
    created_at VARCHAR(40) NOT NULL,
    INDEX idx_contract_analysis_contract_created (contract_id, created_at),
    INDEX idx_contract_analysis_status_finished (contract_id, status, finished_at)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{CONTRACT_OBLIGATIONS_TABLE}} (
    id VARCHAR(36) PRIMARY KEY,
    contract_id VARCHAR(36) NOT NULL,
    analysis_id VARCHAR(36) NOT NULL,
    item_index INT NOT NULL DEFAULT 0,
    obligation_type VARCHAR(64) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT NULL,
    responsible_party VARCHAR(255) NULL,
    beneficiary_party VARCHAR(255) NULL,
    trigger_type VARCHAR(32) NOT NULL,
    trigger_event_type VARCHAR(64) NULL,
    trigger_event_id VARCHAR(36) NULL,
    trigger_description TEXT NULL,
    planned_date VARCHAR(40) NULL,
    actual_completed_at VARCHAR(40) NULL,
    time_rule_json TEXT NOT NULL,
    amount VARCHAR(191) NULL,
    currency VARCHAR(32) NULL,
    percentage VARCHAR(64) NULL,
    status VARCHAR(32) NOT NULL,
    risk_level VARCHAR(16) NOT NULL,
    evidence_json TEXT NULL,
    source_page INT NULL,
    source_chunk_id VARCHAR(191) NULL,
    confidence DECIMAL(5,4) NOT NULL DEFAULT 0,
    needs_review BOOLEAN NOT NULL DEFAULT FALSE,
    note TEXT NULL,
    created_at VARCHAR(40) NOT NULL,
    updated_at VARCHAR(40) NOT NULL,
    INDEX idx_contract_obligation_contract (contract_id, status),
    INDEX idx_contract_obligation_due (planned_date, status),
    INDEX idx_contract_obligation_analysis_item (analysis_id, item_index)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{CONTRACT_RISKS_TABLE}} (
    id VARCHAR(36) PRIMARY KEY,
    contract_id VARCHAR(36) NOT NULL,
    analysis_id VARCHAR(36) NOT NULL,
    risk_fingerprint VARCHAR(512) NOT NULL,
    risk_type VARCHAR(64) NOT NULL,
    risk_name VARCHAR(255) NOT NULL,
    risk_level VARCHAR(16) NOT NULL,
    risk_source VARCHAR(16) NOT NULL,
    status VARCHAR(32) NOT NULL,
    severity INT NOT NULL DEFAULT 1,
    likelihood INT NOT NULL DEFAULT 1,
    urgency INT NOT NULL DEFAULT 1,
    description TEXT NOT NULL,
    impact TEXT NULL,
    reason TEXT NOT NULL,
    suggestion TEXT NOT NULL,
    evidence_json TEXT NULL,
    source_page INT NULL,
    source_chunk_id VARCHAR(191) NULL,
    related_obligation_id VARCHAR(36) NULL,
    confidence DECIMAL(5,4) NOT NULL DEFAULT 0,
    needs_review BOOLEAN NOT NULL DEFAULT FALSE,
    created_at VARCHAR(40) NOT NULL,
    INDEX idx_contract_risk_contract_level (contract_id, risk_level),
    INDEX idx_contract_risk_type (contract_id, risk_type),
    INDEX idx_contract_risk_analysis_status (analysis_id, status, risk_level),
    INDEX idx_contract_risk_analysis_fingerprint (analysis_id, risk_fingerprint(191))
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS {{CONTRACT_EVENTS_TABLE}} (
    id VARCHAR(36) PRIMARY KEY,
    contract_id VARCHAR(36) NOT NULL,
    event_type VARCHAR(128) NOT NULL,
    event_date VARCHAR(40) NOT NULL,
    related_obligation_id VARCHAR(36) NULL,
    description TEXT NULL,
    source VARCHAR(64) NOT NULL,
    created_by VARCHAR(128) NOT NULL,
    created_at VARCHAR(40) NOT NULL,
    INDEX idx_contract_event_contract_time (contract_id, event_date),
    INDEX idx_contract_event_type (contract_id, event_type),
    INDEX idx_contract_event_idempotency (contract_id, event_type, event_date, related_obligation_id)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
