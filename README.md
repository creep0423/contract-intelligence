# Contract Intelligence

**知契**

> 企业合同履约与风险智能<br>
> 从原文证据，到业务判断。

Contract Intelligence 是面向企业合同签后履约管理的证据驱动智能系统。当前正式运行时仅提供合同产品，不提供场景切换或通用知识问答入口。

## Overview

系统覆盖合同文件解析、条款级切分、结构化抽取、履约义务、TimeRule、履约事件、规则与语义风险、补充协议冲突、Source Evidence、合同问答和版本化 Benchmark。

本系统是企业内部辅助风控工具，不替代法务或律师的专业判断。

## Core Capabilities

| 能力 | 当前实现 |
| --- | --- |
| 合同工作台 | 列表、上传、详情、摘要、关键字段、风险、义务、时间轴、事件和分析历史 |
| 合同分析 | Pydantic Structured Output；事实、义务、风险和摘要分阶段执行 |
| 时间计算 | 固定日期、合同签署/生效相对日期、事件触发日期、工作日与周期规则 |
| 风险识别 | 规则风险、LLM 语义风险、稳定 fingerprint、处理状态和历史分析版本 |
| 补充协议 | 多文档版本、独立引用和潜在冲突双证据 |
| 合同问答 | Evidence-grounded 回答、无证据拒答、HTTP 历史与 WebSocket Streaming |
| 安全 | tenant/dataset/visibility/role 数据域、IDOR 防护、可信上游身份、Prompt Injection Guard |
| Benchmark | 标注 Schema、确定性 Matcher、Metrics、Gate、Run Comparison 和人工复核队列 |

## Architecture

- Python 3.12
- FastAPI / WebSocket
- LangChain 与 OpenAI-compatible DashScope LLM
- BGE-M3 Embedding 与 BGE Reranker
- Milvus Dense + BM25 Hybrid Retrieval
- MySQL / Redis
- 原生 HTML/CSS/JavaScript

核心 Python package 为 `qa_core`。当前合同风险场景标识为 `tender_contract_risk`。

## Contract Intelligence Pipeline

```text
Upload
→ Parse
→ Clause-aware Parent/Child Chunking
→ Milvus Hybrid Dense + BM25 Index/Retrieval
→ Structured Extraction
→ Obligations
→ TimeRule Resolution
→ Contract Events
→ Deterministic + Semantic Risk Engine
→ Source Evidence Validation
→ Summary / Timeline / Contract QA
```

摘要不是后续风险或义务分析的事实源；下游业务判断依赖结构化抽取和可信原文 Chunk。

## Security Model

合同 HTTP 与 WebSocket 使用统一的 `ContractAccessContext`。生产环境必须设置 `CONTRACT_TRUSTED_UPSTREAM_TOKEN`，可信网关应删除客户端自带身份 Header，再注入 `x-tenant-id`、`x-user-id`、`x-visibility`、`x-user-roles` 和 `x-contract-upstream-token`。

合同列表、详情、文件、分析、义务、风险、事件、时间轴和问答均在查询层应用 tenant/data scope。Milvus 检索和 parent/citation 获取继续携带合同 dataset 与权限过滤。

合同正文被视为不可信数据。抽取、义务、风险、摘要和 QA 使用共享 Prompt Injection Guard；Source Evidence 的 document、contract、page 和 chunk metadata 以后端检索上下文为准，Quote 必须通过原文校验。这是一套纵深防护机制，不代表绝对安全。

## Configuration

```powershell
Copy-Item .env.contract-intelligence.example .env.contract-intelligence
notepad .env.contract-intelligence
```

至少填写 `MYSQL_PASSWORD`、`MINIO_ROOT_USER`、`MINIO_ROOT_PASSWORD`、`DASHSCOPE_API_KEY` 和 `ADMIN_API_TOKEN`。生产及 UAT 环境还必须配置非空 `CONTRACT_TRUSTED_UPSTREAM_TOKEN`。示例文件不包含默认口令或真实 secret。

展示品牌配置与数据库、Milvus、Redis、权限和路由标识解耦：

```text
PRODUCT_DISPLAY_NAME=知契
PRODUCT_ENGLISH_DESCRIPTOR=Enterprise Intelligence
CONTRACT_PRODUCT_DESCRIPTOR=Contract Intelligence
```

## Docker Deployment

新部署使用独立的 [docker-compose.contract-intelligence.yml](docker-compose.contract-intelligence.yml)，不会读取、停止或删除其他部署资源。

以下端口和命令描述通用/本地部署。受保护的 Resume Release Real UAT 是独立运行时：Compose project 为 `contract-intelligence-real-uat`，API 为 `http://127.0.0.1:28100`；不要将两者的 project、端口或数据卷混用。

| 服务 | 默认宿主机地址 |
| --- | --- |
| Web/API | `http://127.0.0.1:18000` |
| OpenAPI | `http://127.0.0.1:18000/api/docs` |
| MySQL | `127.0.0.1:23307` |
| Redis | `127.0.0.1:16379` |
| Milvus | `127.0.0.1:29530` |
| Milvus health | `127.0.0.1:19091` |
| MinIO / Console | `http://127.0.0.1:19000` / `http://127.0.0.1:19001` |

首次部署：

```powershell
docker build -f Dockerfile.base -t contract-intelligence-base:py312 .
docker compose -f docker-compose.contract-intelligence.yml --env-file .env.contract-intelligence up -d mysql redis etcd minio milvus
docker compose -f docker-compose.contract-intelligence.yml --env-file .env.contract-intelligence build api
docker compose -f docker-compose.contract-intelligence.yml --env-file .env.contract-intelligence run --rm api python scripts/rebuild_kb_version.py --scenario tender_contract_risk --new-version --force --quality-gate --activate
docker compose -f docker-compose.contract-intelligence.yml --env-file .env.contract-intelligence up -d api
```

或执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/deploy_contract_intelligence.ps1 -InitializeKnowledgeBase
```

所有数据库、向量、对象存储、缓存、合同文件、报告和日志卷均使用 `contract-intelligence-*` 独立名称。禁止使用 `down -v` 或 volume prune 清理业务数据。

## Benchmark

- `benchmark_version`: `contract-benchmark-v1.0.0`
- `dataset_version`: `small-synthetic-v1.0.0`
- Synthetic cases: 10
- Current exploratory Real UAT cases: 2（不是大规模生产 Benchmark）
- Current Real UAT reference: AI-assisted provisional Silver，**NOT Human Gold**
- Exploratory aggregate: Obligation precision / recall / F1 = 90.9% / 20.4% / 33.3%；TimeRule fully correct = 5.0%

```powershell
python -m qa_core.evaluation.validate_annotations --suite full
python -m qa_core.evaluation.runner --suite fast
python -m qa_core.evaluation.runner --suite full
python -m qa_core.evaluation.gate <benchmark_report.json>
python -m qa_core.evaluation.compare_runs <baseline.json> <candidate.json>
```

默认离线模式不调用付费模型。`offline`、`mock`、`snapshot` 和 `real_model` 必须明确区分；Synthetic Benchmark 不能替代真实企业 UAT。

## Local Validation

```powershell
python -m pytest tests -q
python -m compileall -q app.py qa_core tests
node --check static/js/contracts.js
node --check static/js/admin.js
python -m pip check
docker compose -f docker-compose.contract-intelligence.yml --env-file .env.contract-intelligence config -q
```

## Data Identifiers

当前内部标识包括 `qa_core`、`tender_contract_risk`、合同 API/JSON Schema、MySQL 业务表、Milvus collection、`contract:<contract_id>` dataset namespace、Benchmark ID 和 artifact schema。

## Limitations

- 真实模型分析需要有效供应商密钥并会产生费用；
- 当前 Resume Release 仅评估 machine-readable derived PDF input；OCR 不在范围内且未评估；
- 多模态输入未评估；GraphRAG 未启用；Capability Router 不包含在本次发布中；
- 当前 Real UAT 仅有 2 个 exploratory cases，参考标注是 AI-assisted provisional Silver，不是 Human Gold；
- 当前 exploratory baseline 的 Obligation recall 与 TimeRule accuracy 仍然有限，上述低指标不得解释为生产质量；
- 语义风险与合同冲突可能需要法务裁定；
- 合成 Benchmark 不能代表真实合同分布；
- 单机 Compose 不是高可用生产编排方案；

## Repository Data Policy

版本库只保存源码、配置模板、产品文档、测试和明确标记的 synthetic 数据。真实合同、私有 UAT/Gold、上传文件、日志、报告、Benchmark 运行产物、模型文件、数据库文件、证书和本地环境 secret 必须留在 Git 之外。提交前应复核 `.gitignore` 并执行 secret/private-data 扫描。
