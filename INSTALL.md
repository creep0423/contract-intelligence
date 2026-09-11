# Contract Intelligence Deployment

Contract Intelligence 使用独立 Compose project、容器、网络、镜像、端口和数据卷。该部署不会管理或修改宿主机上的其他 Docker stack。

## Prerequisites

- Docker Engine / Docker Desktop with Compose v2
- `models/bge-m3`、`models/bge-reranker-large`、`models/bert_intent_classifier_v1`
- 有效 `DASHSCOPE_API_KEY`
- 非占位的 `ADMIN_API_TOKEN`
- 随机生成且仅在本地或 secret manager 中保存的 `MYSQL_PASSWORD`、`MINIO_ROOT_USER` 和 `MINIO_ROOT_PASSWORD`
- UAT/生产环境由可信上游注入的 `CONTRACT_TRUSTED_UPSTREAM_TOKEN`

## Configure and deploy

```powershell
Copy-Item .env.contract-intelligence.example .env.contract-intelligence
notepad .env.contract-intelligence
docker build -f Dockerfile.base -t contract-intelligence-base:py312 .
powershell -ExecutionPolicy Bypass -File scripts/deploy_contract_intelligence.ps1 -InitializeKnowledgeBase
```

- Application: `http://127.0.0.1:18000/`
- OpenAPI: `http://127.0.0.1:18000/api/docs`

## Optional future data migration

新环境不自动导入任何既有数据库、向量、对象、缓存或合同文件。需要迁移时，应在单独维护窗口中：

1. MySQL 逻辑备份并导入，比较表、索引、外键、row count 和校验和；
2. Milvus 导出或重新索引，比较 schema 和 vector count，并执行付款检索、引用、无证据和租户隔离测试；
3. 合同文件按 tenant/contract namespace 复制并比较 SHA-256，同时验证数据库 metadata；
4. MinIO 对象逐项校验 bucket/key/checksum；
5. Redis 默认不迁移，由新环境自然重建缓存。

不得使用 `docker compose down -v`、`docker volume prune` 或其他会删除持久化数据的命令。
