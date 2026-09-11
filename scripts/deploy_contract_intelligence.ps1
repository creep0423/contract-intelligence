param(
    [switch]$InitializeKnowledgeBase
)

$ErrorActionPreference = "Stop"
$composeFile = "docker-compose.contract-intelligence.yml"
$environmentFile = ".env.contract-intelligence"
$baseImage = "contract-intelligence-base:py312"

if (-not (Test-Path -LiteralPath $composeFile)) {
    throw "Missing $composeFile"
}
if (-not (Test-Path -LiteralPath $environmentFile)) {
    throw "Copy .env.contract-intelligence.example to .env.contract-intelligence first."
}

docker image inspect $baseImage *> $null
if ($LASTEXITCODE -ne 0) {
    docker build -f Dockerfile.base -t $baseImage .
    if ($LASTEXITCODE -ne 0) { throw "Contract Intelligence base image build failed." }
}

docker compose -f $composeFile --env-file $environmentFile up -d mysql redis etcd minio milvus
if ($LASTEXITCODE -ne 0) { throw "Contract Intelligence infrastructure startup failed." }

docker compose -f $composeFile --env-file $environmentFile build api
if ($LASTEXITCODE -ne 0) { throw "Contract Intelligence API image build failed." }

if ($InitializeKnowledgeBase) {
    docker compose -f $composeFile --env-file $environmentFile run --rm api `
        python scripts/rebuild_kb_version.py --scenario tender_contract_risk --new-version --force --quality-gate --activate
    if ($LASTEXITCODE -ne 0) { throw "Contract Intelligence knowledge-base initialization failed." }
}

docker compose -f $composeFile --env-file $environmentFile up -d api
if ($LASTEXITCODE -ne 0) { throw "Contract Intelligence API startup failed." }

docker compose -f $composeFile --env-file $environmentFile ps
Write-Host "Contract Intelligence is available at http://127.0.0.1:18000/"
