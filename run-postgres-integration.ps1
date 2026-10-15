$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $repoRoot

if (-not (Test-Path -LiteralPath '.\.venv\Scripts\python.exe')) {
    throw 'Repository-local .venv\Scripts\python.exe is required.'
}

$compose = @('-f', 'docker-compose.integration.yaml', '-p', 'gvt-rag-integration')
docker info --format '{{.ServerVersion}}' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Docker engine is unavailable.' }
$existing = docker compose @compose ps -a -q integration-db
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the dedicated integration service.' }
if ($existing) { throw 'The integration service already exists; refusing to reuse or remove it.' }

$env:RAG_TEST_DATABASE_URL = 'postgresql+psycopg://rag_test:rag_test_password@127.0.0.1:55439/rag_integration_test'
try {
    docker compose @compose up -d --wait integration-db
    if ($LASTEXITCODE -ne 0) { throw 'Integration PostgreSQL did not become healthy.' }
    & '.\.venv\Scripts\python.exe' -m unittest discover -s tests/integration -v
    if ($LASTEXITCODE -ne 0) { throw 'PostgreSQL integration tests failed.' }
} finally {
    docker compose @compose down
    Remove-Item Env:RAG_TEST_DATABASE_URL -ErrorAction SilentlyContinue
}
