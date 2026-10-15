# Version-aware GOV.UK guidance RAG

[![CI](https://github.com/icey-wyvern/rag-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/icey-wyvern/rag-demo/actions/workflows/ci.yml)

A version-aware, evidence-first retrieval and answer system for evolving GOV.UK guidance. It stores immutable versions as they are observed locally, retrieves eligible passages using PostgreSQL full-text search, pgvector cosine search, or hybrid rank fusion, and returns claim-level answers with cited passages and exact supporting quotes. The application validates citation IDs and quotes before returning an answer. It can abstain when evidence is unavailable or the provider requests abstention; invalid generated output is rejected.

The repository includes separate retrieval, generation/evidence, and hostile-content evaluations. A disposable PostgreSQL/pgvector suite exercises migrations, persistence, retrieval, and the HTTP query path with deterministic fake model providers.

## Core capabilities

- **Version-aware ingestion:** a stable source identity links immutable observed snapshots and passages. Repeated identical observations do not create duplicate snapshots; changed observations retain their predecessor.
- **Observed-as-of retrieval:** each source contributes its latest snapshot observed by a timezone-aware cutoff. This is the latest version **this corpus had observed** at or before that time. It does not reconstruct authoritative GOV.UK state at arbitrary historical dates.
- **Three retrieval modes:** Lexical uses PostgreSQL English full-text search; Semantic uses stored embeddings and pgvector cosine distance; Hybrid combines their ranks with reciprocal rank fusion (RRF, constant 60). Hybrid is the API and UI default.
- **Evidence-first answers:** application-owned prompt-local IDs identify retrieved passages. Every answered claim needs a citation whose ID exists and whose exact supporting quote occurs in that passage. Missing citations, invented IDs, invalid quotes, and malformed output fail closed. Exact-quote validation checks reference integrity, not semantic entailment or universal truth.
- **Explicit abstention:** no retrieved evidence produces a deterministic abstention without a generation call; the provider can also request abstention with a reason.
- **Hostile-content boundary:** questions and source content are untrusted data, serialized separately from fixed application instructions. Validation remains authoritative after generation. This reduces risk; it does not make a model immune to prompt injection.
- **Change intelligence:** an internal service compares two observed snapshots of one source and retains older/newer provenance. It classifies passages as unchanged, added, removed, or modified and can request an evidence-bound summary. Textual differences are not automatically semantic contradictions.

See [architecture](docs/architecture.md), [evaluation and validation scope](docs/evaluation.md), and the [demo walkthrough](docs/demo.md).

## Verify without API credentials

These commands use fixtures and fake providers. They do not require an OpenAI key, Azure account, or live GOV.UK request. Run them from the repository root with the existing repository-local `.venv` and frontend dependencies.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m api.security_eval
.\.venv\Scripts\python.exe -m api.generation_eval_cli
```

```powershell
Set-Location web
npm run test
npm run build
npm run lint
Set-Location ..
```

With Docker Desktop running, exercise the **real** isolated PostgreSQL/pgvector path:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-postgres-integration.ps1
```

The execution-policy bypass applies only to that PowerShell process; it does not change the machine-wide policy. The runner starts the dedicated `integration-db` service from `docker-compose.integration.yaml` with test-only credentials, a separate Compose project, and disposable tmpfs storage. It refuses to reuse an existing test container, never uses the normal `appdb` volume, and removes its own container/network afterward. Ordinary backend discovery safely skips database tests when `RAG_TEST_DATABASE_URL` is absent.

The generation fixture deliberately includes two validly cited responses using the wrong source or snapshot. Its `13/15` contract pass result is a diagnostic fixture result, **not** live-model accuracy. See [evaluation](docs/evaluation.md) for the metric definitions.

## Run the full application

This path uses live GOV.UK acquisition and OpenAI embedding/generation adapters. It requires Docker and a configured OpenAI key. Do not use these commands as part of credential-free verification.

1. Copy `.env.example` to `.env` and replace the placeholder `OPENAI_API_KEY`. The tracked example contains no real credentials.
2. Start PostgreSQL and the API container. The Docker build also builds the React/TypeScript frontend; FastAPI serves those assets, so this command starts both backend and UI.

   ```powershell
   docker compose up --build -d db webapi
   ```

3. Apply Alembic migrations. PostgreSQL initialization creates the application role and `vector` extension; **Alembic owns application tables, indexes, and constraints**.

   ```powershell
   docker compose run --rm webapi alembic upgrade head
   ```

4. Acquire the configured guidance through the live GOV.UK Content API, then create passage embeddings through OpenAI. These are explicit, opt-in commands; nothing polls for changes automatically.

   ```powershell
   docker compose run --rm webapi python -m api.govuk_ingest_cli
   docker compose run --rm webapi python -m api.passage_embedding_cli
   ```

5. Open `http://localhost:8080`. The UI uses `/api/status` readiness. Lexical retrieval needs versioned passages; Hybrid and Semantic also need matching stored embeddings and a query embedding provider. Generated queries use the configured OpenAI answer provider.

The configured acquisition paths are in `api/govuk_sources.py`. Local history grows as differing responses are observed; source publication/update timestamps are retained as provenance and do not set temporal eligibility. The curated observation A/B fixtures are illustrative test data, not verified historical GOV.UK publication versions. This full live deployment flow has not been end-to-end validated here; the database/query stack has been validated with fake providers.

**Existing database caution:** migration `20261015_04` drops obsolete `corpus_state`, `documents`, and `chunks` tables and their rows, leaving versioned tables intact. Existing installations whose old tables were created by PostgreSQL bootstrap SQL may need a database owner to apply that migration. Review old data and ownership before upgrading; downgrade restores table structure, not deleted rows.

## Public query API

`POST /api/query` accepts `q` (or `question`), `mode` (`generate` or `chunks`), `retrieval_mode` (`hybrid`, `semantic`, or `lexical`), optional timezone-aware `as_of`, and `k` (1–20, default 5). The defaults are generated answers and Hybrid retrieval. A naive `as_of` is rejected. `chunks` mode returns retrieved passages for inspection without invoking generation.

```powershell
$uri = "http://localhost:8080/api/query"
Invoke-RestMethod -Method Post -Uri $uri -ContentType "application/json" -Body (@{ q = "What does the observed guidance say about user needs?" } | ConvertTo-Json -Compress)
Invoke-RestMethod -Method Post -Uri $uri -ContentType "application/json" -Body (@{ q = "What does the observed guidance say about user needs?"; retrieval_mode = "lexical" } | ConvertTo-Json -Compress)
Invoke-RestMethod -Method Post -Uri $uri -ContentType "application/json" -Body (@{ q = "What guidance had this corpus observed?"; retrieval_mode = "lexical"; as_of = "2026-10-15T12:00:00+00:00" } | ConvertTo-Json -Compress)
```

The timestamp is an **illustrative cutoff**, not a claim that a particular source version exists at that time. A generated response has `status` (`answered` or `abstained`), `answer`, `claims` with cited quotes when answered, cited `sources`/`citations`, retrieval-inspection `chunks`, and small `retrieval` metadata. Abstention is HTTP 200 with a reason and no supporting source links. Invalid generated answers are rejected with a sanitized error rather than returned as valid answers. Retrieved chunks are never presented as answer citations.

The UI offers Hybrid, Semantic, and Lexical selection; an optional local observed-as-of control; generated answers with expandable provenance; explicit abstention; and Chunks only mode for inspecting retrieval. See the [walkthrough](docs/demo.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| `api/` | FastAPI wiring, acquisition, versioned persistence/retrieval, evidence-first generation, comparison, and evaluators |
| `migrations/` and `initdb/` | Alembic application schema and minimal PostgreSQL bootstrap |
| `eval/` | Reviewable retrieval and generation fixture datasets |
| `tests/integration/` | Disposable real PostgreSQL/pgvector migration, query, and API tests |
| `web/` | React/TypeScript UI built into the API container |
| `docs/` | Architecture, evaluation, and demonstration guides |

## Validation scope

Real PostgreSQL/pgvector integration has passed locally and in [hosted GitHub Actions](https://github.com/icey-wyvern/rag-demo/actions/workflows/ci.yml): Alembic migration cycles, immutable ingestion, embedding persistence, PostgreSQL full-text search, cosine retrieval, Hybrid RRF, temporal selection, readiness, and `/api/query`. Structured generation, citation checks, abstention, and hostile-content cases use deterministic fake providers. No production-scale traffic, arbitrary historical GOV.UK reconstruction, semantic entailment verification, live Azure validation, or prompt-injection immunity is claimed.
