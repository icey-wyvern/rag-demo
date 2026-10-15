# GOV.UK RAG Demo

A retrieval-augmented generation demo over a seeded set of GOV.UK strategy and resilience documents. The stack is FastAPI, PostgreSQL with pgvector, React, and the OpenAI API.

## Features

- Semantic retrieval over a fixed GOV.UK corpus
- Answer generation from retrieved chunks, with links to retrieved source documents
- Chunk inspection mode for debugging retrieval quality
- Recoverable corpus ingestion with explicit readiness/status reporting
- Single-command local startup with Docker Compose

## Seeded documents

- National Risk Register 2025
- The UK Government Resilience Framework
- British Energy Security Strategy
- Government Cyber Security Strategy 2022-2030
- Net Zero Strategy: Build Back Greener

## Local development

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Optional: place local copies of the seeded PDFs in [docs/README.md](docs/README.md) to avoid network downloads during ingestion.
3. Start the stack:

```bash
docker compose up --build
```

4. Open `http://localhost:8080`.

`webapi` serves the UI immediately. `ingest` builds the retrieval corpus in the background. The UI polls corpus status and unlocks querying once ingestion is complete.

PostgreSQL runs locally through Docker Compose. Initial ingestion may download the GOV.UK PDFs. OpenAI is required for embeddings, and generative queries also use OpenAI, so the demo is not fully offline.

## Commands

Build the corpus again:

```bash
docker compose run --rm ingest python -m api.ingest_cli --force
```

Reset local corpus state:

```bash
docker compose run --rm webapi python -m api.reset_cli --yes
```

Run backend tests:

```bash
python -m unittest discover -s tests
```

Install frontend dependencies from the tracked lockfile and run checks:

```bash
cd web
npm ci
npm run build
npm run test
```

## Architecture

- `api/main.py`: FastAPI endpoints and corpus readiness handling
- `api/ingest.py`: recoverable corpus ingestion pipeline
- `api/store.py`: database persistence for corpus state, documents, chunks, and vector search
- `api/generator.py`: answer generation from retrieved chunks and source-link assembly
- `web/src`: status-aware chat UI

## Operational notes

- `GET /health` reports API health and corpus state.
- `GET /api/status` reports corpus readiness for the UI.
- `POST /api/query` returns `503` until the corpus is ready.
- Seed PDFs are cached under `/app/data/seed-cache` inside Docker to make re-runs faster.

## Notes

- On first startup, the `ingest` service downloads the seeded PDFs if local copies are not present.
- Re-running ingestion replaces the indexed chunks for each seeded document.
- If ingestion fails, the API still starts and surfaces the error through `GET /health`, `GET /api/status`, and the UI status banner.
