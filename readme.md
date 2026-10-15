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

Apply the additive versioned corpus schema (not yet used by the API):

```bash
docker compose run --rm webapi alembic upgrade head
```

## Optional versioned GOV.UK acquisition

After applying the Alembic schema, `python -m api.govuk_ingest_cli` fetches the three configured Service Standard guidance paths in `api/govuk_sources.py` through the GOV.UK Content API. This command is separate from the legacy PDF ingestion and is not used by `/api/query`.

The Content API's `content_id` supplies stable logical identity. The command extracts guidance body text, then the versioned ingestion service hashes those exact normalized UTF-8 bytes. Each differing observation becomes an immutable local snapshot; an identical observation is unchanged. Source publication/update timestamps are kept separate from the time we observed the response. This does not provide arbitrary historical GOV.UK bodies: local history grows only as differing responses are observed.

The small fixtures under `tests/fixtures` use trimmed fields and guidance text from a public Content API response observed on 15 October 2026. The two bodies are curated subsets for deterministic change tests, not a claim that both were historically published as shown or authoritative current content. Tests use these fixtures without network access. The configured paths have not all been live validated.

## Versioned lexical retrieval and evaluation

`api.versioned_retriever.VersionedRetriever` searches the new versioned corpus. Its lexical baseline uses PostgreSQL English full-text search (`websearch_to_tsquery`, `to_tsvector`, `ts_rank_cd`) over `passages`, with visible scores and a GIN expression index. This is an internal Python interface; `/api/query` still uses the legacy corpus.

Without `as_of`, each source contributes only its latest locally observed snapshot. With a timezone-aware `as_of`, each contributes its latest snapshot observed at or before that time. Ties use snapshot creation time and ID. This reconstructs what this corpus had observed, not arbitrary historical GOV.UK state. Source publication/update times are provenance, not eligibility cutoffs.

After populating the versioned PostgreSQL corpus, run `python -m api.retrieval_eval_cli` for the small current-source dataset, or pass `--dataset PATH` for a corpus-specific dataset. The separate `eval/fixture_retrieval_cases.json` has seven deterministic scenarios, including illustrative observations A and B at 15 and 16 October 2026. It requires a matching isolated fixture corpus for a meaningful database run; its two bodies are not verified historical publications. The evaluator reports Recall@1, Recall@3, and MRR over cases with expected evidence; cases expecting none pass only when retrieval returns no results and have a separate accuracy. Temporal accuracy covers `as_of` cases, including expected absence. Failed case IDs are reported. No benchmark number is claimed here.

## Opt-in versioned passage embeddings

`passage_embeddings` stores derived vectors separately from immutable passage text and hashes. Each row belongs to one passage and one provider/model pair, so reruns skip existing vectors and a different model can coexist without replacing history. The current column is `vector(1536)` and the indexer rejects other dimensions; this supports the configured `text-embedding-3-small` default, not arbitrary vector sizes.

After applying Alembic migrations and ingesting versioned passages, run `python -m api.passage_embedding_cli` with a configured `OPENAI_API_KEY`. The command uses the configured `EMBED_MODEL`, indexes missing passages in batches, and reports already indexed, newly indexed, and remaining counts. Each batch commits atomically; a failed batch rolls back while earlier batches remain. This is optional and makes OpenAI calls when run. `/api/query` does not use these embeddings.

## Internal retrieval modes

The versioned retriever accepts `mode="lexical"`, `"semantic"`, or `"hybrid"`. All modes select the same latest locally observed snapshot per source, with the same optional observed-as-of cutoff described above. Semantic retrieval embeds the question through the configured provider, searches only stored embeddings for that provider/model, and orders eligible passages by exact pgvector cosine distance (`<=>`); its Evidence score is **distance, lower is better**. No ANN index is added yet: corpus scale and index tuning have not been measured.

Hybrid retrieval combines lexical and semantic ranks with Reciprocal Rank Fusion, `sum(1 / (60 + rank))`, rather than mixing their incomparable raw scores. Each component requests `min(60, 3 × final K)` candidates; the final Evidence score is an RRF score, higher is better, with both component ranks when present. Its two database rankings share one repeatable-read snapshot so concurrent ingestion cannot mix versions. Run `python -m api.retrieval_eval_cli --mode all` against a populated local corpus to compare the same cases and metrics. The CLI also accepts `lexical`, `semantic`, or `hybrid` individually. Semantic and hybrid modes call the configured query embedding provider, so they require its API key when using OpenAI. Fake-vector tests check mechanics, not semantic quality; real benchmark numbers require a populated corpus and real provider run. The public API and legacy RAG path remain unchanged.

## Internal evidence-first answers

`api.versioned_answer.VersionedAnswerService` can compose the internal retriever with `EvidenceFirstAnswerService`. The service assigns prompt-local IDs (`E1`, `E2`, ...), sends provenance and passage text as clearly delimited reference material, and requests structured claims or an abstention. It resolves each citation against the supplied Evidence and requires a non-empty, exact quotation present in that passage. Duplicate identical citations are collapsed. Any invalid claim rejects the whole answer; the human-readable answer is assembled only from accepted claims.

With no retrieved Evidence, the service abstains without a model call. The model may also abstain with a reason and no claims. The optional OpenAI adapter uses the configured `GEN_MODEL` and requires `OPENAI_API_KEY`; offline tests use fake providers. Validation establishes genuine retrieved evidence IDs, exact quoted excerpts, and a citation for every returned claim. **It does not prove semantic entailment or factual truth.** This path is internal and is not connected to `/api/query`; the public demo still uses the legacy generator.

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
