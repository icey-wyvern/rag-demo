# GOV.UK RAG Demo

A retrieval-augmented generation demo over locally observed GOV.UK guidance versions. The stack is FastAPI, PostgreSQL with pgvector, React, and the OpenAI API.

## Features

- Temporal lexical, semantic, and hybrid retrieval over versioned guidance passages
- Evidence-first answers with validated quotations and links to cited sources
- Passage inspection mode for debugging retrieval quality
- Versioned-corpus readiness/status reporting
- Single-command local startup with Docker Compose

## Legacy seeded documents

- National Risk Register 2025
- The UK Government Resilience Framework
- British Energy Security Strategy
- Government Cyber Security Strategy 2022-2030
- Net Zero Strategy: Build Back Greener

## Local development

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Start PostgreSQL and the API/UI:

```bash
docker compose up --build -d db webapi
```

3. Apply the versioned schema and populate the versioned guidance corpus:

```bash
docker compose run --rm webapi alembic upgrade head
docker compose run --rm webapi python -m api.govuk_ingest_cli
docker compose run --rm webapi python -m api.passage_embedding_cli
```

4. Open `http://localhost:8080`.

`webapi` serves the UI immediately. The UI polls versioned-corpus status and unlocks its default hybrid queries after current passages, matching embeddings, and an OpenAI provider configuration exist. The separate `ingest` Compose service still builds the legacy PDF corpus and is not used by `/api/query`.

PostgreSQL runs locally through Docker Compose. Versioned acquisition calls the GOV.UK Content API, and embedding/indexing and generated queries use OpenAI; this deployment path has not been live validated in this slice. Lexical passage queries can run without embeddings if versioned passages exist, and the UI enables Lexical when those passages are ready.

## Commands

Build the corpus again:

```bash
docker compose run --rm ingest python -m api.ingest_cli --force
```

Reset local corpus state:

```bash
docker compose run --rm webapi python -m api.reset_cli --yes
```

Apply the versioned corpus schema used by `/api/query`:

```bash
docker compose run --rm webapi alembic upgrade head
```

## Optional versioned GOV.UK acquisition

After applying the Alembic schema, `python -m api.govuk_ingest_cli` fetches the three configured Service Standard guidance paths in `api/govuk_sources.py` through the GOV.UK Content API. This command populates the versioned corpus read by `/api/query`; it is separate from legacy PDF ingestion.

The Content API's `content_id` supplies stable logical identity. The command extracts guidance body text, then the versioned ingestion service hashes those exact normalized UTF-8 bytes. Each differing observation becomes an immutable local snapshot; an identical observation is unchanged. Source publication/update timestamps are kept separate from the time we observed the response. This does not provide arbitrary historical GOV.UK bodies: local history grows only as differing responses are observed.

The small fixtures under `tests/fixtures` use trimmed fields and guidance text from a public Content API response observed on 15 October 2026. The two bodies are curated subsets for deterministic change tests, not a claim that both were historically published as shown or authoritative current content. Tests use these fixtures without network access. The configured paths have not all been live validated.

## Versioned lexical retrieval and evaluation

`api.versioned_retriever.VersionedRetriever` searches the versioned corpus used by `/api/query`. Its lexical baseline uses PostgreSQL English full-text search (`websearch_to_tsquery`, `to_tsvector`, `ts_rank_cd`) over `passages`, with visible scores and a GIN expression index.

Without `as_of`, each source contributes only its latest locally observed snapshot. With a timezone-aware `as_of`, each contributes its latest snapshot observed at or before that time. Ties use snapshot creation time and ID. This reconstructs what this corpus had observed, not arbitrary historical GOV.UK state. Source publication/update times are provenance, not eligibility cutoffs.

After populating the versioned PostgreSQL corpus, run `python -m api.retrieval_eval_cli` for the small current-source dataset, or pass `--dataset PATH` for a corpus-specific dataset. The separate `eval/fixture_retrieval_cases.json` has seven deterministic scenarios, including illustrative observations A and B at 15 and 16 October 2026. It requires a matching isolated fixture corpus for a meaningful database run; its two bodies are not verified historical publications. The evaluator reports Recall@1, Recall@3, and MRR over cases with expected evidence; cases expecting none pass only when retrieval returns no results and have a separate accuracy. Temporal accuracy covers `as_of` cases, including expected absence. Failed case IDs are reported. No benchmark number is claimed here.

## Opt-in versioned passage embeddings

`passage_embeddings` stores derived vectors separately from immutable passage text and hashes. Each row belongs to one passage and one provider/model pair, so reruns skip existing vectors and a different model can coexist without replacing history. The current column is `vector(1536)` and the indexer rejects other dimensions; this supports the configured `text-embedding-3-small` default, not arbitrary vector sizes.

After applying Alembic migrations and ingesting versioned passages, run `python -m api.passage_embedding_cli` with a configured `OPENAI_API_KEY`. The command uses the configured `EMBED_MODEL`, indexes missing passages in batches, and reports already indexed, newly indexed, and remaining counts. Each batch commits atomically; a failed batch rolls back while earlier batches remain. This makes OpenAI calls when run. Matching embeddings are needed by `/api/query` in its default hybrid or explicit semantic mode; lexical mode can run without them.

## Internal retrieval modes

The versioned retriever accepts `mode="lexical"`, `"semantic"`, or `"hybrid"`. All modes select the same latest locally observed snapshot per source, with the same optional observed-as-of cutoff described above. Semantic retrieval embeds the question through the configured provider, searches only stored embeddings for that provider/model, and orders eligible passages by exact pgvector cosine distance (`<=>`); its Evidence score is **distance, lower is better**. No ANN index is added yet: corpus scale and index tuning have not been measured.

Hybrid retrieval combines lexical and semantic ranks with Reciprocal Rank Fusion, `sum(1 / (60 + rank))`, rather than mixing their incomparable raw scores. Each component requests `min(60, 3 × final K)` candidates; the final Evidence score is an RRF score, higher is better, with both component ranks when present. Its two database rankings share one repeatable-read snapshot so concurrent ingestion cannot mix versions. Run `python -m api.retrieval_eval_cli --mode all` against a populated local corpus to compare the same cases and metrics. The CLI also accepts `lexical`, `semantic`, or `hybrid` individually. Semantic and hybrid modes call the configured query embedding provider, so they require its API key when using OpenAI. Fake-vector tests check mechanics, not semantic quality; real benchmark numbers require a populated corpus and real provider run.

## Internal evidence-first answers

`api.versioned_answer.VersionedAnswerService` can compose the internal retriever with `EvidenceFirstAnswerService`. The service assigns prompt-local IDs (`E1`, `E2`, ...), sends provenance and passage text as clearly delimited reference material, and requests structured claims or an abstention. It resolves each citation against the supplied Evidence and requires a non-empty, exact quotation present in that passage. Duplicate identical citations are collapsed. Any invalid claim rejects the whole answer; the human-readable answer is assembled only from accepted claims.

With no retrieved Evidence, the service abstains without a model call. The model may also abstain with a reason and no claims. The OpenAI adapter uses the configured `GEN_MODEL` and requires `OPENAI_API_KEY`; offline tests use fake providers. Validation establishes genuine retrieved evidence IDs, exact quoted excerpts, and a citation for every returned claim. **It does not prove semantic entailment or factual truth.** `/api/query` now uses this service for generated answers; the legacy generator remains in the repository for later cleanup.

## Internal version comparison

`api.version_comparison.VersionComparisonService` resolves two explicit snapshot IDs, a snapshot and its recorded predecessor, or the latest locally observed snapshot at/before each of two timezone-aware cutoffs for one source. Explicit pairs must be given older first; reversed or equal observation times, self comparisons, missing snapshots, and different sources are rejected. The as-of lookup uses the same observation-time ordering (then creation time and ID) as versioned ingestion. If both cutoffs select the same snapshot, comparison is rejected.

`VersionChangeSet` is derived from immutable snapshots; no change rows are stored. At each passage ordinal, equal hashes mean `unchanged`, differing hashes mean `modified`, an ordinal present only in the newer snapshot means `added`, and one only in the older snapshot means `removed`. Changes are ordered by ordinal and include exact older/newer passage text, hashes, IDs, and snapshot observation provenance. Counts include all four classes and both passage totals. A large insertion or chunk-boundary shift can make subsequent ordinal changes look more substantial than the human conceptual edit.

`VersionChangeSummaryService` optionally turns material changes into separate, explicitly labeled older/newer Evidence and reuses `EvidenceFirstAnswerService` for structured claims, prompt-local IDs, exact quote checks, and rejection of fabricated citations. A claim citing a modified passage must cite its older and newer sides. With no material changes it returns “No observed content changes.” without calling the provider. Textual change does **not** establish semantic contradiction; the model is instructed to describe differences and cite both relevant observations for a claimed reversal. The validator does not prove semantic entailment. Fixtures A and B are illustrative curated observations, not verified historical GOV.UK publication versions. Change summaries remain an internal service; `/api/query` serves ordinary evidence-bound answers.

## Hostile source content in internal answers

The question, passage text, titles, and URLs are untrusted input. Application-owned instruction profiles are sent as system instructions; the provider's user message is one JSON object with a `question` string and an `evidence` array. Each evidence record is serialized by the JSON library with a prompt-local application-assigned ID, separate metadata, and unmodified passage text. Role-looking strings, markup, quotes, and newlines inside source text remain data. Version-change summaries use this same serializer with application-owned older/newer labels.

The provider receives the selected question, evidence, fixed instructions, and structured-output settings; credentials are used only to configure the provider client and are not placed in prompt messages. After generation, the application rejects unknown IDs, missing citations, absent or altered quotes, malformed output, and contradictory status. The deterministic `tests/fixtures/hostile_evidence_cases.json` dataset and `python -m api.security_eval` exercise these rules with fake providers, including a valid answer *about* hostile-looking text. Keyword stripping is not the primary defence because it would delete legitimate source content and would not enforce the output contract. This is risk reduction and mechanical contract enforcement, not proof that a model is immune to prompt injection or that cited claims are true.

## Separate evaluation layers

Retrieval evaluation measures whether relevant versioned evidence was found (Recall@K, MRR, and observed-as-of accuracy). `python -m api.generation_eval_cli` instead runs the reviewable `eval/generation_cases.json` cases through the real internal answer and version-change services with fake providers and pre-supplied Evidence. `python -m api.security_eval` remains a separate adversarial boundary regression suite. None of these metrics is combined into one score. The generation CLI is entirely offline and its numbers are fixture results, not live-model benchmarks.

The generation report distinguishes answered, abstained, and rejected. Status accuracy counts exact status matches; abstention accuracy covers cases expected to answer or abstain, with separate correct-abstention, incorrect-abstention, and incorrect-answer counts. Citation coverage is cited claims divided by accepted answered claims. Citation validity is validated citations divided by citations returned in accepted answers; both normally equal 1 because invalid answers are rejected before a result exists. Expected-evidence accuracy checks whether **all** required provenance selectors were cited (or **any** when a case explicitly requests that mode). Selectors can specify source key, snapshot ID/hash, and passage ID/hash. Temporal accuracy requires exactly the expected cited snapshot set and, for an `as_of` case, observation times no later than that cutoff. A version-change case runs through the real change summary service, including its older/newer citation rule. Metrics with no eligible cases are `null`.

The dataset deliberately includes validly cited answers using the wrong source or observation; these fail expected-evidence or temporal checks while citation validity stays intact. Because the runner receives evidence directly, it reports evidence-selection and version diagnostics, **not** retrieval failures. Its `as_of` field is evaluation context: it does not run retrieval, and historical wording means what this local corpus had observed by that time, not a verified historical GOV.UK publication. The fixtures are synthetic or illustrative. This framework checks application behaviour, citation contracts, required evidence use, and expected rejections; it does not prove semantic entailment, objective truth, answer completeness, or style. No LLM judge or external model call is used.

## Public query API

The UI uses Hybrid retrieval by default; its selector also offers Semantic and Lexical. An optional local date/time is converted to a timezone-aware observed-as-of instant before the request. This selects the latest version the local corpus had observed by that time, not a guaranteed historical GOV.UK state. Generated answers display validated claims with their exact cited quotes, source links, observation times, and available source update times. Abstention is shown explicitly without supporting-source cards. Chunks only remains available to inspect retrieved passages; those passages are labeled as retrieval results, not answer citations.

`POST /api/query` accepts the existing `{ "q": "...", "mode": "generate" | "chunks" }` request (or `question` instead of `q`). `retrieval_mode` defaults to `hybrid` and also accepts `lexical` or `semantic`; `k` defaults to 5. Optional `as_of` must include a timezone and selects the latest version this corpus had observed at or before that instant. Invalid options get a 4xx response. Hybrid/semantic requests fail clearly if the matching semantic index is unavailable; there is no silent lexical or legacy fallback.

Generated queries run versioned retrieval followed by evidence-first structured generation and exact citation/quote validation. The response retains the frontend's `answer`, `citations` (source links with `title`, `url`, `page`), and `chunks` (retrieved passage snippets). It adds `status`, validated `claims`, cited `sources`, passage/snapshot provenance, and small retrieval metadata. Source links come only from cited passages, deduplicated by snapshot/passage; `chunks` are explicitly retrieval inspection data. Abstention returns HTTP 200 with a usable reason and no source links. Chunks mode retrieves versioned passages without calling the generation provider. Provider, validation, and retrieval failures return sanitized errors rather than raw exception text.

`/api/status` keeps the frontend's readiness fields but now reports versioned table state, current passage availability, matching OpenAI embedding availability, and provider configuration. A missing versioned schema reports pending; an unavailable database returns 503. The legacy tables, vectorstore, generator, and ingestion code remain for a later cleanup slice, but `/api/query` no longer uses them. HTTP tests inject fake retrievers and providers; real PostgreSQL/pgvector end-to-end retrieval has not been exercised for this migration.

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

- `api/main.py`: FastAPI endpoints and versioned readiness/dependency wiring
- `api/query_service.py`: versioned retrieval, evidence-first generation, and response serialization
- `api/ingest.py`, `api/store.py`, `api/generator.py`: retained legacy PDF RAG implementation
- `web/src`: status-aware chat UI

## Operational notes

- `GET /health` reports API health and corpus state.
- `GET /api/status` reports versioned-corpus readiness for the UI.
- `POST /api/query` returns `503` if versioned passages or requested semantic prerequisites are unavailable.
- Legacy seed PDFs remain cached under `/app/data/seed-cache` if the separate legacy ingest service is run.

## Notes

- Versioned schema migration, GOV.UK acquisition, and passage embeddings are explicit setup steps above.
- The separate legacy `ingest` service still downloads seeded PDFs when run and does not make `/api/query` ready.
