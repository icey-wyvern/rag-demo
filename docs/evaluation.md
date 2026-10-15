# Evaluation and validation

The repository keeps retrieval quality, generated-answer contracts, and hostile-content regression checks separate. None is a weighted overall RAG score. Deterministic fixtures and fake providers make the generation and security checks reproducible without API credentials; they are not measurements of a live model's linguistic quality.

## Retrieval evaluation

`api.retrieval_eval` calls the real retriever against an **already populated matching corpus**. The CLI defaults to `eval/current_retrieval_cases.json` and lexical mode; `--mode lexical`, `semantic`, `hybrid`, or `all` selects the mode. Semantic/Hybrid modes need stored embeddings and a query embedding provider. `eval/fixture_retrieval_cases.json` contains illustrative observation A/B cases for a matching isolated fixture corpus. These are not verified historical GOV.UK publication versions.

| Metric | Meaning |
| --- | --- |
| Recall@1, Recall@3 | Fraction of positive cases with expected provenance among the first 1 or 3 results |
| MRR | Mean reciprocal rank of the first expected result in positive cases |
| Temporal accuracy | Fraction of as-of cases whose result uses the expected eligible snapshot, including expected absence |
| No-evidence accuracy | Fraction of cases expecting no evidence that return no results |
| Failed case IDs | Individual cases that miss their deterministic expectation |

The retrieval evaluator needs a real corpus whose source keys and snapshot hashes match its dataset. No retrieval benchmark number is claimed for an arbitrary local corpus. The PostgreSQL integration tests separately validate the real full-text, cosine, hybrid, and temporal query mechanics with designed fixture data.

## Generation and evidence evaluation

Run the offline fixture evaluator from the repository root:

```powershell
.\.venv\Scripts\python.exe -m api.generation_eval_cli
```

`eval/generation_cases.json` supplies evidence and fake provider responses to the **real** `EvidenceFirstAnswerService` or version-change answer service. This isolates answer behaviour from retrieval. The dataset covers answered and multi-evidence claims, provider-requested and no-evidence abstention, temporal provenance, added/removed/modified change claims, fabricated/missing/invalid citations, and representative hostile-text cases.

| Metric | Meaning |
| --- | --- |
| Answer-status accuracy | Exact `answered`, `abstained`, or `rejected` match against expected status |
| Abstention accuracy | Correct answered-versus-abstained decision for cases with an answerability expectation |
| Citation coverage | Claims with at least one validated citation divided by accepted answered claims |
| Citation validity | Validated citations divided by citations returned in accepted answers |
| Expected-evidence accuracy | Accepted answers satisfying all required provenance selectors, or any when the case explicitly requests that mode |
| Temporal evidence accuracy | Accepted temporal cases citing exactly the expected snapshot set within the cutoff |
| Contract pass rate | Cases matching all declared deterministic expectations, with failed case IDs |

Provenance selectors use source key, snapshot ID/hash, or passage ID/hash rather than relying only on prompt-local `E1` IDs. A citation can be structurally valid yet use the wrong source or snapshot; this lowers expected-evidence or temporal accuracy while citation validity remains true. Rejection is distinct from abstention, with categories including unknown evidence, missing citation, invalid quote, malformed output, and contradictory status. The report also distinguishes deterministic no-provider-call abstention, incorrect abstention, and incorrect answer.

The committed 15-case fixture is designed to produce **13 passes and two diagnostic failures**, `wrong_source` and `wrong_snapshot`. Its expected report has answer-status and abstention accuracy 1.0, citation coverage and validity 1.0, expected-evidence accuracy 0.75, temporal accuracy 0.8, and contract pass rate 13/15. These values describe a **fake-provider contract fixture**, not model accuracy. Because evidence is supplied directly, a missing expected citation is reported as evidence-selection or temporal-version failure, not automatically as a retrieval failure. The evaluator cannot prove semantic entailment, complete answers, objective truth, or style.

## Hostile-content security regression

Run the separate offline dataset:

```powershell
.\.venv\Scripts\python.exe -m api.security_eval
```

`tests/fixtures/hostile_evidence_cases.json` contains fake system/developer-role text, fake evidence IDs, requests to omit or forge citations, schema overrides, secret-exfiltration instructions using no real secrets, unsupported claims, markup/fence escapes, and a question-side override. It also includes valid answers **about** hostile-looking text, demonstrating that source text is preserved as data. The evaluator runs these cases through the production answer validator with fake providers and reports case counts, failed IDs, and failure categories. The committed dataset has 11 cases expected to pass its specified accept/reject/abstain outcomes. It checks boundaries and fail-closed handling, not prompt-injection immunity or a live model's response to attacks.

## What real PostgreSQL validates

`tests/integration/` runs on disposable `pgvector/pgvector:pg16`, with local fixtures and fake embedding/answer providers. It exercises Alembic empty → HEAD → base → HEAD; absence of old tables and presence of versioned schema, indexes, constraints, and `vector(1536)`; immutable ingestion; embedding persistence/idempotency; PostgreSQL full-text ranking; pgvector cosine ordering and provider/model filtering; hybrid RRF; observed-as-of selection; readiness; and in-process `/api/query`. This suite passed **9/9 locally and in hosted GitHub Actions** for the preceding cleanup commit. The ordinary backend suite skips DB tests without the explicit integration URL; the dedicated runner supplies only its isolated test URL.

These tests use no live OpenAI, Azure, or GOV.UK application requests. They do not validate production-scale traffic, arbitrary historical reconstruction, live provider quality, semantic truth, or live Azure operation. See the [README](../readme.md#verify-without-api-credentials) for repeatable commands and [architecture](architecture.md) for the data boundaries.
