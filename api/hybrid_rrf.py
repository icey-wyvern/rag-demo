"""Deterministic passage-level Reciprocal Rank Fusion."""

from collections.abc import Sequence
from dataclasses import replace

from .evidence import Evidence


DEFAULT_RRF_CONSTANT = 60
CANDIDATE_MULTIPLIER = 3
MAX_CANDIDATES = 60


def candidate_depth(k: int) -> int:
    return min(MAX_CANDIDATES, CANDIDATE_MULTIPLIER * k)


def fuse_rrf(
    lexical: Sequence[Evidence], semantic: Sequence[Evidence], *,
    k: int, constant: int = DEFAULT_RRF_CONSTANT,
) -> list[Evidence]:
    """Fuse ranks, never incomparable lexical and vector scores."""
    if isinstance(constant, bool) or not isinstance(constant, int) or not 1 <= constant <= 100:
        raise ValueError("RRF constant must be an integer between 1 and 100")
    scores: dict[object, float] = {}
    examples: dict[object, Evidence] = {}
    ranks: dict[object, dict[str, int]] = {}
    for name, results in (("lexical", lexical), ("semantic", semantic)):
        seen = set()
        for evidence in results:
            key = evidence.passage_id
            if key in seen:
                continue
            rank = len(seen) + 1
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (constant + rank)
            examples.setdefault(key, evidence)
            ranks.setdefault(key, {})[name] = rank
    ordered = sorted(
        scores,
        key=lambda key: (
            -scores[key], examples[key].source_key,
            examples[key].passage_ordinal, examples[key].passage_id,
        ),
    )[:k]
    return [
        replace(
            examples[key], retrieval_method="hybrid_rrf", score=scores[key], rank=rank,
            score_kind="rrf", lexical_rank=ranks[key].get("lexical"),
            semantic_rank=ranks[key].get("semantic"),
        )
        for rank, key in enumerate(ordered, 1)
    ]
