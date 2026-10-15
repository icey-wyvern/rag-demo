"""Run retrieval-only evaluation against an already populated local PostgreSQL corpus."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .db import SessionLocal
from .config import get_settings
from .passage_embedding_provider import OpenAIPassageEmbeddingProvider
from .retrieval_eval import compare_modes, evaluate_mode, load_cases
from .versioned_retriever import VersionedRetriever


DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "eval" / "current_retrieval_cases.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mode", choices=("lexical", "semantic", "hybrid", "all"), default="lexical")
    args = parser.parse_args()
    provider = None
    if args.mode != "lexical":
        settings = get_settings()
        provider = OpenAIPassageEmbeddingProvider(model=settings.EMBED_MODEL, api_key=settings.OPENAI_API_KEY)
    retriever = VersionedRetriever(SessionLocal, embedding_provider=provider)
    cases = load_cases(args.dataset)
    if args.mode == "all":
        comparison = compare_modes(retriever, cases)
        print(json.dumps(asdict(comparison), indent=2))
        return 1 if any(summary.failed_case_ids for summary in comparison.summaries.values()) else 0
    summary = evaluate_mode(retriever, cases, args.mode)
    print(json.dumps({"mode": args.mode, "summary": asdict(summary)}, indent=2))
    return 1 if summary.failed_case_ids else 0


if __name__ == "__main__":
    raise SystemExit(main())
