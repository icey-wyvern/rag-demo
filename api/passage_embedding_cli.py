"""Opt-in indexing of versioned passages with the configured OpenAI model."""

import argparse

from .config import get_settings
from .db import SessionLocal
from .passage_embedding_indexer import PassageEmbeddingIndexer
from .passage_embedding_provider import OpenAIPassageEmbeddingProvider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    settings = get_settings()
    provider = OpenAIPassageEmbeddingProvider(model=settings.EMBED_MODEL, api_key=settings.OPENAI_API_KEY)
    result = PassageEmbeddingIndexer(SessionLocal, provider, batch_size=args.batch_size).index_missing()
    print(
        f"provider={result.provider} model={result.model} "
        f"total={result.total_passages} already_indexed={result.already_indexed} "
        f"newly_indexed={result.newly_indexed} remaining={result.remaining} batches={result.batches}"
    )
    return 0 if result.remaining == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
