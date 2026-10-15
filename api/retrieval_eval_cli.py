"""Run retrieval-only evaluation against an already populated local PostgreSQL corpus."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .db import SessionLocal
from .retrieval_eval import evaluate, load_cases
from .versioned_retriever import VersionedRetriever


DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "eval" / "current_retrieval_cases.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    summary = evaluate(VersionedRetriever(SessionLocal), load_cases(args.dataset))
    print(json.dumps(asdict(summary), indent=2))
    return 1 if summary.failed_case_ids else 0


if __name__ == "__main__":
    raise SystemExit(main())
