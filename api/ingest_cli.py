import argparse
import sys

from .ingest import run_ingest


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and ingest the seeded GOV.UK corpus.")
    parser.add_argument("--force", action="store_true", help="Rebuild the corpus even if it is already ready.")
    args = parser.parse_args()

    result = run_ingest(force=args.force)
    print(f"status={result.status} documents_ready={result.documents_ready}/{result.documents_total}")
    if result.message:
        print(result.message)
    return 0 if result.status == "ready" else 1


if __name__ == "__main__":
    sys.exit(main())
