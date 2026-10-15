"""Opt-in ingestion of the configured GOV.UK guidance into versioned tables."""

from .db import SessionLocal
from .govuk_acquisition import GovUkClient, GovUkNormalizer
from .govuk_sources import SERVICE_STANDARD_PATHS
from .versioned_ingest import VersionedIngestionService
from .versioned_text_parser import parse_guidance_passages


def main() -> int:
    client = GovUkClient()
    normalizer = GovUkNormalizer()
    service = VersionedIngestionService(SessionLocal, parser=parse_guidance_passages)
    failures = 0
    for path in SERVICE_STANDARD_PATHS:
        try:
            normalized = normalizer.normalize(client.fetch(path))
            result = service.ingest(normalized.observed)
        except Exception as exc:
            failures += 1
            print(f"{path}: failed: {type(exc).__name__}: {exc}")
            continue
        print(
            f"{path}: {result.status}; source={result.source_id}; "
            f"snapshot={result.snapshot_id}; passages={result.passage_count}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
