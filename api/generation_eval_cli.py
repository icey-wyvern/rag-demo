"""Offline fixture evaluation of the internal generation and evidence contract."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .generation_eval import DEFAULT_DATASET, evaluate_cases, load_cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    report = evaluate_cases(load_cases(args.dataset))
    summary = asdict(report)
    summary.pop("observations")
    print(json.dumps({
        "evaluation": "deterministic_fake_provider_fixture",
        "report": summary,
        "failed_cases": [asdict(item) for item in report.observations if not item.passed],
    }, indent=2))
    return 0  # Fixture cases may deliberately demonstrate detected failures.


if __name__ == "__main__":
    raise SystemExit(main())
