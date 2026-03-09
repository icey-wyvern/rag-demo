import argparse
import sys

from .db import SessionLocal
from .store import reset_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset the local corpus state and indexed chunks.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm that the local corpus data should be deleted.",
    )
    args = parser.parse_args()
    if not args.yes:
        print("Refusing to reset without --yes.")
        return 1

    db = SessionLocal()
    try:
        reset_corpus(db)
        db.commit()
    finally:
        db.close()

    print("Local corpus state cleared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
