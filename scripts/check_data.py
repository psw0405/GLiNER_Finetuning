"""
Convenience wrapper around src.dataio validation.

Usage:
    python scripts/check_data.py --train data/train.jsonl --valid data/valid.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/check_data.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataio import dataset_stats, load_jsonl, print_stats, validate_records  # noqa: E402


def check_file(path: Path, name: str) -> bool:
    """Validate *path* and print statistics. Returns True if no errors."""
    print(f"\nChecking {name}: {path}")
    if not path.exists():
        print(f"  [ERROR] File not found: {path}", file=sys.stderr)
        return False

    records = load_jsonl(path)
    errors = validate_records(records, source_name=str(path), warn_overlaps=True)

    if errors:
        print(f"  {len(errors)} validation error(s):")
        for err in errors[:50]:
            print(f"    {err}")
        if len(errors) > 50:
            print(f"    … and {len(errors) - 50} more")
        return False

    print(f"  ✓ No errors found ({len(records)} records)")
    print_stats(dataset_stats(records), title=name)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate train/valid JSONL dataset files.")
    parser.add_argument("--train", type=Path, required=True, help="Path to training JSONL.")
    parser.add_argument("--valid", type=Path, required=True, help="Path to validation JSONL.")
    args = parser.parse_args()

    ok_train = check_file(args.train, "TRAIN")
    ok_valid = check_file(args.valid, "VALID")

    print()
    if ok_train and ok_valid:
        print("All checks passed.")
    else:
        print("One or more checks failed – see errors above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
