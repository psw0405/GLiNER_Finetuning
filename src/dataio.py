"""
JSONL loader and dataset validation for the GLiNER finetuning pipeline.

Expected record format:
  {"text": "...", "entities": [{"start": int, "end": int, "label": "..."}]}

  start/end are 0-based character indices; end is exclusive (text[start:end] == span).
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from src.labels import LABEL_SET


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load all non-empty lines from a JSONL file and return as a list of dicts."""
    records: list[dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON – {exc}") from exc
    return records


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_records(
    records: list[dict[str, Any]],
    source_name: str = "<dataset>",
    *,
    warn_overlaps: bool = True,
) -> list[str]:
    """
    Validate a list of records.

    Returns a list of error strings (empty = all good).
    Prints a warning for overlapping spans when *warn_overlaps* is True.
    """
    errors: list[str] = []
    overlap_count = 0

    for idx, record in enumerate(records):
        prefix = f"{source_name}[{idx}]"

        # Required key: text
        if "text" not in record:
            errors.append(f"{prefix}: missing key 'text'")
            continue
        text: str = record["text"]
        if not isinstance(text, str):
            errors.append(f"{prefix}: 'text' must be a string")
            continue

        # Required key: entities
        if "entities" not in record:
            errors.append(f"{prefix}: missing key 'entities'")
            continue
        entities = record["entities"]
        if not isinstance(entities, list):
            errors.append(f"{prefix}: 'entities' must be a list")
            continue

        spans: list[tuple[int, int]] = []
        for eidx, ent in enumerate(entities):
            eprefix = f"{prefix}.entities[{eidx}]"

            # Required entity keys
            for key in ("start", "end", "label"):
                if key not in ent:
                    errors.append(f"{eprefix}: missing key '{key}'")
            if any(k not in ent for k in ("start", "end", "label")):
                continue

            start: int = ent["start"]
            end: int = ent["end"]
            label: str = ent["label"]

            # Type checks
            if not isinstance(start, int) or not isinstance(end, int):
                errors.append(f"{eprefix}: 'start' and 'end' must be integers")
                continue
            if not isinstance(label, str):
                errors.append(f"{eprefix}: 'label' must be a string")
                continue

            # Bounds check
            if not (0 <= start < end <= len(text)):
                errors.append(
                    f"{eprefix}: invalid span [{start},{end}) for text of length {len(text)}"
                )
                continue

            # Non-empty span
            span_text = text[start:end]
            if not span_text.strip():
                errors.append(f"{eprefix}: span text '{span_text!r}' is empty or whitespace-only")
                continue

            # Label membership
            if label not in LABEL_SET:
                errors.append(
                    f"{eprefix}: unknown label '{label}' (not in LABELS list)"
                )

            spans.append((start, end))

        # Overlap detection
        if warn_overlaps and len(spans) > 1:
            for i in range(len(spans)):
                for j in range(i + 1, len(spans)):
                    s1, e1 = spans[i]
                    s2, e2 = spans[j]
                    if not (e1 <= s2 or e2 <= s1):
                        overlap_count += 1

    if warn_overlaps and overlap_count:
        warnings.warn(
            f"{source_name}: {overlap_count} overlapping span pair(s) detected. "
            "GLiNER handles overlapping spans during training, but review is recommended.",
            stacklevel=2,
        )

    return errors


def validate_file(
    path: str | Path,
    *,
    warn_overlaps: bool = True,
) -> list[str]:
    """Load a JSONL file and validate its records. Returns list of error strings."""
    records = load_jsonl(path)
    return validate_records(records, source_name=str(path), warn_overlaps=warn_overlaps)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def dataset_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute basic statistics for a list of records."""
    label_counts: Counter[str] = Counter()
    span_count = 0
    text_lengths: list[int] = []

    for record in records:
        text_lengths.append(len(record.get("text", "")))
        for ent in record.get("entities", []):
            label_counts[ent.get("label", "<missing>")] += 1
            span_count += 1

    return {
        "num_records": len(records),
        "num_spans": span_count,
        "avg_text_length": sum(text_lengths) / max(len(text_lengths), 1),
        "label_counts": dict(label_counts.most_common()),
    }


def print_stats(stats: dict[str, Any], title: str = "Dataset") -> None:
    """Pretty-print dataset statistics."""
    print(f"\n--- {title} ---")
    print(f"  Records : {stats['num_records']}")
    print(f"  Spans   : {stats['num_spans']}")
    print(f"  Avg len : {stats['avg_text_length']:.1f} chars")
    print("  Label counts:")
    for label, count in stats["label_counts"].items():
        print(f"    {label:<28} {count}")
