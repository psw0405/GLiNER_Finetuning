"""
Rule-based pseudo-label generator for Korean text.

Reads raw Korean sentences (one per line from a .txt file, or from a CSV)
and emits JSONL records with automatically detected entity spans ready for
GLiNER finetuning.

Usage examples
--------------
# From plain-text file:
python scripts/auto_label_ko.py \\
    --format txt \\
    --input_path data/raw_sentences.txt \\
    --train_out data/train.jsonl \\
    --valid_out data/valid.jsonl \\
    --valid_split 0.1

# From CSV (column name "sentence"):
python scripts/auto_label_ko.py \\
    --format csv \\
    --input_path data/raw.csv \\
    --text_column sentence \\
    --train_out data/train.jsonl \\
    --valid_out data/valid.jsonl \\
    --valid_split 0.1 \\
    --seed 42

# Write a single combined output instead of splitting:
python scripts/auto_label_ko.py \\
    --format txt \\
    --input_path data/raw_sentences.txt \\
    --output_path data/labeled.jsonl

Notes
-----
* Overlap resolution strategy: when two detected spans overlap, the **longer**
  span is kept.  If both spans have the same length, the span produced by the
  recognizer with a **lower priority index** (earlier in RECOGNIZER_PRIORITY)
  is kept.  All other overlapping spans are discarded.
* Dictionary files (data/dicts/*.txt) can be edited to extend built-in lists.
* This produces **noisy pseudo-labels**.  Model quality depends on rule quality.
  Manual review of a sample is strongly recommended before training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path when called as a script
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.labels import LABEL_SET  # noqa: E402

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Span(NamedTuple):
    start: int
    end: int
    label: str


# ---------------------------------------------------------------------------
# Dictionary loading helpers
# ---------------------------------------------------------------------------

_DICTS_DIR = Path(__file__).resolve().parent.parent / "data" / "dicts"


def _load_dict_file(filename: str) -> list[str]:
    """Load a word list from data/dicts/<filename>, ignoring # comment lines."""
    path = _DICTS_DIR / filename
    if not path.exists():
        return []
    entries: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                entries.append(line)
    return entries


def _build_dict_pattern(entries: list[str]) -> re.Pattern | None:
    """Build an alternation regex from a list of strings (longest first)."""
    if not entries:
        return None
    # Sort longest first so the regex engine prefers longer matches.
    sorted_entries = sorted(entries, key=len, reverse=True)
    pattern_str = "|".join(re.escape(e) for e in sorted_entries)
    return re.compile(pattern_str)


# ---------------------------------------------------------------------------
# Recognizer implementations
# ---------------------------------------------------------------------------
# Each recognizer is a callable:  text -> list[Span]
# ---------------------------------------------------------------------------

# --- Date ---

_RE_DATE_FULL = re.compile(
    r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일"
)
_RE_DATE_MONTH_DAY = re.compile(
    r"(?<!\d)\d{1,2}월\s*\d{1,2}일"
)
_RE_DATE_ISO = re.compile(
    r"\d{4}-\d{2}-\d{2}"
)
_RE_DATE_YEAR_MONTH = re.compile(
    r"\d{4}년\s*\d{1,2}월(?!\s*\d)"
)
_RE_DATE_YEAR_ONLY = re.compile(
    r"(?<!\d)\d{4}년(?!\s*\d)"
)


def _recognize_date(text: str) -> list[Span]:
    spans: list[Span] = []
    for pat in (_RE_DATE_FULL, _RE_DATE_MONTH_DAY, _RE_DATE_ISO,
                _RE_DATE_YEAR_MONTH, _RE_DATE_YEAR_ONLY):
        for m in pat.finditer(text):
            spans.append(Span(m.start(), m.end(), "Date"))
    return spans


# --- Time ---

_RE_TIME_CLOCK = re.compile(
    r"(?:오전|오후)?\s*\d{1,2}시(?:\s*\d{1,2}분)?(?:\s*\d{1,2}초)?"
)
_RE_TIME_AMPM_ONLY = re.compile(
    r"(?:오전|오후)\s+\d{1,2}시"
)


def _recognize_time(text: str) -> list[Span]:
    spans: list[Span] = []
    for m in _RE_TIME_CLOCK.finditer(text):
        spans.append(Span(m.start(), m.end(), "Time"))
    return spans


# --- DateDuraion (non-time durations) ---

_RE_DATE_DURATION = re.compile(
    r"\d+\s*(?:"
    r"일간|일\s*동안"
    r"|주간|주\s*동안"
    r"|개월간|개월\s*동안|개월"
    r"|년간|년\s*동안"
    r")"
)


def _recognize_date_duration(text: str) -> list[Span]:
    return [Span(m.start(), m.end(), "DateDuraion") for m in _RE_DATE_DURATION.finditer(text)]


# --- TimeDuration ---

_RE_TIME_DURATION = re.compile(
    r"\d+\s*(?:"
    r"시간(?:\s*(?:동안|간))?"
    r"|분(?:\s*(?:동안|간))?"
    r"|초(?:\s*(?:동안|간))?"
    r")"
)


def _recognize_time_duration(text: str) -> list[Span]:
    return [Span(m.start(), m.end(), "TimeDuration") for m in _RE_TIME_DURATION.finditer(text)]


# --- QuantityAge ---

_RE_AGE = re.compile(
    r"만\s*\d+\s*세"
    r"|\d+\s*세(?!\s*(?:기|기의))"   # avoid "세기" (century)
    r"|\d+\s*살"
)


def _recognize_age(text: str) -> list[Span]:
    return [Span(m.start(), m.end(), "QuantityAge") for m in _RE_AGE.finditer(text)]


# --- QunatityTemperature ---

_RE_TEMP = re.compile(
    r"(?:섭씨|화씨)?\s*-?\d+(?:\.\d+)?\s*(?:°C|℃|°F|℉|도(?=[로가이을를은는에서의과와도만\s,\.;]|$))"
)


def _recognize_temperature(text: str) -> list[Span]:
    return [Span(m.start(), m.end(), "QunatityTemperature") for m in _RE_TEMP.finditer(text)]


# --- QuantityPrice ---

# Real-estate / price context keywords
_RE_REALESTATE_CTX = re.compile(r"㎡|아파트|부동산|분양|매매|전세|월세|평당|당\s*\d")

_RE_PRICE_WON = re.compile(
    r"\d{1,3}(?:,\d{3})+\s*원"           # e.g. 1,320원 / 1,320,000원
    r"|\d+\s*억\s*(?:\d+\s*만\s*)?원?"   # e.g. 1억 / 1억 3천만 원
    r"|\d+\s*만\s*원"                     # e.g. 500만 원
    r"|\d+\s*천\s*원"                     # e.g. 3천 원
    r"|\d+\s*백\s*원"                     # e.g. 5백 원
    r"|\d+\s*원(?!\w)"                    # bare number+원
)
_RE_PRICE_UNIT = re.compile(
    r"\d+\s*억"                           # 억 alone (context required)
    r"|\d+\s*만(?!\s*원)"                 # 만 alone without 원
    r"|\d+\s*천(?!\s*원)"
)


def _recognize_price(text: str) -> list[Span]:
    spans: list[Span] = []
    for m in _RE_PRICE_WON.finditer(text):
        spans.append(Span(m.start(), m.end(), "QuantityPrice"))
    # Bare units (억/만/천) only get QuantityPrice if real-estate context detected
    if _RE_REALESTATE_CTX.search(text):
        for m in _RE_PRICE_UNIT.finditer(text):
            spans.append(Span(m.start(), m.end(), "QuantityPrice"))
    return spans


# --- Organization ---

def _build_org_recognizer() -> re.Pattern:
    """Build org-suffix pattern from dictionary file and hardcoded fallback."""
    suffixes = _load_dict_file("org_suffix.txt")
    if not suffixes:
        # Minimal hardcoded fallback
        suffixes = ["부", "청", "처", "위원회", "공사", "공단", "연구원", "재단",
                    "협회", "은행", "증권"]
    # Sort longest first
    suffixes_sorted = sorted(suffixes, key=len, reverse=True)
    suf_pattern = "|".join(re.escape(s) for s in suffixes_sorted)
    # Match: at least 2 Hangul chars forming a word, ending in suffix
    return re.compile(
        r"[가-힣]{2,}(?:" + suf_pattern + r")(?![가-힣])"
    )


_RE_ORG = _build_org_recognizer()

# Hardcoded exact-match org names (well-known Korean orgs that may not match suffix rules)
_ORG_EXACT: frozenset[str] = frozenset([
    "기획재정부", "교육부", "국방부", "외교부", "법무부", "행정안전부",
    "보건복지부", "환경부", "고용노동부", "여성가족부", "국토교통부",
    "해양수산부", "중소벤처기업부", "과학기술정보통신부", "문화체육관광부",
    "농림축산식품부", "산업통상자원부",
    "국회", "대법원", "헌법재판소", "감사원", "국가정보원",
    "한국은행", "금융감독원", "금융위원회",
    "삼성전자", "현대자동차", "SK하이닉스", "LG전자", "카카오", "네이버",
    "롯데", "포스코", "한화", "KT",
])


def _recognize_org(text: str) -> list[Span]:
    spans: list[Span] = []
    for m in _RE_ORG.finditer(text):
        spans.append(Span(m.start(), m.end(), "Organization"))
    for name in _ORG_EXACT:
        start = 0
        while True:
            idx = text.find(name, start)
            if idx == -1:
                break
            spans.append(Span(idx, idx + len(name), "Organization"))
            start = idx + 1
    return spans


# --- LocationCountry ---

def _build_country_recognizer() -> re.Pattern | None:
    entries = _load_dict_file("location_country.txt")
    if not entries:
        entries = ["미국", "중국", "일본", "한국", "러시아", "프랑스", "독일",
                   "영국", "북한", "캐나다", "호주", "인도"]
    return _build_dict_pattern(entries)


_RE_COUNTRY = _build_country_recognizer()


def _recognize_country(text: str) -> list[Span]:
    if _RE_COUNTRY is None:
        return []
    return [Span(m.start(), m.end(), "LocationCountry") for m in _RE_COUNTRY.finditer(text)]


# --- LocationCity ---

def _build_city_recognizer() -> re.Pattern | None:
    entries = _load_dict_file("location_city.txt")
    if not entries:
        entries = ["서울", "부산", "인천", "대구", "대전", "광주", "울산",
                   "세종", "제주", "해운대", "송도"]
    return _build_dict_pattern(entries)


_RE_CITY = _build_city_recognizer()


def _recognize_city(text: str) -> list[Span]:
    if _RE_CITY is None:
        return []
    return [Span(m.start(), m.end(), "LocationCity") for m in _RE_CITY.finditer(text)]


# ---------------------------------------------------------------------------
# Recognizer registry & priority
# ---------------------------------------------------------------------------
# Lower index = higher priority (kept when two spans of equal length conflict).
RECOGNIZER_PRIORITY: list[str] = [
    "QunatityTemperature",   # be specific before bare numbers
    "QuantityAge",
    "QuantityPrice",
    "DateDuraion",
    "TimeDuration",
    "Date",
    "Time",
    "Organization",
    "LocationCountry",
    "LocationCity",
]

# Map label -> priority index (lower = higher priority)
_LABEL_PRIORITY: dict[str, int] = {label: i for i, label in enumerate(RECOGNIZER_PRIORITY)}


def _all_recognizers(text: str) -> list[Span]:
    """Run all recognizers on *text* and return the merged, de-overlapped spans."""
    raw: list[Span] = []
    raw.extend(_recognize_temperature(text))
    raw.extend(_recognize_age(text))
    raw.extend(_recognize_price(text))
    raw.extend(_recognize_date_duration(text))
    raw.extend(_recognize_time_duration(text))
    raw.extend(_recognize_date(text))
    raw.extend(_recognize_time(text))
    raw.extend(_recognize_org(text))
    raw.extend(_recognize_country(text))
    raw.extend(_recognize_city(text))
    return raw


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------

def _resolve_overlaps(spans: list[Span]) -> list[Span]:
    """
    Remove overlapping spans using the following strategy:

    1. For each pair of overlapping spans, keep the **longer** one.
    2. If both have the same character length, keep the one whose label has a
       **lower** index in RECOGNIZER_PRIORITY (earlier = higher priority).
    3. Ties on both length and priority are broken by keeping the span that
       starts earlier in the text.

    The algorithm is greedy (iterative) and runs in O(n^2) in the worst case,
    which is fine for the typical sentence-level span counts produced here.
    """
    if not spans:
        return []

    def _key(s: Span) -> tuple[int, int, int]:
        length = s.end - s.start
        priority = _LABEL_PRIORITY.get(s.label, len(RECOGNIZER_PRIORITY))
        return (-length, priority, s.start)

    sorted_spans = sorted(spans, key=_key)

    kept: list[Span] = []
    for candidate in sorted_spans:
        overlap = False
        for accepted in kept:
            # Spans overlap when they share at least one character
            if candidate.start < accepted.end and candidate.end > accepted.start:
                overlap = True
                break
        if not overlap:
            kept.append(candidate)

    # Sort kept spans by position for readability
    kept.sort(key=lambda s: s.start)
    return kept


# ---------------------------------------------------------------------------
# Full pipeline for a single sentence
# ---------------------------------------------------------------------------

def label_sentence(text: str) -> dict:
    """
    Apply all rule-based recognizers to *text* and return a JSONL-ready dict:
    {"text": ..., "entities": [{"start": ..., "end": ..., "label": ...}, ...]}

    Spans are guaranteed to be:
    - Non-overlapping (via _resolve_overlaps)
    - Correctly bounded within *text*
    - Non-empty / non-whitespace
    - Using only labels from LABEL_SET
    """
    raw_spans = _all_recognizers(text)

    # Basic sanity filtering before overlap resolution
    valid: list[Span] = []
    for s in raw_spans:
        if not (0 <= s.start < s.end <= len(text)):
            continue
        if not text[s.start:s.end].strip():
            continue
        if s.label not in LABEL_SET:
            continue
        valid.append(s)

    final = _resolve_overlaps(valid)
    entities = [{"start": s.start, "end": s.end, "label": s.label} for s in final]
    return {"text": text, "entities": entities}


# ---------------------------------------------------------------------------
# Input readers
# ---------------------------------------------------------------------------

def _read_txt(path: Path) -> list[str]:
    """Read one sentence per non-empty line from a plain-text file."""
    sentences: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.strip():
                sentences.append(line)
    return sentences


def _read_csv(path: Path, text_column: str) -> list[str]:
    """Read sentences from a CSV file using the specified column name."""
    sentences: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or text_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{text_column}' not found in CSV. "
                f"Available columns: {list(reader.fieldnames or [])}"
            )
        for row in reader:
            text = row[text_column]
            if text and text.strip():
                sentences.append(text)
    return sentences


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records → {path}")


# ---------------------------------------------------------------------------
# Train / valid split
# ---------------------------------------------------------------------------

def _split(
    records: list[dict],
    valid_split: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Deterministically split records into (train, valid)."""
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_valid = max(1, round(len(records) * valid_split))
    valid_idx = set(indices[:n_valid])
    train = [records[i] for i in range(len(records)) if i not in valid_idx]
    valid = [records[i] for i in range(len(records)) if i in valid_idx]
    return train, valid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Rule-based pseudo-label generator for Korean sentences. "
            "Reads raw text and emits JSONL for GLiNER finetuning."
        )
    )
    p.add_argument(
        "--input_path", required=True, type=Path,
        help="Path to input file (plain-text or CSV)."
    )
    p.add_argument(
        "--format", choices=["txt", "csv"], default="txt",
        help="Input file format (default: txt)."
    )
    p.add_argument(
        "--text_column", default="text",
        help="Column name for text when --format csv (default: text)."
    )
    p.add_argument(
        "--output_path", type=Path, default=None,
        help="Write all labeled records to a single JSONL file (skips split)."
    )
    p.add_argument(
        "--train_out", type=Path, default=Path("data/train.jsonl"),
        help="Output path for training split (default: data/train.jsonl)."
    )
    p.add_argument(
        "--valid_out", type=Path, default=Path("data/valid.jsonl"),
        help="Output path for validation split (default: data/valid.jsonl)."
    )
    p.add_argument(
        "--valid_split", type=float, default=0.1,
        help="Fraction of records to use for validation (default: 0.1)."
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic split (default: 42)."
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input_path.exists():
        print(f"[ERROR] Input file not found: {args.input_path}", file=sys.stderr)
        sys.exit(1)

    # Read sentences
    print(f"Reading sentences from: {args.input_path}  (format={args.format})")
    if args.format == "txt":
        sentences = _read_txt(args.input_path)
    else:
        sentences = _read_csv(args.input_path, args.text_column)
    print(f"  Loaded {len(sentences)} sentences")

    if not sentences:
        print("[ERROR] No sentences found in input file.", file=sys.stderr)
        sys.exit(1)

    # Label
    print("Applying rule-based recognizers …")
    records = [label_sentence(s) for s in sentences]
    total_spans = sum(len(r["entities"]) for r in records)
    print(f"  Generated {total_spans} entity spans across {len(records)} records")

    # Output
    if args.output_path is not None:
        _write_jsonl(records, args.output_path)
    else:
        if not (0.0 < args.valid_split < 1.0):
            print(
                f"[ERROR] --valid_split must be in (0, 1), got {args.valid_split}",
                file=sys.stderr,
            )
            sys.exit(1)
        train_records, valid_records = _split(records, args.valid_split, args.seed)
        print(f"  Split: {len(train_records)} train / {len(valid_records)} valid")
        _write_jsonl(train_records, args.train_out)
        _write_jsonl(valid_records, args.valid_out)

    print("\nDone.  Remember: these are noisy pseudo-labels.")
    print("Review a sample before using for training.")


if __name__ == "__main__":
    main()
