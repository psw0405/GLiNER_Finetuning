"""
Evaluation script: compute precision, recall and F1 on a validation set.

Usage:
    python -m src.eval \\
        --model_dir outputs/<run>/final \\
        --valid data/valid.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from src.dataio import load_jsonl
from src.labels import LABELS, normalize_label


# ---------------------------------------------------------------------------
# Span matching helpers
# ---------------------------------------------------------------------------

def _gold_spans(record: dict) -> set[tuple[int, int, str]]:
    """Return set of (start, end, label) from a character-span record."""
    return {
        (int(e["start"]), int(e["end"]), normalize_label(e["label"]))
        for e in record.get("entities", [])
    }


def _pred_spans(
    model: object,
    text: str,
    labels: list[str],
    threshold: float,
) -> set[tuple[int, int, str]]:
    """Run model inference and return predicted (start, end, label) spans."""
    entities = model.predict_entities(text, labels, threshold=threshold)
    spans = set()
    for ent in entities:
        # GLiNER returns character-level start/end
        spans.add((ent["start"], ent["end"], normalize_label(ent["label"])))
    return spans


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def evaluate(
    model_dir: Path,
    valid_path: Path,
    threshold: float,
    labels: list[str],
) -> None:
    # Load model
    print(f"Loading model from {model_dir} …")
    from gliner import GLiNER  # noqa: PLC0415

    model = GLiNER.from_pretrained(str(model_dir))
    model.eval()

    # Load data
    records = load_jsonl(valid_path)
    print(f"Evaluating on {len(records)} records …\n")

    overall_tp = overall_fp = overall_fn = 0
    per_label: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    import torch  # noqa: PLC0415

    with torch.no_grad():
        for record in records:
            gold = _gold_spans(record)
            pred = _pred_spans(model, record["text"], labels, threshold)

            for span in pred:
                if span in gold:
                    overall_tp += 1
                    per_label[span[2]]["tp"] += 1
                else:
                    overall_fp += 1
                    per_label[span[2]]["fp"] += 1
            for span in gold:
                if span not in pred:
                    overall_fn += 1
                    per_label[span[2]]["fn"] += 1

    # Overall
    prec, rec, f1 = _prf(overall_tp, overall_fp, overall_fn)
    print(f"{'Label':<28}  {'P':>6}  {'R':>6}  {'F1':>6}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    print("-" * 72)
    print(
        f"{'OVERALL':<28}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}  "
        f"{overall_tp:>5}  {overall_fp:>5}  {overall_fn:>5}"
    )
    print("-" * 72)

    for label in sorted(per_label):
        d = per_label[label]
        lp, lr, lf = _prf(d["tp"], d["fp"], d["fn"])
        print(
            f"  {label:<26}  {lp:>6.3f}  {lr:>6.3f}  {lf:>6.3f}  "
            f"{d['tp']:>5}  {d['fp']:>5}  {d['fn']:>5}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a finetuned GLiNER model on a validation set."
    )
    parser.add_argument(
        "--model_dir", type=Path, required=True,
        help="Path to saved model directory (e.g. outputs/run2/final)."
    )
    parser.add_argument(
        "--valid", type=Path, default=Path("data/valid.jsonl"),
        help="Path to validation JSONL file."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold for entity extraction."
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Override label list (defaults to all 26 project labels)."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.model_dir.exists():
        print(f"[ERROR] model_dir does not exist: {args.model_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.valid.exists():
        print(f"[ERROR] valid path does not exist: {args.valid}", file=sys.stderr)
        sys.exit(1)

    labels = args.labels if args.labels else LABELS

    evaluate(
        model_dir=args.model_dir,
        valid_path=args.valid,
        threshold=args.threshold,
        labels=labels,
    )


if __name__ == "__main__":
    main()
