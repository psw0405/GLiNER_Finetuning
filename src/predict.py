"""
Inference script: load a saved GLiNER model and extract entities from text.

Usage:
    python -m src.predict \\
        --model_dir outputs/run1/final \\
        --text "Alice met Bob in Paris on Monday."

    # Or provide a JSONL file:
    python -m src.predict \\
        --model_dir outputs/run1/final \\
        --input_file data/valid.jsonl \\
        --output_file predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.labels import LABELS


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def load_model(model_dir: str | Path) -> object:
    """Load a finetuned (or base) GLiNER model from *model_dir*."""
    from gliner import GLiNER  # noqa: PLC0415

    model = GLiNER.from_pretrained(str(model_dir))
    model.eval()
    return model


def predict_text(
    model: object,
    text: str,
    labels: list[str],
    threshold: float = 0.5,
) -> list[dict]:
    """
    Run inference on *text* and return a list of entity dicts:
      [{"start": int, "end": int, "label": str, "text": str, "score": float}, ...]
    """
    import torch  # noqa: PLC0415

    with torch.no_grad():
        entities = model.predict_entities(text, labels, threshold=threshold)

    results = []
    for ent in entities:
        results.append(
            {
                "start": ent["start"],
                "end": ent["end"],
                "label": ent["label"],
                "text": ent.get("text", text[ent["start"]: ent["end"]]),
                "score": round(float(ent.get("score", 0.0)), 4),
            }
        )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference with a saved GLiNER model."
    )
    parser.add_argument(
        "--model_dir", type=Path, required=True,
        help="Path to saved model directory (e.g. outputs/run1/final)."
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Single text string to extract entities from."
    )
    parser.add_argument(
        "--input_file", type=Path, default=None,
        help="Input JSONL file with 'text' key per line."
    )
    parser.add_argument(
        "--output_file", type=Path, default=None,
        help="Output JSONL file for predictions (stdout if omitted)."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold for entity extraction."
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Labels to extract (defaults to all 26 project labels)."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.model_dir.exists():
        print(f"[ERROR] model_dir does not exist: {args.model_dir}", file=sys.stderr)
        sys.exit(1)

    if args.text is None and args.input_file is None:
        print("[ERROR] Provide --text or --input_file.", file=sys.stderr)
        sys.exit(1)

    labels = args.labels if args.labels else LABELS

    print(f"Loading model from {args.model_dir} …")
    model = load_model(args.model_dir)

    out_fh = None
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        out_fh = args.output_file.open("w", encoding="utf-8")

    def _emit(record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        if out_fh:
            out_fh.write(line + "\n")
        else:
            print(line)

    try:
        if args.text:
            entities = predict_text(model, args.text, labels, args.threshold)
            _emit({"text": args.text, "entities": entities})
        else:
            if not args.input_file.exists():
                print(f"[ERROR] input_file does not exist: {args.input_file}", file=sys.stderr)
                sys.exit(1)
            with args.input_file.open("r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        print(
                            f"[WARN] line {lineno}: invalid JSON – {exc}", file=sys.stderr
                        )
                        continue
                    text = record.get("text", "")
                    entities = predict_text(model, text, labels, args.threshold)
                    _emit({"text": text, "entities": entities})
    finally:
        if out_fh:
            out_fh.close()


if __name__ == "__main__":
    main()
