import argparse
import json
from pathlib import Path
from typing import Any

from gliner import GLiNER


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_label_config(path: Path) -> tuple[list[str], dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data["labels"]
    aliases = data.get("aliases", {})
    return labels, aliases


def normalize_label(label: str, aliases: dict[str, str]) -> str:
    return aliases.get(label, label)


def is_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or b_end <= a_start)


def resolve_overlaps(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_entities = sorted(entities, key=lambda x: (x["start"], -(x["end"] - x["start"]), -x.get("score", 0.0)))
    selected: list[dict[str, Any]] = []
    for candidate in sorted_entities:
        overlapped = False
        for idx, current in enumerate(selected):
            if is_overlap(candidate["start"], candidate["end"], current["start"], current["end"]):
                overlapped = True
                if candidate.get("score", 0.0) > current.get("score", 0.0):
                    selected[idx] = candidate
                break
        if not overlapped:
            selected.append(candidate)
    return sorted(selected, key=lambda x: (x["start"], x["end"]))


def predict_entities(model: GLiNER, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
    try:
        predicted = model.predict_entities(text, labels=labels, threshold=threshold)
    except TypeError:
        predicted = model.predict_entities(text, labels=labels)

    result: list[dict[str, Any]] = []
    for item in predicted:
        result.append(
            {
                "start": int(item["start"]),
                "end": int(item["end"]),
                "label": str(item["label"]),
                "text": str(item.get("text", text[item["start"] : item["end"]])),
                "score": float(item.get("score", 0.0)),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="urchade/gliner_small-v2")
    parser.add_argument("--input_jsonl", type=Path, default=Path("data/raw/news_10000.jsonl"))
    parser.add_argument("--output_jsonl", type=Path, default=Path("data/interim/news_10000_tagged.jsonl"))
    parser.add_argument("--labels_config", type=Path, default=Path("config/labels.json"))
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    labels, aliases = load_label_config(args.labels_config)
    label_set = set(labels)

    model = GLiNER.from_pretrained(args.model_name)
    corpus = read_jsonl(args.input_jsonl)

    if args.max_samples is not None:
        corpus = corpus[: args.max_samples]

    output_rows: list[dict[str, Any]] = []
    for row in corpus:
        text = row["text"]
        predicted = predict_entities(model=model, text=text, labels=labels, threshold=args.threshold)

        normalized: list[dict[str, Any]] = []
        for ent in predicted:
            ent["label"] = normalize_label(ent["label"], aliases)
            if ent["label"] not in label_set:
                continue
            if not (0 <= ent["start"] < ent["end"] <= len(text)):
                continue
            normalized.append(ent)

        cleaned = resolve_overlaps(normalized)

        output_rows.append(
            {
                "id": row.get("id"),
                "text": text,
                "entities": cleaned,
                "source": row.get("source", "unknown"),
            }
        )

    write_jsonl(args.output_jsonl, output_rows)
    print(f"saved={len(output_rows)} path={args.output_jsonl}")


if __name__ == "__main__":
    main()
