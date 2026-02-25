import argparse
import json
import random
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


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


def build_char_to_token_map(offsets: list[tuple[int, int]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for token_idx, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        for char_idx in range(start, end):
            mapping[char_idx] = token_idx
    return mapping


def span_to_token_indices(start: int, end: int, char_to_token: dict[int, int]) -> tuple[int, int] | None:
    token_indices = [char_to_token[i] for i in range(start, end) if i in char_to_token]
    if not token_indices:
        return None
    return min(token_indices), max(token_indices)


def convert_row(row: dict[str, Any], tokenizer: AutoTokenizer, max_length: int) -> dict[str, Any] | None:
    text = row["text"]
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    char_to_token = build_char_to_token_map(offsets)

    ner: list[list[Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for ent in row.get("entities", []):
        start = int(ent["start"])
        end = int(ent["end"])
        label = str(ent["label"])
        mapped = span_to_token_indices(start=start, end=end, char_to_token=char_to_token)
        if mapped is None:
            continue
        t_start, t_end = mapped
        key = (t_start, t_end, label)
        if key in seen:
            continue
        seen.add(key)
        ner.append([t_start, t_end, label])

    if not tokens:
        return None

    return {"tokenized_text": tokens, "ner": ner}


def split_dataset(rows: list[dict[str, Any]], valid_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    random.seed(seed)
    copied = rows[:]
    random.shuffle(copied)
    valid_size = int(len(copied) * min(max(valid_ratio, 0.0), 0.5))
    valid = copied[:valid_size]
    train = copied[valid_size:]
    return train, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, default=Path("data/interim/news_10000_tagged.jsonl"))
    parser.add_argument("--train_output", type=Path, default=Path("data/processed/train_gliner.jsonl"))
    parser.add_argument("--valid_output", type=Path, default=Path("data/processed/valid_gliner.jsonl"))
    parser.add_argument("--tokenizer_name", default="urchade/gliner_small-v2")
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tagged_rows = read_jsonl(args.input_jsonl)

    converted: list[dict[str, Any]] = []
    for row in tagged_rows:
        item = convert_row(row=row, tokenizer=tokenizer, max_length=args.max_length)
        if item is not None:
            converted.append(item)

    train_rows, valid_rows = split_dataset(rows=converted, valid_ratio=args.valid_ratio, seed=args.seed)
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.valid_output, valid_rows)

    print(f"train={len(train_rows)} valid={len(valid_rows)}")
    print(f"saved_train={args.train_output}")
    print(f"saved_valid={args.valid_output}")


if __name__ == "__main__":
    main()
