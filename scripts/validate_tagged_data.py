import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tagged_jsonl", type=Path, required=True)
    parser.add_argument("--labels_config", type=Path, default=Path("config/labels.json"))
    args = parser.parse_args()

    labels = set(json.loads(args.labels_config.read_text(encoding="utf-8"))["labels"])
    rows = read_jsonl(args.tagged_jsonl)

    bad_spans = 0
    bad_labels = 0
    overlap_count = 0

    for row in rows:
        text = row["text"]
        entities = row.get("entities", [])

        for i, ent in enumerate(entities):
            start, end, label = int(ent["start"]), int(ent["end"]), str(ent["label"])
            if not (0 <= start < end <= len(text)):
                bad_spans += 1
            if label not in labels:
                bad_labels += 1
            for j in range(i + 1, len(entities)):
                other = entities[j]
                s2, e2 = int(other["start"]), int(other["end"])
                if not (end <= s2 or e2 <= start):
                    overlap_count += 1

    print(f"rows={len(rows)}")
    print(f"bad_spans={bad_spans}")
    print(f"bad_labels={bad_labels}")
    print(f"overlaps={overlap_count}")


if __name__ == "__main__":
    main()
