import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="urchade/gliner_small-v2")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--real_news_path", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    python = sys.executable

    run(
        [
            python,
            "scripts/build_news_corpus.py",
            "--output_jsonl",
            "data/raw/news_10000.jsonl",
            "--num_samples",
            str(args.num_samples),
            "--real_ratio",
            "0.6",
            "--seed",
            str(args.seed),
        ]
        + (["--real_news_path", str(args.real_news_path)] if args.real_news_path else [])
    )

    run(
        [
            python,
            "scripts/auto_tag_gliner.py",
            "--model_name",
            args.model_name,
            "--input_jsonl",
            "data/raw/news_10000.jsonl",
            "--output_jsonl",
            "data/interim/news_10000_tagged.jsonl",
            "--labels_config",
            "config/labels.json",
            "--threshold",
            str(args.threshold),
        ]
    )

    run(
        [
            python,
            "scripts/validate_tagged_data.py",
            "--tagged_jsonl",
            "data/interim/news_10000_tagged.jsonl",
            "--labels_config",
            "config/labels.json",
        ]
    )

    run(
        [
            python,
            "scripts/convert_to_gliner.py",
            "--input_jsonl",
            "data/interim/news_10000_tagged.jsonl",
            "--train_output",
            "data/processed/train_gliner.jsonl",
            "--valid_output",
            "data/processed/valid_gliner.jsonl",
            "--tokenizer_name",
            args.model_name,
            "--valid_ratio",
            str(args.valid_ratio),
            "--seed",
            str(args.seed),
        ]
    )

    if not args.skip_train:
        run(
            [
                python,
                "scripts/train_gliner.py",
                "--model_name",
                args.model_name,
                "--train_jsonl",
                "data/processed/train_gliner.jsonl",
                "--valid_jsonl",
                "data/processed/valid_gliner.jsonl",
                "--output_dir",
                "outputs/run_001",
            ]
        )


if __name__ == "__main__":
    main()
