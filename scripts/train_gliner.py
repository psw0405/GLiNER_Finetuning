import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def call_training_function(args: argparse.Namespace) -> None:
    training_module = importlib.import_module("gliner.training")
    candidates = ["train", "run_train", "fit"]
    selected = None
    for name in candidates:
        if hasattr(training_module, name):
            selected = getattr(training_module, name)
            break
    if selected is None:
        raise RuntimeError("gliner.training 모듈에서 train/run_train/fit 함수를 찾지 못했습니다.")

    signature = inspect.signature(selected)
    call_kwargs: dict[str, Any] = {}
    values = {
        "model_name": args.model_name,
        "model": args.model_name,
        "train_file": str(args.train_jsonl),
        "train_path": str(args.train_jsonl),
        "valid_file": str(args.valid_jsonl),
        "eval_file": str(args.valid_jsonl),
        "valid_path": str(args.valid_jsonl),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "per_device_train_batch_size": args.batch_size,
        "seed": args.seed,
    }
    for key in signature.parameters:
        if key in values:
            call_kwargs[key] = values[key]

    selected(**call_kwargs)


def call_model_method(args: argparse.Namespace) -> None:
    gliner_module = importlib.import_module("gliner")
    model_cls = getattr(gliner_module, "GLiNER")
    model = model_cls.from_pretrained(args.model_name)
    candidates = ["train_model", "fit", "train"]

    train_rows = load_jsonl(args.train_jsonl)
    valid_rows = load_jsonl(args.valid_jsonl)

    for method_name in candidates:
        if hasattr(model, method_name):
            method = getattr(model, method_name)
            signature = inspect.signature(method)
            values = {
                "train_data": train_rows,
                "training_data": train_rows,
                "train_dataset": train_rows,
                "eval_data": valid_rows,
                "valid_data": valid_rows,
                "eval_dataset": valid_rows,
                "epochs": args.epochs,
                "num_epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "output_dir": str(args.output_dir),
                "seed": args.seed,
            }
            call_kwargs: dict[str, Any] = {}
            for key in signature.parameters:
                if key in values:
                    call_kwargs[key] = values[key]
            method(**call_kwargs)
            if hasattr(model, "save_pretrained"):
                args.output_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(args.output_dir / "final"))
            return

    raise RuntimeError("GLiNER 모델 객체에서 train_model/fit/train 메서드를 찾지 못했습니다.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="urchade/gliner_small-v2")
    parser.add_argument("--train_jsonl", type=Path, default=Path("data/processed/train_gliner.jsonl"))
    parser.add_argument("--valid_jsonl", type=Path, default=Path("data/processed/valid_gliner.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/run_001"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.train_jsonl.exists() or not args.valid_jsonl.exists():
        raise FileNotFoundError("train/valid GLiNER JSONL 파일이 필요합니다.")

    try:
        call_training_function(args)
    except Exception as first_error:
        try:
            call_model_method(args)
        except Exception as second_error:
            raise RuntimeError(
                "GLiNER 학습 API 호출에 실패했습니다. gliner 버전을 확인하고 README의 실행 가이드를 따라주세요. "
                f"training_module_error={first_error} model_method_error={second_error}"
            )

    print(f"training finished: {args.output_dir}")


if __name__ == "__main__":
    main()
