"""
Training entrypoint for GLiNER finetuning.

Usage:
    python -m src.train \\
        --train data/train.jsonl \\
        --valid data/valid.jsonl \\
        --output_dir outputs/run1

Character-level spans in the JSONL are converted to whitespace-tokenised
word-level spans, which is the format GLiNER's Trainer expects.
"""

from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.dataio import dataset_stats, load_jsonl, print_stats, validate_records
from src.labels import LABELS, LABEL_SET, normalize_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _whitespace_tokenize(text: str) -> tuple[list[str], list[int], list[int]]:
    """Split text on whitespace and return (words, start_chars, end_chars)."""
    words, starts, ends = [], [], []
    for m in re.finditer(r"\S+", text):
        words.append(m.group())
        starts.append(m.start())
        ends.append(m.end())
    return words, starts, ends


def char_span_to_word_span(
    char_start: int,
    char_end: int,
    word_starts: list[int],
    word_ends: list[int],
) -> tuple[int, int] | None:
    """
    Convert an exclusive character span [char_start, char_end) to an
    inclusive word-index span [word_start, word_end].

    Returns None when the span cannot be mapped to any word.
    """
    ws: int | None = None
    we: int | None = None

    for i, (s, e) in enumerate(zip(word_starts, word_ends)):
        # word overlaps with the character span
        if s < char_end and e > char_start:
            if ws is None:
                ws = i
            we = i

    if ws is None or we is None:
        return None
    return ws, we


def convert_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """
    Convert one character-span record to GLiNER's word-span format:
      {"tokenized_text": [...], "ner": [[word_start, word_end, label], ...]}

    Returns None when the text produces no tokens.
    """
    text: str = record["text"]
    words, starts, ends = _whitespace_tokenize(text)
    if not words:
        return None

    ner: list[list[Any]] = []
    for ent in record.get("entities", []):
        result = char_span_to_word_span(
            ent["start"], ent["end"], starts, ends
        )
        if result is not None:
            ws, we = result
            ner.append([ws, we, normalize_label(ent["label"])])

    return {"tokenized_text": words, "ner": ner}


def prepare_gliner_data(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert a list of character-span records to GLiNER training format."""
    converted = []
    for rec in records:
        gliner_rec = convert_record(rec)
        if gliner_rec is not None:
            converted.append(gliner_rec)
    return converted


def normalize_entity_labels(
    records: list[dict[str, Any]],
    source_name: str,
) -> tuple[int, dict[str, int]]:
    """Normalize entity labels in-place and return (changed_count, unknown_counts)."""
    changed_count = 0
    unknown_counts: Counter[str] = Counter()

    for record in records:
        entities = record.get("entities", [])
        if not isinstance(entities, list):
            continue

        for ent in entities:
            label = ent.get("label")
            if not isinstance(label, str):
                continue

            normalized = normalize_label(label)
            if normalized != label:
                ent["label"] = normalized
                changed_count += 1

            if normalized not in LABEL_SET:
                unknown_counts[normalized] += 1

    if changed_count:
        print(f"Normalized {changed_count} label(s) in {source_name}")

    if unknown_counts:
        print(f"[WARN] Unknown labels remain in {source_name} after normalization:")
        for label, count in unknown_counts.most_common(20):
            print(f"  - {label}: {count}")

    return changed_count, dict(unknown_counts)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model_name: str,
    train_path: Path,
    valid_path: Path,
    output_dir: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    max_length: int,
    seed: int,
) -> None:
    set_seed(seed)

    # ------------------------------------------------------------------
    # Load & validate data
    # ------------------------------------------------------------------
    print("Loading training data …")
    train_records = load_jsonl(train_path)
    valid_records = load_jsonl(valid_path)

    normalize_entity_labels(train_records, str(train_path))
    normalize_entity_labels(valid_records, str(valid_path))

    for records, name in [(train_records, str(train_path)), (valid_records, str(valid_path))]:
        errors = validate_records(records, source_name=name)
        if errors:
            for e in errors[:20]:
                print(f"  [ERROR] {e}", file=sys.stderr)
            if len(errors) > 20:
                print(f"  … and {len(errors) - 20} more errors", file=sys.stderr)
            print("Fix validation errors before training.", file=sys.stderr)
            sys.exit(1)

    print_stats(dataset_stats(train_records), "Train")
    print_stats(dataset_stats(valid_records), "Valid")

    train_data = prepare_gliner_data(train_records)
    valid_data = prepare_gliner_data(valid_records)
    print(f"\nConverted {len(train_data)}/{len(train_records)} train records")
    print(f"Converted {len(valid_data)}/{len(valid_records)} valid records")

    if not train_data:
        print("No training data after conversion.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"\nLoading base model: {model_name} …")
    from gliner import GLiNER  # noqa: PLC0415

    model = GLiNER.from_pretrained(model_name)

    device = torch.device("cpu")
    model = model.to(device)

    # ------------------------------------------------------------------
    # Training via GLiNER's Trainer API (with manual loop fallback)
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _train_via_trainer(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            output_dir=output_dir,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=seed,
        )
    except Exception as trainer_exc:  # noqa: BLE001
        print(
            f"[WARN] GLiNER Trainer API failed ({trainer_exc!r}); "
            "falling back to manual training loop. "
            "Check that your gliner version (>=0.2.22) is installed and "
            "the training data is in the correct word-span format.",
        )
        _train_manual(
            model=model,
            train_data=train_data,
            valid_data=valid_data,
            output_dir=output_dir,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Save model + metadata
    # ------------------------------------------------------------------
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))

    # Persist run config alongside model for reproducibility
    run_config = {
        "base_model": model_name,
        "labels": LABELS,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "max_length": max_length,
        "seed": seed,
    }
    (final_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nModel saved to: {final_dir}")
    print("Training complete.")


# ---------------------------------------------------------------------------
# Trainer-API path
# ---------------------------------------------------------------------------

def _train_via_trainer(
    model: Any,
    train_data: list[dict],
    valid_data: list[dict],
    output_dir: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> None:
    """Use GLiNER's built-in Trainer if available."""
    from gliner.training import Trainer, TrainingArguments  # noqa: PLC0415
    from gliner.data_processing.collator import DataCollator  # noqa: PLC0415

    arg_params = inspect.signature(TrainingArguments.__init__).parameters
    train_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": lr,
        "weight_decay": 0.01,
        "others_lr": lr * 2,
        "others_weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "dataloader_num_workers": 0,
        "fp16": False,
        "bf16": False,
        "seed": seed,
        "report_to": "none",
        "logging_steps": 10,
        "remove_unused_columns": False,
    }

    # Transformers / GLiNER version compatibility.
    if "evaluation_strategy" in arg_params:
        train_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in arg_params:
        train_kwargs["eval_strategy"] = "epoch"

    if "no_cuda" in arg_params:
        train_kwargs["no_cuda"] = True
    elif "use_cpu" in arg_params:
        train_kwargs["use_cpu"] = True

    if "do_train" in arg_params:
        train_kwargs["do_train"] = True
    if "do_eval" in arg_params:
        train_kwargs["do_eval"] = True

    filtered_kwargs = {k: v for k, v in train_kwargs.items() if k in arg_params}
    training_args = TrainingArguments(**filtered_kwargs)

    data_collator = DataCollator(
        model.config,
        data_processor=getattr(model, "data_processor", None),
        prepare_labels=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=data_collator,
    )
    trainer.train()

    global_step = getattr(getattr(trainer, "state", None), "global_step", None)
    if isinstance(global_step, int) and global_step == 0:
        raise RuntimeError(
            "GLiNER Trainer finished with zero optimization steps. "
            "Falling back to manual loop."
        )


# ---------------------------------------------------------------------------
# Manual loop fallback
# ---------------------------------------------------------------------------

def _train_manual(
    model: Any,
    train_data: list[dict],
    valid_data: list[dict],
    output_dir: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> None:
    """Minimal manual PyTorch training loop using GLiNER's data collator."""
    from gliner.data_processing.collator import DataCollator  # noqa: PLC0415

    collator = DataCollator(
        model.config,
        data_processor=getattr(model, "data_processor", None),
        prepare_labels=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * max(1, len(train_data) // batch_size)
    )

    model.train()
    total_steps = 0
    for epoch in range(1, epochs + 1):
        # Shuffle each epoch
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        epoch_loss = 0.0
        steps = 0
        collate_failures = 0
        forward_failures = 0

        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start : batch_start + batch_size]
            batch_records = [train_data[i] for i in batch_indices]

            try:
                batch = collator(batch_records)
            except Exception as collate_exc:  # noqa: BLE001
                print(f"[WARN] Collation failed for batch at index {batch_start}: {collate_exc}")
                collate_failures += 1
                continue

            # Move tensors to device (CPU)
            batch = {
                k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad()
            try:
                outputs = model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            except Exception:  # noqa: BLE001
                try:
                    outputs = model(**batch, compute_loss=True)
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                except Exception as fwd_exc:  # noqa: BLE001
                    print(f"[WARN] Forward pass failed at batch {batch_start}: {fwd_exc}")
                    forward_failures += 1
                    continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            steps += 1
            total_steps += 1

        if steps == 0:
            raise RuntimeError(
                "Manual training loop executed zero optimization steps in an epoch. "
                f"collate_failures={collate_failures}, "
                f"forward_failures={forward_failures}, "
                f"records={len(train_data)}, batch_size={batch_size}."
            )

        avg_loss = epoch_loss / max(steps, 1)
        print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}")

        # Checkpoint
        ckpt_dir = output_dir / f"checkpoint-epoch{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))

    if total_steps == 0:
        raise RuntimeError("Manual training loop completed with zero optimization steps.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finetune GLiNER on a custom JSONL dataset (CPU-only)."
    )
    parser.add_argument(
        "--train", type=Path, default=Path("data/train.jsonl"),
        help="Path to training JSONL file."
    )
    parser.add_argument(
        "--valid", type=Path, default=Path("data/valid.jsonl"),
        help="Path to validation JSONL file."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("outputs/run1"),
        help="Directory to save checkpoints and final model."
    )
    parser.add_argument(
        "--base_model", default="urchade/gliner_small-v2",
        help="Base model name or path (Hugging Face hub)."
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    for p, name in [(args.train, "--train"), (args.valid, "--valid")]:
        if not p.exists():
            print(f"[ERROR] {name} path does not exist: {p}", file=sys.stderr)
            sys.exit(1)

    train(
        model_name=args.base_model,
        train_path=args.train,
        valid_path=args.valid,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
