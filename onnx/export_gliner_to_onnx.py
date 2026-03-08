"""
Export a finetuned GLiNER model to ONNX.

Usage:
    python onnx/export_gliner_to_onnx.py \
        --model_dir outputs/run2/final \
        --output_dir onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gliner import GLiNER

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labels import LABELS


class _SpanExportWrapper(torch.nn.Module):
    def __init__(self, gliner_model: GLiNER, threshold: float) -> None:
        super().__init__()
        self.model = gliner_model.model
        self.threshold = threshold

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        words_mask: torch.Tensor,
        text_lengths: torch.Tensor,
        span_idx: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            words_mask=words_mask,
            text_lengths=text_lengths,
            span_idx=span_idx,
            span_mask=span_mask,
            threshold=self.threshold,
        )
        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output[0]
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        return logits


class _TokenExportWrapper(torch.nn.Module):
    def __init__(self, gliner_model: GLiNER, threshold: float) -> None:
        super().__init__()
        self.model = gliner_model.model
        self.threshold = threshold

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        words_mask: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            words_mask=words_mask,
            text_lengths=text_lengths,
            threshold=self.threshold,
        )
        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output[0]
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        return logits


def _load_labels(model_dir: Path) -> list[str]:
    run_config = model_dir / "run_config.json"
    if run_config.exists():
        try:
            cfg = json.loads(run_config.read_text(encoding="utf-8"))
            labels = cfg.get("labels")
            if isinstance(labels, list) and labels and all(isinstance(x, str) for x in labels):
                return labels
        except Exception:
            pass
    return LABELS


def _required_input_names(span_mode: str) -> list[str]:
    base = ["input_ids", "attention_mask", "words_mask", "text_lengths"]
    if span_mode != "token_level":
        base.extend(["span_idx", "span_mask"])
    return base


def _prepare_dummy_inputs(
    gliner_model: GLiNER,
    labels: list[str],
    sample_text: str,
    input_names: list[str],
) -> dict[str, torch.Tensor]:
    model_input, _ = gliner_model.prepare_model_inputs([sample_text], labels, prepare_entities=False)

    dummy: dict[str, torch.Tensor] = {}
    for name in input_names:
        value = model_input.get(name)
        if value is None or not isinstance(value, torch.Tensor):
            raise RuntimeError(f"Required input '{name}' is missing or not a tensor.")
        dummy[name] = value.detach().cpu()
    return dummy


def _dynamic_axes(input_names: list[str]) -> dict[str, dict[int, str]]:
    axes: dict[str, dict[int, str]] = {}
    for name in input_names:
        if name in {"input_ids", "attention_mask", "words_mask", "token_type_ids"}:
            axes[name] = {0: "batch", 1: "seq_len"}
        elif name == "text_lengths":
            axes[name] = {0: "batch"}
        elif name == "span_idx":
            axes[name] = {0: "batch", 1: "num_spans"}
        elif name == "span_mask":
            axes[name] = {0: "batch", 1: "num_spans"}
    axes["logits"] = {0: "batch", 1: "num_words"}
    return axes


def _copy_assets(model_dir: Path, output_dir: Path) -> list[str]:
    asset_files = [
        "gliner_config.json",
        "run_config.json",
        "spm.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]

    copied: list[str] = []
    for name in asset_files:
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            copied.append(name)
    return copied


def _verify_export(
    wrapper: torch.nn.Module,
    input_names: list[str],
    dummy_inputs: dict[str, torch.Tensor],
    onnx_path: Path,
) -> dict[str, Any]:
    import onnxruntime as ort

    wrapper.eval()
    with torch.no_grad():
        args = tuple(dummy_inputs[name] for name in input_names)
        torch_logits = wrapper(*args)
        if isinstance(torch_logits, torch.Tensor):
            torch_logits_np = torch_logits.cpu().numpy()
        else:
            torch_logits_np = np.asarray(torch_logits)

    ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {name: dummy_inputs[name].cpu().numpy() for name in input_names}
    ort_logits = ort_sess.run(["logits"], ort_inputs)[0]

    diff = np.abs(torch_logits_np - ort_logits)
    return {
        "torch_shape": list(torch_logits_np.shape),
        "onnx_shape": list(ort_logits.shape),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
    }


def export_onnx(
    model_dir: Path,
    output_dir: Path,
    onnx_file: str,
    sample_text: str,
    opset: int,
    threshold: float,
    verify: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / onnx_file

    labels = _load_labels(model_dir)
    model = GLiNER.from_pretrained(str(model_dir))
    model.eval()
    model.model.eval()
    model.model.to(torch.device("cpu"))

    span_mode = str(model.config.span_mode)
    input_names = _required_input_names(span_mode)
    dummy_inputs = _prepare_dummy_inputs(model, labels, sample_text, input_names)

    if span_mode == "token_level":
        wrapper: torch.nn.Module = _TokenExportWrapper(model, threshold=threshold)
    else:
        wrapper = _SpanExportWrapper(model, threshold=threshold)
    wrapper.eval()

    args = tuple(dummy_inputs[name] for name in input_names)
    export_kwargs = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": input_names,
        "output_names": ["logits"],
        "dynamic_axes": _dynamic_axes(input_names),
    }
    try:
        torch.onnx.export(
            wrapper,
            args,
            str(onnx_path),
            dynamo=False,
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(
            wrapper,
            args,
            str(onnx_path),
            **export_kwargs,
        )

    copied_assets = _copy_assets(model_dir, output_dir)
    (output_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    verification: dict[str, Any] | None = None
    if verify:
        verification = _verify_export(wrapper, input_names, dummy_inputs, onnx_path)

    manifest = {
        "model_dir": str(model_dir),
        "onnx_file": onnx_file,
        "span_mode": span_mode,
        "input_names": input_names,
        "output_names": ["logits"],
        "labels_count": len(labels),
        "labels_file": "labels.json",
        "copied_assets": copied_assets,
        "sample_text": sample_text,
        "opset": opset,
        "threshold": threshold,
        "verification": verification,
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] ONNX exported: {onnx_path}")
    print(f"[OK] Labels saved: {output_dir / 'labels.json'}")
    if verification is not None:
        print(
            "[OK] Verification "
            f"max_abs_diff={verification['max_abs_diff']:.6f}, "
            f"mean_abs_diff={verification['mean_abs_diff']:.6f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export GLiNER model to ONNX.")
    parser.add_argument("--model_dir", type=Path, default=Path("outputs/run2/final"))
    parser.add_argument("--output_dir", type=Path, default=Path("onnx"))
    parser.add_argument("--onnx_file", default="model.onnx")
    parser.add_argument("--sample_text", default="삼성전자는 2026년 3월 8일 서울에서 신제품을 공개했다.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip_verify", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model_dir.exists():
        raise FileNotFoundError(f"model_dir does not exist: {args.model_dir}")

    export_onnx(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        onnx_file=args.onnx_file,
        sample_text=args.sample_text,
        opset=args.opset,
        threshold=args.threshold,
        verify=not args.skip_verify,
    )


if __name__ == "__main__":
    main()
