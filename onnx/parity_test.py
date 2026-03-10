from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from gliner import GLiNER


@dataclass(frozen=True)
class EntityKey:
    start: int
    end: int
    label: str
    text: str


def load_labels_file(labels_path: Path) -> list[str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"labels file not found: {labels_path}")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError(f"labels file has invalid format: {labels_path}")
    if not labels:
        raise ValueError(f"labels file is empty: {labels_path}")
    if any(not x for x in labels):
        raise ValueError(f"labels file contains empty label: {labels_path}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"labels file contains duplicate labels: {labels_path}")
    return labels


def normalize_entities(items: Iterable[dict]) -> dict[EntityKey, float]:
    norm: dict[EntityKey, float] = {}
    for item in items:
        key = EntityKey(
            start=int(item["start"]),
            end=int(item["end"]),
            label=str(item["label"]),
            text=str(item["text"]),
        )
        norm[key] = float(item["score"])
    return norm


def discover_cpp_binary(cpp_dir: Path) -> Path | None:
    candidates = [
        cpp_dir / "build" / "gliner_tizen_demo.exe",
        cpp_dir / "build" / "Release" / "gliner_tizen_demo.exe",
        cpp_dir / "build-vs" / "Release" / "gliner_tizen_demo.exe",
        cpp_dir / "build-vs" / "gliner_tizen_demo.exe",
        cpp_dir / "build" / "gliner_tizen_demo",
        cpp_dir / "build" / "Release" / "gliner_tizen_demo",
        cpp_dir / "build-vs" / "Release" / "gliner_tizen_demo",
        cpp_dir / "build-vs" / "gliner_tizen_demo",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_cpp_inference(
    cpp_bin: Path,
    model_dir: Path,
    text: str,
    threshold: float,
    flat_ner: bool,
    multi_label: bool,
    labels_file: Path | None = None,
) -> list[dict]:
    cmd = [
        str(cpp_bin),
        str(model_dir),
        text,
        str(threshold),
        "1" if flat_ner else "0",
        "1" if multi_label else "0",
    ]
    if labels_file is not None:
        cmd.append(str(labels_file))

    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "C++ inference failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"exit_code: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    out = proc.stdout.strip()
    if not out:
        return []

    try:
        parsed = json.loads(out)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Failed to parse C++ JSON output\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        ) from exc

    if not isinstance(parsed, list):
        raise RuntimeError(f"Unexpected C++ output format. Expected list, got: {type(parsed).__name__}")
    return parsed


def run_python_inference(
    model: GLiNER,
    labels: list[str],
    text: str,
    threshold: float,
    flat_ner: bool,
    multi_label: bool,
) -> list[dict]:
    result = model.predict_entities(
        text,
        labels,
        threshold=threshold,
        flat_ner=flat_ner,
        multi_label=multi_label,
    )
    if not isinstance(result, list):
        raise RuntimeError(f"Unexpected Python output format. Expected list, got: {type(result).__name__}")
    return result


def diff_entities(py_entities: list[dict], cpp_entities: list[dict], score_tol: float) -> tuple[bool, str]:
    py_map = normalize_entities(py_entities)
    cpp_map = normalize_entities(cpp_entities)

    py_keys = set(py_map.keys())
    cpp_keys = set(cpp_map.keys())

    missing_in_cpp = sorted(py_keys - cpp_keys, key=lambda x: (x.start, x.end, x.label, x.text))
    missing_in_py = sorted(cpp_keys - py_keys, key=lambda x: (x.start, x.end, x.label, x.text))

    score_mismatch: list[str] = []
    for key in sorted(py_keys & cpp_keys, key=lambda x: (x.start, x.end, x.label, x.text)):
        delta = abs(py_map[key] - cpp_map[key])
        if delta > score_tol:
            score_mismatch.append(
                f"  - {key} | py={py_map[key]:.8f}, cpp={cpp_map[key]:.8f}, abs_diff={delta:.8f}"
            )

    if not missing_in_cpp and not missing_in_py and not score_mismatch:
        return True, ""

    lines: list[str] = ["Parity mismatch detected."]

    if missing_in_cpp:
        lines.append(f"- Missing in C++ ({len(missing_in_cpp)}):")
        lines.extend([f"  - {k}" for k in missing_in_cpp[:20]])
        if len(missing_in_cpp) > 20:
            lines.append(f"  ... and {len(missing_in_cpp) - 20} more")

    if missing_in_py:
        lines.append(f"- Missing in Python ({len(missing_in_py)}):")
        lines.extend([f"  - {k}" for k in missing_in_py[:20]])
        if len(missing_in_py) > 20:
            lines.append(f"  ... and {len(missing_in_py) - 20} more")

    if score_mismatch:
        lines.append(f"- Score mismatches (>{score_tol}):")
        lines.extend(score_mismatch[:20])
        if len(score_mismatch) > 20:
            lines.append(f"  ... and {len(score_mismatch) - 20} more")

    return False, "\n".join(lines)


def collect_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []

    if args.text:
        texts.extend(args.text)

    if args.text_file:
        for line in Path(args.text_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                texts.append(line)

    if not texts:
        texts = ["삼성전자는 2026년 3월 8일 서울에서 신제품을 공개했다."]

    return texts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare GLiNER Python vs C++ inference parity.")
    parser.add_argument("--model_dir", type=Path, default=Path("onnx"))
    parser.add_argument("--onnx_file", default="model.onnx")
    parser.add_argument("--labels_file", type=Path, default=None, help="Labels JSON array file. Defaults to <model_dir>/labels.json")
    parser.add_argument("--cpp_bin", type=Path, default=None)
    parser.add_argument("--cpp_dir", type=Path, default=Path("onnx/cpp"))
    parser.add_argument("--text", action="append", default=None, help="Input text. Can be passed multiple times.")
    parser.add_argument("--text_file", type=Path, default=None, help="UTF-8 file with one text per line.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--flat_ner", action="store_true", default=True)
    parser.add_argument("--nested_ner", action="store_true", help="If set, use nested NER mode (flat_ner=False).")
    parser.add_argument("--multi_label", action="store_true", default=False)
    parser.add_argument("--score_tol", type=float, default=1e-4)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    model_dir = args.model_dir.resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")

    labels_path = (args.labels_file.resolve() if args.labels_file is not None else (model_dir / "labels.json"))
    labels = load_labels_file(labels_path)

    cpp_bin = args.cpp_bin
    if cpp_bin is None:
        cpp_bin = discover_cpp_binary(args.cpp_dir.resolve())
    if cpp_bin is None or not cpp_bin.exists():
        raise FileNotFoundError(
            "C++ binary not found. Build first, or provide --cpp_bin.\n"
            "Expected one of: onnx/cpp/build*/(Release/)gliner_tizen_demo(.exe)"
        )

    flat_ner = not args.nested_ner
    texts = collect_texts(args)

    print(f"[INFO] model_dir={model_dir}")
    print(f"[INFO] cpp_bin={cpp_bin}")
    print(f"[INFO] labels_file={labels_path} (count={len(labels)})")
    print(
        "[INFO] options="
        f"threshold={args.threshold}, flat_ner={flat_ner}, multi_label={args.multi_label}, score_tol={args.score_tol}"
    )
    print(f"[INFO] texts={len(texts)}")

    py_model = GLiNER.from_pretrained(
        str(model_dir),
        load_onnx_model=True,
        onnx_model_file=args.onnx_file,
    )

    failed = 0

    for idx, text in enumerate(texts, start=1):
        py_entities = run_python_inference(
            py_model,
            labels,
            text,
            threshold=args.threshold,
            flat_ner=flat_ner,
            multi_label=args.multi_label,
        )
        cpp_entities = run_cpp_inference(
            cpp_bin,
            model_dir,
            text,
            threshold=args.threshold,
            flat_ner=flat_ner,
            multi_label=args.multi_label,
            labels_file=labels_path,
        )

        ok, reason = diff_entities(py_entities, cpp_entities, score_tol=args.score_tol)

        print(f"\n[TEXT {idx}] {text}")
        if ok:
            print(f"[OK] parity matched ({len(py_entities)} entities)")
            if args.verbose:
                print("[Python]", json.dumps(py_entities, ensure_ascii=False, indent=2))
                print("[C++]", json.dumps(cpp_entities, ensure_ascii=False, indent=2))
        else:
            failed += 1
            print("[FAIL] parity mismatch")
            print(reason)
            if args.verbose:
                print("[Python]", json.dumps(py_entities, ensure_ascii=False, indent=2))
                print("[C++]", json.dumps(cpp_entities, ensure_ascii=False, indent=2))

    if failed:
        print(f"\n[RESULT] FAILED: {failed}/{len(texts)} texts mismatched")
        return 1

    print(f"\n[RESULT] PASS: {len(texts)}/{len(texts)} texts matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
