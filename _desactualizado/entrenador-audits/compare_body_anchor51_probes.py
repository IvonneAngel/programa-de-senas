"""Compara tres salidas de body_anchor51 y emite un reporte sin datos de imagen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float32)
    if values.shape != (30, 51) or not np.isfinite(values).all():
        raise ValueError(f"Salida inválida {path}: {values.shape}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", dest="python_path", type=Path, required=True)
    parser.add_argument("--typescript", dest="typescript_path", type=Path, required=True)
    parser.add_argument("--rust", dest="rust_path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    outputs = {"python": load(args.python_path), "typescript": load(args.typescript_path), "rust": load(args.rust_path)}
    comparisons = {}
    for left, right in (("python", "typescript"), ("python", "rust"), ("typescript", "rust")):
        delta = np.abs(outputs[left] - outputs[right])
        comparisons[f"{left}_vs_{right}"] = {"max_abs": float(delta.max()), "cells_over_1e-5": int((delta > 1e-5).sum())}
    report = {"shape": [30, 51], "image_data": False, "label_data": False, "comparisons": comparisons}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if any(item["cells_over_1e-5"] for item in comparisons.values()):
        raise SystemExit("La paridad excede la tolerancia 1e-5")


if __name__ == "__main__":
    main()