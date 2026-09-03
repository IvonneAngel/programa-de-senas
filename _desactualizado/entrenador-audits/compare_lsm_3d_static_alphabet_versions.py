"""Verifica si la versión normalizada coincide con la transformación declarada de v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records: list[dict[str, float | str]] = []
    for normalized_path in sorted(args.normalized_root.glob("*/*.txt")):
        original_path = args.original_root / normalized_path.relative_to(args.normalized_root)
        if not original_path.exists():
            raise FileNotFoundError(original_path)
        original = np.loadtxt(original_path, dtype=np.float64)
        normalized = np.loadtxt(normalized_path, dtype=np.float64)
        if original.shape != normalized.shape or original.ndim != 2 or original.shape[1] != 3:
            raise ValueError(f"Forma incompatible: {normalized_path}")
        centered = original - original.mean(axis=0, keepdims=True)
        radius = float(np.linalg.norm(centered, axis=1).max())
        expected = centered / radius
        absolute_error = np.abs(expected - normalized)
        records.append({
            "sample_id": normalized_path.stem,
            "expected_radius": radius,
            "max_abs_error": float(absolute_error.max()),
            "mean_abs_error": float(absolute_error.mean()),
            "expected_centroid_norm": float(np.linalg.norm(expected.mean(axis=0))),
            "observed_centroid_norm": float(np.linalg.norm(normalized.mean(axis=0))),
            "observed_max_radius": float(np.linalg.norm(normalized, axis=1).max()),
        })
    if len(records) != 315:
        raise AssertionError(f"Se esperaban 315 nubes, no {len(records)}")
    max_error = max(float(record["max_abs_error"]) for record in records)
    report = {
        "samples": len(records),
        "max_abs_error": max_error,
        "p95_max_abs_error": float(np.percentile([record["max_abs_error"] for record in records], 95)),
        "exact_or_near_exact_under_1e-6": sum(float(record["max_abs_error"]) <= 1e-6 for record in records),
        "mismatches_over_1e-4": [record for record in records if float(record["max_abs_error"]) > 1e-4],
        "data_modified": False,
        "trained_or_evaluated": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()