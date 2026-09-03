"""Auditoría de contrato para la caché palm_frame126; no calcula ninguna métrica de modelos."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--derived-manifest", type=Path, required=True)
    parser.add_argument("--derived-cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.source_manifest.open(encoding="utf-8", newline="") as handle:
        source_rows = {row["sample_id"]: row for row in csv.DictReader(handle) if row["task"] == "successor_positions126" and row["feature_status"] == "ok"}
    with args.derived_manifest.open(encoding="utf-8", newline="") as handle:
        derived_rows = list(csv.DictReader(handle))
    if len(source_rows) != 1890 or len(derived_rows) != 1890:
        raise ValueError(f"Cobertura inválida: source={len(source_rows)}, derived={len(derived_rows)}")
    absent_mismatch = 0
    nonfinite = 0
    shape_errors = 0
    per_split = Counter()
    for row in derived_rows:
        source = source_rows.get(row["sample_id"])
        if source is None:
            raise ValueError(f"Muestra no reconocida: {row['sample_id']}")
        original = np.load(args.source_cache_root / source["feature_path"], allow_pickle=False)
        derived = np.load(args.derived_cache_root / row["feature_path"], allow_pickle=False)
        if original.shape != (30, 126) or derived.shape != (30, 126):
            shape_errors += 1
            continue
        if not np.isfinite(derived).all():
            nonfinite += 1
        original_hands = np.stack((np.any(original[:, :63] != 0.0, axis=1), np.any(original[:, 63:] != 0.0, axis=1)), axis=1)
        derived_hands = np.stack((np.any(derived[:, :63] != 0.0, axis=1), np.any(derived[:, 63:] != 0.0, axis=1)), axis=1)
        absent_mismatch += int(np.count_nonzero(original_hands != derived_hands))
        per_split[row["split_model"]] += 1
    if per_split != Counter({"train": 1470, "validation": 210, "test": 210}):
        raise ValueError(f"Split inválido: {dict(per_split)}")
    report = {
        "rows": len(derived_rows),
        "shape": [30, 126],
        "splits": dict(per_split),
        "nonfinite_tensors": nonfinite,
        "shape_errors": shape_errors,
        "hand_presence_mismatches": absent_mismatch,
        "metrics_evaluated": False,
        "s09_predictions_or_metrics": False,
    }
    if nonfinite or shape_errors or absent_mismatch:
        raise AssertionError(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()