"""Audita bone_vector126 frente a positions126 sin entrenar ni seleccionar modelos."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def hand_presence(sequence: np.ndarray) -> np.ndarray:
    return np.abs(sequence.reshape(30, 2, 63)).sum(axis=2) > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.source_manifest.open(encoding="utf-8", newline="") as handle:
        source_rows = {row["sample_id"]: row for row in csv.DictReader(handle) if row["feature_status"] == "ok"}
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1890:
        raise ValueError(f"Se requieren 1,890 filas, no {len(rows)}")
    count_by_split = Counter()
    presence_mismatches: list[str] = []
    checked = 0
    excluded = 0
    for row in rows:
        count_by_split[row["split_project"]] += 1
        if row["feature_status"] != "ok":
            excluded += 1
            continue
        source_row = source_rows[row["sample_id"]]
        source = np.load(args.source_cache_root / source_row["feature_path"], allow_pickle=False)
        derived = np.load(args.cache_root / row["feature_path"], allow_pickle=False)
        if source.shape != (30, 126) or derived.shape != (30, 126) or not np.isfinite(derived).all():
            raise ValueError(f"Tensor inválido: {row['sample_id']}")
        if not np.array_equal(hand_presence(source), hand_presence(derived)):
            presence_mismatches.append(row["sample_id"])
        checked += 1
    report = {"rows": len(rows), "checked": checked, "excluded": excluded, "by_split": dict(count_by_split), "presence_mismatches": presence_mismatches, "presence_preserved": not presence_mismatches, "s09_predictions": False, "metrics_evaluated": False}
    if presence_mismatches:
        raise AssertionError(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()