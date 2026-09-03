"""Deriva bone_angular166 desde bone_vector126 sin evaluación ni extracción de vídeo."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bone_angular166 import transform_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-cache-root", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    accepted = [row for row in rows if row["task"] == "successor_positions126" and row["feature_status"] == "ok"]
    if len(accepted) != 1890:
        raise ValueError(f"Se requieren 1,890 bone_vector126, no {len(accepted)}")
    derived: list[dict[str, str]] = []
    for row in accepted:
        transformed = transform_sequence(np.load(args.cache_root / row["feature_path"], allow_pickle=False))
        updated = dict(row)
        target_relative = Path("bone_angular166") / f"{row['sample_id']}.npy"
        target = args.output_cache_root / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, transformed, allow_pickle=False)
        updated["extractor"] = "bone_vector126_plus_20_interframe_bone_angles_per_hand"
        updated["feature_status"] = "ok"
        updated["feature_error"] = ""
        updated["feature_path"] = str(target_relative)
        derived.append(updated)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*derived[0].keys()])
        writer.writeheader()
        writer.writerows(derived)
    print(json.dumps({"rows": len(derived), "ok": len(derived), "shape": [30, 166], "metrics_evaluated": False, "s09_evaluated": False}))


if __name__ == "__main__":
    main()