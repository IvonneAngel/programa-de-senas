"""Deriva bone_cov168 desde bone_vector126 sin evaluar clasificadores."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bone_cov168 import transform_sequence


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
    sparse_hands = 0
    for row in accepted:
        source = args.cache_root / row["feature_path"]
        transformed, observations = transform_sequence(np.load(source, allow_pickle=False))
        target_relative = Path("bone_cov168") / f"{row['sample_id']}.npy"
        target = args.output_cache_root / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, transformed, allow_pickle=False)
        updated = dict(row)
        updated["extractor"] = "bone_cov168_chain_length_palm_covariance_upper_triangle_from_bone_vector126"
        updated["covariance_hand_observations"] = f"{observations[0]},{observations[1]}"
        updated["feature_status"] = "ok"
        updated["feature_error"] = ""
        updated["feature_path"] = str(target_relative)
        sparse_hands += int(min(observations) < 2)
        derived.append(updated)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*derived[0].keys()])
        writer.writeheader()
        writer.writerows(derived)
    print(json.dumps({"rows": len(derived), "ok": len(derived), "sparse_hand_clips": sparse_hands, "metrics_evaluated": False}))


if __name__ == "__main__":
    main()