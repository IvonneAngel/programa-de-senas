"""Deriva bone_code190 desde bone_vector126 con un codebook train-only congelado."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from bone_code190 import load_centers, transform_sequence


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--expected-codebook-sha256", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-cache-root", type=Path, required=True)
    args = parser.parse_args()
    observed_hash = sha256(args.codebook)
    if observed_hash != args.expected_codebook_sha256:
        raise ValueError("SHA-256 del codebook no coincide con el prerregistro")
    centers = load_centers(args.codebook)
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    accepted = [row for row in rows if row["task"] == "successor_positions126" and row["feature_status"] == "ok"]
    if len(accepted) != 1890:
        raise ValueError(f"Se requieren 1,890 bone_vector126, no {len(accepted)}")
    derived: list[dict[str, str]] = []
    for row in accepted:
        transformed, counts = transform_sequence(np.load(args.cache_root / row["feature_path"], allow_pickle=False), centers)
        target_relative = Path("bone_code190") / f"{row['sample_id']}.npy"
        target = args.output_cache_root / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, transformed, allow_pickle=False)
        updated = dict(row)
        updated["extractor"] = "bone_code190_train_only_kmeans32_hand_codebook_one_hot"
        updated["codebook_sha256"] = observed_hash
        updated["code_hand_observations"] = f"{counts[0]},{counts[1]}"
        updated["feature_status"] = "ok"
        updated["feature_error"] = ""
        updated["feature_path"] = str(target_relative)
        derived.append(updated)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*derived[0].keys()])
        writer.writeheader()
        writer.writerows(derived)
    print(json.dumps({"rows": len(derived), "codebook_sha256": observed_hash, "metrics_evaluated": False}))


if __name__ == "__main__":
    main()