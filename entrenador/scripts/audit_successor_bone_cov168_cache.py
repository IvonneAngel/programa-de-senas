"""Audita artefactos bone_cov168 sin cargar un clasificador ni evaluar métricas."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--bone-manifest", type=Path, required=True)
    parser.add_argument("--bone-cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with args.bone_manifest.open(encoding="utf-8", newline="") as handle:
        bones = {row["sample_id"]: row for row in csv.DictReader(handle)}
    accepted = [row for row in rows if row["task"] == "successor_positions126" and row["feature_status"] == "ok"]
    if len(accepted) != 1890 or len({row["label_lsm"] for row in accepted}) != 210:
        raise ValueError("La caché cov exige 1,890 clips y 210 clases")
    splits = {split: sum(row["split_model"] == split for row in accepted) for split in ("train", "validation", "test")}
    if splits != {"train": 1470, "validation": 210, "test": 210}:
        raise ValueError(f"Split inválido: {splits}")
    sparse = 0
    for row in accepted:
        value = np.load(args.cache_root / row["feature_path"], allow_pickle=False)
        bone_row = bones.get(row["sample_id"])
        if bone_row is None:
            raise ValueError(f"Sin par bone: {row['sample_id']}")
        bone = np.load(args.bone_cache_root / bone_row["feature_path"], allow_pickle=False)
        if value.shape != (30, 168) or not np.isfinite(value).all() or not np.array_equal(value[:, :126], bone.astype(np.float32)):
            raise ValueError(f"Contrato cov inválido: {row['sample_id']}")
        observed = tuple(int(value) for value in row["covariance_hand_observations"].split(","))
        sparse += int(min(observed) < 2)
    report = {"kind": "successor_bone_cov168_cache_audit", "samples": len(accepted), "classes": 210, "splits": splits, "shape": [30, 168], "sparse_hand_clips": sparse, "metrics_evaluated": False, "s08_metrics_evaluated": False, "s09_metrics_evaluated": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()