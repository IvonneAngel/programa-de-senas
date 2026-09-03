"""Deriva bone_tetra136 desde caches recuperadas sin evaluación de modelos."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bone_tetra136 import transform_sequence


def accepted_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = [row for row in rows if row["task"] == "successor_positions126" and row["feature_status"] == "ok"]
    if len(result) != 1890:
        raise ValueError(f"{path}: se requieren 1,890 filas, no {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bone-manifest", type=Path, required=True)
    parser.add_argument("--bone-cache-root", type=Path, required=True)
    parser.add_argument("--positions-manifest", type=Path, required=True)
    parser.add_argument("--positions-cache-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-cache-root", type=Path, required=True)
    args = parser.parse_args()
    bone_rows = accepted_rows(args.bone_manifest)
    position_rows = {row["sample_id"]: row for row in accepted_rows(args.positions_manifest)}
    derived: list[dict[str, str]] = []
    for bone in bone_rows:
        position = position_rows.get(bone["sample_id"])
        if position is None or any(bone[key] != position[key] for key in ("label_lsm", "signer_id", "split_model")):
            raise ValueError(f"{bone['sample_id']}: par bone/positions inconsistente")
        transformed = transform_sequence(
            np.load(args.bone_cache_root / bone["feature_path"], allow_pickle=False),
            np.load(args.positions_cache_root / position["feature_path"], allow_pickle=False),
        )
        updated = dict(bone)
        target_relative = Path("bone_tetra136") / f"{bone['sample_id']}.npy"
        target = args.output_cache_root / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, transformed, allow_pickle=False)
        updated["extractor"] = "bone_vector126_plus_5_signed_mcp_pip_tip_tetra_volumes_per_hand"
        updated["feature_path"] = str(target_relative)
        updated["feature_status"] = "ok"
        updated["feature_error"] = ""
        derived.append(updated)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*derived[0].keys()])
        writer.writeheader()
        writer.writerows(derived)
    print(json.dumps({"rows": len(derived), "ok": len(derived), "shape": [30, 136], "metrics_evaluated": False, "s09_evaluated": False}))


if __name__ == "__main__":
    main()