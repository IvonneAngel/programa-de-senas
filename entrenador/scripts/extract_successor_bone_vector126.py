"""Deriva bone_vector126 desde positions126 sin vídeos ni evaluación."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bone_vector126 import transform_sequence


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
        raise ValueError(f"Se requieren 1,890 positions126, no {len(accepted)}")
    derived: list[dict[str, str]] = []
    for row in accepted:
        source = args.cache_root / row["feature_path"]
        sequence = np.load(source, allow_pickle=False)
        transformed, degeneracies = transform_sequence(sequence)
        updated = dict(row)
        updated["extractor"] = "bone_vector126_20_directed_bones_mean_mcp_scale_from_positions126"
        updated["bone_degenerate_hands"] = str(degeneracies)
        if degeneracies:
            updated["feature_status"] = "excluded"
            updated["feature_error"] = f"degenerate_palm_scale_hands={degeneracies}"
            updated["feature_path"] = ""
        else:
            target_relative = Path("bone_vector126") / f"{row['sample_id']}.npy"
            target = args.output_cache_root / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, transformed, allow_pickle=False)
            updated["feature_status"] = "ok"
            updated["feature_error"] = ""
            updated["feature_path"] = str(target_relative)
        derived.append(updated)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*derived[0].keys()])
        writer.writeheader()
        writer.writerows(derived)
    print(json.dumps({"rows": len(derived), "ok": sum(row["feature_status"] == "ok" for row in derived), "excluded": sum(row["feature_status"] == "excluded" for row in derived), "metrics_evaluated": False}))


if __name__ == "__main__":
    main()