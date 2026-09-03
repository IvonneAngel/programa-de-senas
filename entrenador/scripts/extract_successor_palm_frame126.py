"""Deriva palm_frame126 de positions126 sin datos de video, etiquetas ni métricas."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from palm_frame126 import transform_sequence


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
        raise ValueError(f"Se requieren 1,890 posiciones126 recuperadas, no {len(accepted)}")
    derived: list[dict[str, str]] = []
    for row in accepted:
        source = args.cache_root / row["feature_path"]
        sequence = np.load(source, allow_pickle=False)
        target_relative = Path("palm_frame126") / f"{row['sample_id']}.npy"
        target = args.output_cache_root / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, transform_sequence(sequence), allow_pickle=False)
        updated = dict(row)
        updated["feature_path"] = str(target_relative)
        updated["extractor"] = "palm_frame126_mcp9_mcp5_axisangle_from_positions126"
        updated["feature_error"] = ""
        derived.append(updated)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(derived[0].keys()))
        writer.writeheader()
        writer.writerows(derived)
    print(json.dumps({"rows": len(derived), "shape": [30, 126], "metrics_evaluated": False, "output_manifest": str(args.output_manifest)}))


if __name__ == "__main__":
    main()