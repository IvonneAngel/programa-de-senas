"""Audita bone_code190 sin clasificador, logits ni métricas de reconocimiento."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--bone-manifest", type=Path, required=True)
    parser.add_argument("--bone-cache-root", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--expected-codebook-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.codebook) != args.expected_codebook_sha256:
        raise ValueError("Codebook alterado después del prerregistro")
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with args.bone_manifest.open(encoding="utf-8", newline="") as handle:
        bones = {row["sample_id"]: row for row in csv.DictReader(handle)}
    accepted = [row for row in rows if row["task"] == "successor_positions126" and row["feature_status"] == "ok"]
    splits = {split: sum(row["split_model"] == split for row in accepted) for split in ("train", "validation", "test")}
    if len(accepted) != 1890 or len({row["label_lsm"] for row in accepted}) != 210 or splits != {"train": 1470, "validation": 210, "test": 210}:
        raise ValueError("Población o split code190 inválidos")
    left_absent = right_absent = 0
    for row in accepted:
        value = np.load(args.cache_root / row["feature_path"], allow_pickle=False)
        bone = np.load(args.bone_cache_root / bones[row["sample_id"]]["feature_path"], allow_pickle=False)
        if value.shape != (30, 190) or not np.isfinite(value).all() or not np.array_equal(value[:, :126], bone.astype(np.float32)):
            raise ValueError(f"Contrato inválido: {row['sample_id']}")
        for start, observation_column in ((126, 0), (158, 1)):
            codes = value[:, start:start + 32]
            observations = int(row["code_hand_observations"].split(",")[observation_column])
            sums = codes.sum(axis=1)
            if not np.all((sums == 0.0) | (sums == 1.0)):
                raise ValueError(f"Código no one-hot: {row['sample_id']}")
            if int((sums == 1.0).sum()) != observations:
                raise ValueError(f"Observación manual inconsistente: {row['sample_id']}")
            if observation_column == 0:
                left_absent += int(observations == 0)
            else:
                right_absent += int(observations == 0)
    report = {"kind": "successor_bone_code190_cache_audit", "samples": len(accepted), "classes": 210, "splits": splits, "shape": [30, 190], "codebook_sha256": args.expected_codebook_sha256, "left_absent_clips": left_absent, "right_absent_clips": right_absent, "metrics_evaluated": False, "s08_metrics_evaluated": False, "s09_metrics_evaluated": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()