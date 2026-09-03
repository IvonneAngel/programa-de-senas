"""Consolida los siete artefactos LOSO emparejados sin acceder a datos de signos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SIGNERS = tuple(f"S{number:02d}" for number in range(1, 8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    folds = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("s08_read") or payload.get("s09_read") or len(payload.get("folds", [])) != 1:
            raise ValueError(f"Artefacto LOSO inválido: {path}")
        folds.append(payload["folds"][0])
    if {fold["signer"] for fold in folds} != set(SIGNERS) or len(folds) != 7:
        raise ValueError("Se requieren exactamente S01-S07 una vez")
    folds = sorted(folds, key=lambda fold: fold["signer"])
    deltas = np.asarray([float(fold["delta_macro_f1"]) for fold in folds], dtype=np.float64)
    rng = np.random.default_rng(2026)
    bootstrap = rng.choice(deltas, size=(10_000, len(deltas)), replace=True).mean(axis=1)
    report = {
        "protocol": "LOSO S01-S07, seed 42, TCN from scratch; paired positions126 vs bone_vector126",
        "folds": folds,
        "delta_macro_f1": {
            "mean": float(deltas.mean()),
            "per_signer": {fold["signer"]: float(fold["delta_macro_f1"]) for fold in folds},
            "bootstrap_seed": 2026,
            "bootstrap_samples": 10_000,
            "ci95": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))],
        },
        "all_deltas_positive": bool(np.all(deltas > 0.0)),
        "s08_read": False,
        "s09_read": False,
        "test_evaluated": False,
        "model_selection_changed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()