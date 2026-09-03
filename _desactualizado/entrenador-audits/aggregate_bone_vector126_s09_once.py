"""Agrega artefactos S09 existentes sin volver a ejecutar inferencia."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if {row.get("seed") for row in rows} != {13, 21, 42} or len(rows) != 3:
        raise ValueError("Se requieren exactamente los artefactos s13/s21/s42")
    if any(not row.get("s09_evaluated") or row.get("training_performed") or row.get("selection_performed") or row.get("retry_allowed") for row in rows):
        raise ValueError("Artefacto S09 no conforme")
    if any(row.get("signers") != ["S09"] or row.get("samples") != 210 or row.get("classes") != 210 for row in rows):
        raise ValueError("Población S09 no conforme")
    macro = np.asarray([float(row["macro_f1"]) for row in rows], dtype=np.float64)
    accuracy = np.asarray([float(row["accuracy"]) for row in rows], dtype=np.float64)
    report = {
        "kind": "aggregate_of_single_authorized_s09_evaluations",
        "per_seed": {str(row["seed"]): {"macro_f1": row["macro_f1"], "accuracy": row["accuracy"], "checkpoint": row["checkpoint"]} for row in sorted(rows, key=lambda item: item["seed"])},
        "macro_f1": {"mean": float(macro.mean()), "median": float(np.median(macro)), "min": float(macro.min()), "max": float(macro.max())},
        "accuracy": {"mean": float(accuracy.mean()), "median": float(np.median(accuracy)), "min": float(accuracy.min()), "max": float(accuracy.max())},
        "s09_signer_clusters": 1,
        "test_bootstrap_by_signer_meaningful": False,
        "test_control_evaluated": False,
        "test_delta_vs_control_available": False,
        "reason": "S09 se reservó para los tres checkpoints candidatos; no se evaluó control post hoc y S09 tiene un solo firmante.",
        "further_s09_evaluation_allowed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()