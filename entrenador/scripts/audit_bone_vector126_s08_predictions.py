"""Audita CSV de predicciones S08 de múltiples semillas sin acceder a S09."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def macro_f1(true: list[str], predicted: list[str], labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(actual == label and guessed == label for actual, guessed in zip(true, predicted))
        fp = sum(actual != label and guessed == label for actual, guessed in zip(true, predicted))
        fn = sum(actual == label and guessed != label for actual, guessed in zip(true, predicted))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else (2.0 * tp) / denominator)
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    by_seed: dict[int, list[dict[str, str]]] = {}
    for path in args.predictions:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 210:
            raise ValueError(f"{path}: se requieren 210 predicciones S08")
        seed = int(rows[0]["seed"])
        if seed in by_seed or any(int(row["seed"]) != seed for row in rows):
            raise ValueError(f"Semilla inválida o repetida: {path}")
        if len({row["label_lsm"] for row in rows}) != 210:
            raise ValueError(f"{path}: cobertura de clase S08 inválida")
        by_seed[seed] = rows
    reports = {}
    for seed, rows in sorted(by_seed.items()):
        labels = sorted(row["label_lsm"] for row in rows)
        true = [row["label_lsm"] for row in rows]
        predicted = [row["prediction_lsm"] for row in rows]
        errors = Counter((actual, guessed) for actual, guessed in zip(true, predicted) if actual != guessed)
        reports[str(seed)] = {
            "rows": len(rows),
            "macro_f1": macro_f1(true, predicted, labels),
            "accuracy": float(np.mean([actual == guessed for actual, guessed in zip(true, predicted)])),
            "classes": len(labels),
            "signers": sorted({row["signer_id"] for row in rows}),
            "confidence": {"mean": float(np.mean([float(row["confidence"]) for row in rows])), "median": float(np.median([float(row["confidence"]) for row in rows]))},
            "margin": {"mean": float(np.mean([float(row["margin"]) for row in rows])), "median": float(np.median([float(row["margin"]) for row in rows]))},
            "top_confusions": [{"true": actual, "predicted": guessed, "count": count} for (actual, guessed), count in errors.most_common(12)],
        }
    signer_sets = {tuple(report["signers"]) for report in reports.values()}
    report = {
        "seeds": reports,
        "s09_read": False,
        "all_predictions_are_s08_only": True,
        "signer_clusters_available_for_bootstrap": len(next(iter(signer_sets))) if len(signer_sets) == 1 else 0,
        "bootstrap_by_signer_meaningful": False,
        "bootstrap_reason": "S08 contiene un solo firmante; un bootstrap agrupado no estima variación interfirmante.",
        "model_selection_changed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()