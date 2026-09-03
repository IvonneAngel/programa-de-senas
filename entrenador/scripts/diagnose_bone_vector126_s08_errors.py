"""Diagnóstico descriptivo de predicciones S08 congeladas de bone_vector126."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def motion_metrics(sequence: np.ndarray) -> dict[str, float]:
    values = np.asarray(sequence, dtype=np.float32)
    if values.shape != (30, 126) or not np.isfinite(values).all():
        raise ValueError(f"Secuencia inválida: {values.shape}")
    hands = values.reshape(30, 2, 21, 3)
    present = np.linalg.norm(hands, axis=(2, 3)) > 1e-6
    transitions = present[1:] & present[:-1]
    delta = np.linalg.norm(hands[1:] - hands[:-1], axis=(2, 3))
    return {
        "hand_coverage": float(present.mean()),
        "bimanual_fraction": float((present.sum(axis=1) == 2).mean()),
        "motion_mean": float(delta[transitions].mean()) if transitions.any() else 0.0,
        "motion_p95": float(np.percentile(delta[transitions], 95)) if transitions.any() else 0.0,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        return {"count": 0}
    result: dict[str, float | int] = {"count": len(rows)}
    for field in ("confidence", "margin", "hand_coverage", "bimanual_fraction", "motion_mean", "motion_p95"):
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        result[f"{field}_median"] = float(np.median(values))
        result[f"{field}_mean"] = float(values.mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.predictions.open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    if len(predictions) != 210 or any(row["signer_id"] != "S08" for row in predictions):
        raise ValueError("El diagnóstico exige exactamente 210 predicciones S08 congeladas")
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        manifest = {row["sample_id"]: row for row in csv.DictReader(handle)}
    records: list[dict[str, object]] = []
    for prediction in predictions:
        row = manifest.get(prediction["sample_id"])
        if row is None or row["split_model"] != "validation" or row["signer_id"] != "S08" or row["feature_status"] != "ok":
            raise ValueError(f"Fila S08 inválida: {prediction['sample_id']}")
        metrics = motion_metrics(np.load(args.cache_root / row["feature_path"], allow_pickle=False))
        records.append({
            "sample_id": prediction["sample_id"], "label": prediction["label_lsm"], "prediction": prediction["prediction_lsm"],
            "correct": prediction["correct"] == "1", "confidence": float(prediction["confidence"]), "margin": float(prediction["margin"]), **metrics,
        })
    correct = [row for row in records if bool(row["correct"])]
    incorrect = [row for row in records if not bool(row["correct"])]
    confusions = Counter((str(row["label"]), str(row["prediction"])) for row in incorrect)
    report = {
        "kind": "bone_vector126_s08_error_diagnosis_descriptive",
        "prediction_source": str(args.predictions),
        "s08_read": True,
        "s09_read": False,
        "inference_executed": False,
        "samples": len(records),
        "correct_summary": summarize(correct),
        "incorrect_summary": summarize(incorrect),
        "top_confusions": [{"label": label, "prediction": predicted, "count": count} for (label, predicted), count in confusions.most_common(20)],
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("kind", "samples", "correct_summary", "incorrect_summary", "s08_read", "s09_read", "inference_executed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()