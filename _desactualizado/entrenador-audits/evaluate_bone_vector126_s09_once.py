"""Evaluación cerrada de una sola vez para bone_vector126 tras superar la compuerta escrita."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from lsm.models.tcn import TemporalTCN


ALLOWED_SEEDS = {13, 21, 42}


def macro_f1(targets: np.ndarray, predictions: np.ndarray, classes: int) -> float:
    scores = []
    for label in range(classes):
        tp = int(np.sum((targets == label) & (predictions == label)))
        fp = int(np.sum((targets != label) & (predictions == label)))
        fn = int(np.sum((targets == label) & (predictions != label)))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=sorted(ALLOWED_SEEDS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if "s09_authorized=true" not in args.gate.read_text(encoding="utf-8"):
        raise PermissionError("La compuerta S09 no está autorizada")
    if args.out.exists():
        raise FileExistsError(f"Se rechaza sobrescritura de evaluación cerrada: {args.out}")
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    test_rows = [row for row in rows if row["split_model"] == "test" and row["feature_status"] == "ok"]
    if len(test_rows) != 210 or len({row["label_lsm"] for row in test_rows}) != 210:
        raise ValueError("S09 debe tener exactamente 210 clips y etiquetas")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    labels: dict[str, int] = state["labels"]
    if len(labels) != 210 or set(labels) != {row["label_lsm"] for row in test_rows}:
        raise ValueError("Checkpoint incompatible con población S09")
    model = TemporalTCN(feature_dim=126, classes=210, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    true: list[int] = []
    predicted: list[int] = []
    per_clip: list[dict[str, str | int | float]] = []
    inverse = {index: label for label, index in labels.items()}
    with torch.no_grad():
        for row in test_rows:
            values = np.load(args.cache_root / row["feature_path"], allow_pickle=False)
            if values.shape != (30, 126) or not np.isfinite(values).all():
                raise ValueError(f"S09 inválida: {row['sample_id']}")
            probabilities = torch.softmax(model(torch.from_numpy(values.astype(np.float32, copy=False)).unsqueeze(0))[0], dim=0)
            indices = torch.topk(probabilities, 2).indices
            values_top = torch.topk(probabilities, 2).values
            actual = labels[row["label_lsm"]]
            guess = int(indices[0])
            true.append(actual)
            predicted.append(guess)
            per_clip.append({"sample_id": row["sample_id"], "signer_id": row["signer_id"], "label_lsm": row["label_lsm"], "prediction_lsm": inverse[guess], "correct": int(actual == guess), "confidence": float(values_top[0]), "margin": float(values_top[0] - values_top[1])})
    targets = np.asarray(true, dtype=np.int64)
    predictions = np.asarray(predicted, dtype=np.int64)
    report = {"kind": "single_authorized_s09_evaluation", "seed": args.seed, "checkpoint": str(args.checkpoint), "samples": len(per_clip), "classes": 210, "signers": sorted({row["signer_id"] for row in test_rows}), "macro_f1": macro_f1(targets, predictions, 210), "accuracy": float(np.mean(targets == predictions)), "s09_evaluated": True, "training_performed": False, "selection_performed": False, "retry_allowed": False, "gate": str(args.gate), "predictions": per_clip}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, ensure_ascii=False))


if __name__ == "__main__":
    main()