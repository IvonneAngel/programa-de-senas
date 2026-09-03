"""Genera predicciones solo de S08 para auditoría; S09 queda explícitamente prohibido."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from lsm.models.tcn import TemporalTCN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if any(row["split_model"] == "test" for row in rows if row["sample_id"] in {row["sample_id"] for row in rows if row["split_model"] == "validation"}):
        raise AssertionError("Una fila no puede ser validación y test")
    validation = [row for row in rows if row["split_model"] == "validation" and row["feature_status"] == "ok"]
    if len(validation) != 210 or len({row["label_lsm"] for row in validation}) != 210:
        raise ValueError("S08 debe contener 210 muestras y etiquetas")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    labels: dict[str, int] = state["labels"]
    if len(labels) != 210 or set(labels) != {row["label_lsm"] for row in validation}:
        raise ValueError("Etiquetas del checkpoint incompatibles con S08")
    inverse_labels = {index: label for label, index in labels.items()}
    model = TemporalTCN(feature_dim=126, classes=210, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    predictions: list[dict[str, str | int | float]] = []
    with torch.no_grad():
        for row in validation:
            values = np.load(args.cache_root / row["feature_path"], allow_pickle=False)
            if values.shape != (30, 126) or not np.isfinite(values).all():
                raise ValueError(f"S08 inválida: {row['sample_id']}")
            logits = model(torch.from_numpy(values.astype(np.float32, copy=False)).unsqueeze(0))[0]
            probabilities = torch.softmax(logits, dim=0)
            ranking = torch.topk(probabilities, k=2)
            prediction = int(ranking.indices[0])
            predictions.append({
                "seed": args.seed,
                "sample_id": row["sample_id"],
                "signer_id": row["signer_id"],
                "label_lsm": row["label_lsm"],
                "prediction_lsm": inverse_labels[prediction],
                "correct": int(inverse_labels[prediction] == row["label_lsm"]),
                "confidence": float(ranking.values[0]),
                "margin": float(ranking.values[0] - ranking.values[1]),
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps({"seed": args.seed, "rows": len(predictions), "split_read": "validation_only", "s09_read": False, "out": str(args.out)}))


if __name__ == "__main__":
    main()