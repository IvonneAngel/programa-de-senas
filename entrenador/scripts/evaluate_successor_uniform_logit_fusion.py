"""Evalúa la fusión uniforme preregistrada exclusivamente sobre S08."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from lsm.models.tcn import TemporalTCN


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uniform_logits(logit_views: list[torch.Tensor]) -> torch.Tensor:
    if len(logit_views) != 3:
        raise ValueError("La fusión preregistrada exige exactamente tres vistas")
    first = logit_views[0]
    if any(view.shape != first.shape for view in logit_views) or first.ndim != 2 or first.shape[1] != 210:
        raise ValueError("Logits incompatibles con la fusión de 210 clases")
    return torch.stack(logit_views, dim=0).mean(dim=0)


def convex_logits(logit_views: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    if len(weights) != 3 or any(weight < 0 for weight in weights) or not np.isclose(sum(weights), 1.0, atol=1e-12):
        raise ValueError("Los pesos deben ser tres valores no negativos que sumen uno")
    if len(logit_views) != 3:
        raise ValueError("La fusión convexa exige exactamente tres vistas")
    first = logit_views[0]
    if any(view.shape != first.shape for view in logit_views) or first.ndim != 2 or first.shape[1] != 210:
        raise ValueError("Logits incompatibles con la fusión de 210 clases")
    return sum(weight * view for weight, view in zip(weights, logit_views))


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["feature_status"] == "ok" and row["split_model"] == "validation"]
    if len(rows) != 210 or len({row["sample_id"] for row in rows}) != 210 or {row["signer_id"] for row in rows} != {"S08"}:
        raise ValueError("La fusión solo admite exactamente S08 con 210 muestras")
    return sorted(rows, key=lambda row: row["sample_id"])


def load_model(path: Path, feature_dim: int, expected_hash: str) -> tuple[TemporalTCN, dict[str, int]]:
    if sha256(path) != expected_hash:
        raise ValueError(f"Checkpoint alterado: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    labels = artifact["labels"]
    model = TemporalTCN(feature_dim=feature_dim, classes=210, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    model.eval()
    return model, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bone-manifest", type=Path, required=True)
    parser.add_argument("--bone-cache-root", type=Path, required=True)
    parser.add_argument("--bone-checkpoint", type=Path, required=True)
    parser.add_argument("--bone-sha256", required=True)
    parser.add_argument("--cov-manifest", type=Path, required=True)
    parser.add_argument("--cov-cache-root", type=Path, required=True)
    parser.add_argument("--cov-checkpoint", type=Path, required=True)
    parser.add_argument("--cov-sha256", required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--code-cache-root", type=Path, required=True)
    parser.add_argument("--code-checkpoint", type=Path, required=True)
    parser.add_argument("--code-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--weights", type=float, nargs=3, default=[1 / 3, 1 / 3, 1 / 3])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bone_rows, cov_rows, code_rows = load_rows(args.bone_manifest), load_rows(args.cov_manifest), load_rows(args.code_manifest)
    for left, right in ((bone_rows, cov_rows), (bone_rows, code_rows)):
        if [(row["sample_id"], row["label_lsm"]) for row in left] != [(row["sample_id"], row["label_lsm"]) for row in right]:
            raise ValueError("Las vistas no comparten S08 ni etiquetas exactamente")
    bone, labels = load_model(args.bone_checkpoint, 126, args.bone_sha256)
    cov, cov_labels = load_model(args.cov_checkpoint, 168, args.cov_sha256)
    code, code_labels = load_model(args.code_checkpoint, 190, args.code_sha256)
    if labels != cov_labels or labels != code_labels or len(labels) != 210:
        raise ValueError("Los checkpoints no comparten el mismo mapeo de etiquetas")
    targets = np.asarray([labels[row["label_lsm"]] for row in bone_rows], dtype=np.int64)
    predictions: list[int] = []
    with torch.no_grad():
        for bone_row, cov_row, code_row in zip(bone_rows, cov_rows, code_rows):
            features = [
                torch.from_numpy(np.load(args.bone_cache_root / bone_row["feature_path"], allow_pickle=False).astype(np.float32)).unsqueeze(0),
                torch.from_numpy(np.load(args.cov_cache_root / cov_row["feature_path"], allow_pickle=False).astype(np.float32)).unsqueeze(0),
                torch.from_numpy(np.load(args.code_cache_root / code_row["feature_path"], allow_pickle=False).astype(np.float32)).unsqueeze(0),
            ]
            logits = convex_logits([bone(features[0]), cov(features[1]), code(features[2])], args.weights)
            predictions.append(int(logits.argmax(dim=1).item()))
    macro_f1 = float(f1_score(targets, np.asarray(predictions), average="macro", zero_division=0))
    inverse_labels = {value: key for key, value in labels.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "s08_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "true_label", "predicted_label", "correct"])
        writer.writeheader()
        for row, prediction in zip(bone_rows, predictions):
            writer.writerow({"sample_id": row["sample_id"], "true_label": row["label_lsm"], "predicted_label": inverse_labels[prediction], "correct": int(prediction == labels[row["label_lsm"]])})
    report = {"kind": "successor_convex_logit_fusion", "formula": "sum(weight_i*logit_i)", "weights": args.weights, "seed": args.seed, "s08_samples": len(bone_rows), "s09_read": False, "s09_evaluated": False, "training_performed": False, "best_validation_macro_f1": macro_f1, "checkpoint_sha256": {"bone": args.bone_sha256, "cov": args.cov_sha256, "code": args.code_sha256}}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()