"""Evaluación S08 única del bagging cross-signer preregistrado."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from bone_code190 import transform_sequence as code_transform
from lsm.models.tcn import TemporalTCN, parameter_count


FOLDS = tuple(f"S{number:02d}" for number in range(1, 8))


def macro_f1(targets: np.ndarray, predictions: np.ndarray, classes: int = 210) -> float:
    scores = []
    for label in range(classes):
        tp = int(np.sum((targets == label) & (predictions == label)))
        fp = int(np.sum((targets != label) & (predictions == label)))
        fn = int(np.sum((targets == label) & (predictions != label)))
        scores.append(0.0 if (denominator := 2 * tp + fp + fn) == 0 else 2.0 * tp / denominator)
    return float(np.mean(scores))


def rows_for_split(manifest: Path, split: str) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["feature_status"] == "ok" and row["split_model"] == split]
    if len(rows) != 210 or len({row["label_lsm"] for row in rows}) != 210 or len({row["signer_id"] for row in rows}) != 1:
        raise ValueError(f"{split} debe contener 210 clips, 210 etiquetas y un firmante")
    return sorted(rows, key=lambda row: row["sample_id"])


def verify_hashes(path: Path) -> None:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 35:
        raise ValueError("Se requieren 35 huellas cross-signer")
    for line in lines:
        digest, filename = line.split("  ", 1)
        candidate = Path(filename)
        if not candidate.exists() or hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Huella inválida: {filename}")


class CachedDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], root: Path, labels: dict[str, int], feature_dim: int, codebook: np.ndarray | None = None):
        self.rows, self.root, self.labels, self.feature_dim, self.codebook = rows, root, labels, feature_dim, codebook

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        values = np.load(self.root / row["feature_path"], allow_pickle=False)
        if self.codebook is not None:
            values, _ = code_transform(values, self.codebook)
        if values.shape != (30, self.feature_dim) or not np.isfinite(values).all():
            raise ValueError(f"Entrada inválida: {row['sample_id']}")
        return torch.from_numpy(values.astype(np.float32, copy=False)), torch.tensor(self.labels[row["label_lsm"]], dtype=torch.long)


def load_model(path: Path, expected_dim: int) -> tuple[TemporalTCN, dict[str, int]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["feature_dim"]) != expected_dim:
        raise ValueError(f"Dimensión inesperada: {path}")
    labels = payload["labels"]
    model = TemporalTCN(feature_dim=expected_dim, classes=210, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20)
    expected_params = {126: 158_994, 168: 167_058, 190: 171_282}[expected_dim]
    if parameter_count(model) != expected_params:
        raise ValueError("Presupuesto TCN inesperado")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval(), labels


def logits(model: TemporalTCN, dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    all_logits, all_targets = [], []
    with torch.no_grad():
        for features, targets in loader:
            all_logits.append(model(features).cpu().numpy())
            all_targets.append(targets.numpy())
    return np.concatenate(all_logits), np.concatenate(all_targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bone-manifest", type=Path, required=True)
    parser.add_argument("--bone-cache-root", type=Path, required=True)
    parser.add_argument("--cov-manifest", type=Path, required=True)
    parser.add_argument("--cov-cache-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--hashes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify_hashes(args.hashes)
    bone_rows, cov_rows = rows_for_split(args.bone_manifest, "validation"), rows_for_split(args.cov_manifest, "validation")
    if [(row["sample_id"], row["label_lsm"]) for row in bone_rows] != [(row["sample_id"], row["label_lsm"]) for row in cov_rows]:
        raise ValueError("S08 bone/cov incompatible")
    per_fold, target_reference = [], None
    for fold in FOLDS:
        folder = args.fold_root / fold
        bone, labels = load_model(folder / "bone.pt", 126)
        cov, cov_labels = load_model(folder / "cov.pt", 168)
        code, code_labels = load_model(folder / "code.pt", 190)
        if labels != cov_labels or labels != code_labels or sorted(labels.values()) != list(range(210)):
            raise ValueError(f"Etiquetas incompatibles en {fold}")
        codebook = np.load(folder / "codebook32.npy", allow_pickle=False)
        bone_logits, targets = logits(bone, CachedDataset(bone_rows, args.bone_cache_root, labels, 126))
        cov_logits, cov_targets = logits(cov, CachedDataset(cov_rows, args.cov_cache_root, labels, 168))
        code_logits, code_targets = logits(code, CachedDataset(bone_rows, args.bone_cache_root, labels, 190, codebook))
        if not np.array_equal(targets, cov_targets) or not np.array_equal(targets, code_targets):
            raise ValueError(f"Objetivos incompatibles en {fold}")
        if target_reference is None:
            target_reference = targets
        elif not np.array_equal(target_reference, targets):
            raise ValueError("Objetivos distintos entre folds")
        per_fold.append((bone_logits + cov_logits + code_logits) / 3.0)
    combined = np.mean(np.stack(per_fold), axis=0)
    predictions = combined.argmax(axis=1)
    report = {"candidate": "cross_signer_bagging_uniform", "formula": "mean_folds(mean_views(bone,cov,code))", "folds": list(FOLDS), "weights": {"fold": 1 / 7, "view": 1 / 3, "logit": 1 / 21}, "s08_read": True, "s09_read": False, "s09_evaluated": False, "hash_manifest": str(args.hashes), "macro_f1_s08": macro_f1(target_reference, predictions), "samples": int(target_reference.size), "predictions": [{"sample_id": row["sample_id"], "true_label": row["label_lsm"], "predicted_index": int(prediction)} for row, prediction in zip(bone_rows, predictions)]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}))


if __name__ == "__main__":
    main()