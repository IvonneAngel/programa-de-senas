"""Confirmación LOSO emparejada S01-S07 de bone_vector126 contra positions126."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lsm.models.tcn import TemporalTCN, parameter_count


SEED = 42
EPOCHS = 40
PATIENCE = 8
SIGNERS = tuple(f"S{number:02d}" for number in range(1, 8))


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)


def macro_f1(targets: np.ndarray, predictions: np.ndarray, classes: int = 210) -> float:
    scores = []
    for label in range(classes):
        true_positive = int(np.sum((targets == label) & (predictions == label)))
        false_positive = int(np.sum((targets != label) & (predictions == label)))
        false_negative = int(np.sum((targets == label) & (predictions != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(np.mean(scores))


class CachedDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], cache_root: Path, labels: dict[str, int]):
        self.rows = rows
        self.cache_root = cache_root
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        values = np.load(self.cache_root / row["feature_path"], allow_pickle=False)
        if values.shape != (30, 126) or not np.isfinite(values).all():
            raise ValueError(f"Tensor inválido: {row['sample_id']}")
        return torch.from_numpy(values.astype(np.float32, copy=False)), torch.tensor(self.labels[row["label_lsm"]], dtype=torch.long)


def load_s01_s07(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["feature_status"] == "ok" and row["signer_id"] in SIGNERS]
    if len(selected) != 1470 or set(row["signer_id"] for row in selected) != set(SIGNERS):
        raise ValueError("Se requieren exactamente 1,470 clips de S01-S07")
    if len({row["label_lsm"] for row in selected}) != 210:
        raise ValueError("Se requieren 210 clases en S01-S07")
    return selected


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    count = 0
    predicted_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, targets in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Pérdida no finita")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            loss_total += float(loss.detach()) * targets.size(0)
            count += targets.size(0)
            predicted_chunks.append(logits.argmax(dim=1).detach().cpu().numpy())
            target_chunks.append(targets.detach().cpu().numpy())
    targets_all = np.concatenate(target_chunks)
    predictions_all = np.concatenate(predicted_chunks)
    return loss_total / count, macro_f1(targets_all, predictions_all)


def train_fold(rows: list[dict[str, str]], cache_root: Path, holdout: str) -> dict[str, int | float | str]:
    labels = {label: index for index, label in enumerate(sorted({row["label_lsm"] for row in rows}))}
    train_rows = [row for row in rows if row["signer_id"] != holdout]
    validation_rows = [row for row in rows if row["signer_id"] == holdout]
    if len(train_rows) != 1260 or len(validation_rows) != 210:
        raise ValueError(f"Fold {holdout} inválido: {len(train_rows)}/{len(validation_rows)}")
    if len({row["label_lsm"] for row in validation_rows}) != 210:
        raise ValueError(f"{holdout} debe aportar una muestra por clase")
    seed_everything()
    train_loader = DataLoader(CachedDataset(train_rows, cache_root, labels), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(SEED), num_workers=0)
    validation_loader = DataLoader(CachedDataset(validation_rows, cache_root, labels), batch_size=64, shuffle=False, num_workers=0)
    model = TemporalTCN(feature_dim=126, classes=210, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20)
    if parameter_count(model) != 158_994:
        raise AssertionError("Presupuesto TCN inesperado")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    best = -1.0
    best_epoch = 0
    stale = 0
    for epoch in range(1, EPOCHS + 1):
        run_epoch(model, train_loader, criterion, optimizer)
        _, validation_f1 = run_epoch(model, validation_loader, criterion)
        if validation_f1 > best:
            best, best_epoch, stale = validation_f1, epoch, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return {"held_out_signer": holdout, "best_macro_f1": best, "best_epoch": best_epoch, "train_samples": len(train_rows), "validation_samples": len(validation_rows), "seed": SEED}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions-manifest", type=Path, required=True)
    parser.add_argument("--positions-cache-root", type=Path, required=True)
    parser.add_argument("--bone-manifest", type=Path, required=True)
    parser.add_argument("--bone-cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only-signer", choices=SIGNERS, default=None)
    args = parser.parse_args()
    positions_rows = load_s01_s07(args.positions_manifest)
    bone_rows = load_s01_s07(args.bone_manifest)
    if [(row["sample_id"], row["label_lsm"], row["signer_id"]) for row in positions_rows] != [(row["sample_id"], row["label_lsm"], row["signer_id"]) for row in bone_rows]:
        raise ValueError("Las dos representaciones no comparten la misma población S01-S07")
    signers = (args.only_signer,) if args.only_signer else SIGNERS
    folds = []
    for signer in signers:
        control = train_fold(positions_rows, args.positions_cache_root, signer)
        bone = train_fold(bone_rows, args.bone_cache_root, signer)
        folds.append({"signer": signer, "positions126": control, "bone_vector126": bone, "delta_macro_f1": float(bone["best_macro_f1"]) - float(control["best_macro_f1"])})
        print(json.dumps(folds[-1]), flush=True)
    result: dict[str, object] = {"protocol": "LOSO S01-S07, seed 42, 40 epochs, patience 8", "folds": folds, "s08_read": False, "s09_read": False, "models_trained_from_scratch": True, "complete": len(folds) == 7}
    if len(folds) == 7:
        deltas = np.asarray([fold["delta_macro_f1"] for fold in folds], dtype=np.float64)
        rng = np.random.default_rng(2026)
        bootstrap = rng.choice(deltas, size=(10_000, len(deltas)), replace=True).mean(axis=1)
        result["delta_macro_f1"] = {"mean": float(deltas.mean()), "per_signer": {fold["signer"]: fold["delta_macro_f1"] for fold in folds}, "bootstrap_seed": 2026, "bootstrap_samples": 10_000, "ci95": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()