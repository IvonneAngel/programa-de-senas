"""Confirmación LOSO train-only de fusión uniforme bone/cov/code."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader, Dataset

from bone_code190 import transform_sequence as code_transform
from lsm.models.tcn import TemporalTCN, parameter_count


SEED = 42
EPOCHS = 40
PATIENCE = 8
SIGNERS = tuple(f"S{number:02d}" for number in range(1, 8))
CLUSTERS = 32
MAX_CODEBOOK_FRAMES = 20_000


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)


def macro_f1(targets: np.ndarray, predictions: np.ndarray, classes: int = 210) -> float:
    scores = []
    for label in range(classes):
        tp = int(np.sum((targets == label) & (predictions == label)))
        fp = int(np.sum((targets != label) & (predictions == label)))
        fn = int(np.sum((targets == label) & (predictions != label)))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
    return float(np.mean(scores))


def load_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["feature_status"] == "ok" and row["signer_id"] in SIGNERS]
    if len(rows) != 1470 or set(row["signer_id"] for row in rows) != set(SIGNERS) or len({row["label_lsm"] for row in rows}) != 210:
        raise ValueError("LOSO requiere 1,470 clips, S01-S07 y 210 etiquetas")
    return sorted(rows, key=lambda row: row["sample_id"])


def observed_hands(values: np.ndarray) -> np.ndarray:
    sequence = np.asarray(values, dtype=np.float32)
    if sequence.shape != (30, 126) or not np.isfinite(sequence).all():
        raise ValueError("bone inválido")
    hands = sequence.reshape(30, 2, 63).reshape(60, 63)
    return hands[np.linalg.norm(hands, axis=1) > 1e-6]


def fit_codebook(rows: list[dict[str, str]], bone_root: Path) -> np.ndarray:
    frames = np.concatenate([observed_hands(np.load(bone_root / row["feature_path"], allow_pickle=False)) for row in rows], axis=0)
    indices = np.linspace(0, frames.shape[0] - 1, min(MAX_CODEBOOK_FRAMES, frames.shape[0]), dtype=np.int64)
    centers = KMeans(n_clusters=CLUSTERS, random_state=20_260_819, n_init=8, max_iter=300, algorithm="lloyd").fit(frames[indices]).cluster_centers_.astype(np.float32)
    if centers.shape != (32, 63) or not np.isfinite(centers).all():
        raise ValueError("Codebook LOSO inválido")
    return centers


class ViewDataset(Dataset):
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
            raise ValueError(f"Tensor inválido: {row['sample_id']}")
        return torch.from_numpy(values.astype(np.float32, copy=False)), torch.tensor(self.labels[row["label_lsm"]], dtype=torch.long)


def one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    outputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    total_loss = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, truth in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, truth)
            if not torch.isfinite(loss):
                raise FloatingPointError("Pérdida no finita")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += float(loss.detach()) * truth.size(0)
            count += truth.size(0)
            outputs.append(logits.detach().cpu().numpy())
            targets.append(truth.detach().cpu().numpy())
    return total_loss / count, np.concatenate(outputs), np.concatenate(targets)


def train_view(train_rows: list[dict[str, str]], val_rows: list[dict[str, str]], root: Path, labels: dict[str, int], feature_dim: int, expected_params: int, codebook: np.ndarray | None = None) -> tuple[TemporalTCN, float, int]:
    seed_everything()
    train_dataset = ViewDataset(train_rows, root, labels, feature_dim, codebook)
    val_dataset = ViewDataset(val_rows, root, labels, feature_dim, codebook)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(SEED), num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    model = TemporalTCN(feature_dim=feature_dim, classes=210, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20)
    if parameter_count(model) != expected_params:
        raise AssertionError("Presupuesto TCN inesperado")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    best, best_epoch, stale, state = -1.0, 0, 0, None
    for epoch in range(1, EPOCHS + 1):
        one_epoch(model, train_loader, criterion, optimizer)
        _, logits, targets = one_epoch(model, val_loader, criterion)
        value = macro_f1(targets, logits.argmax(axis=1))
        if value > best:
            best, best_epoch, stale = value, epoch, 0
            state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if state is None:
        raise AssertionError("Sin estado mejor")
    model.load_state_dict(state, strict=True)
    return model.eval(), best, best_epoch


def predict(model: TemporalTCN, rows: list[dict[str, str]], root: Path, labels: dict[str, int], feature_dim: int, codebook: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(ViewDataset(rows, root, labels, feature_dim, codebook), batch_size=64, shuffle=False, num_workers=0)
    _, logits, targets = one_epoch(model, loader, nn.CrossEntropyLoss())
    return logits, targets


def run_fold(bone_rows: list[dict[str, str]], cov_rows: list[dict[str, str]], holdout: str, bone_root: Path, cov_root: Path, save_folds: Path | None = None) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    train_bone = [row for row in bone_rows if row["signer_id"] != holdout]
    val_bone = [row for row in bone_rows if row["signer_id"] == holdout]
    train_cov = [row for row in cov_rows if row["signer_id"] != holdout]
    val_cov = [row for row in cov_rows if row["signer_id"] == holdout]
    if len(train_bone) != 1260 or len(val_bone) != 210 or [(x["sample_id"], x["label_lsm"]) for x in train_bone] != [(x["sample_id"], x["label_lsm"]) for x in train_cov] or [(x["sample_id"], x["label_lsm"]) for x in val_bone] != [(x["sample_id"], x["label_lsm"]) for x in val_cov]:
        raise ValueError(f"Fold {holdout} incompatible")
    labels = {label: index for index, label in enumerate(sorted({row["label_lsm"] for row in bone_rows}))}
    codebook = fit_codebook(train_bone, bone_root)
    bone, bone_f1, bone_epoch = train_view(train_bone, val_bone, bone_root, labels, 126, 158_994)
    cov, cov_f1, cov_epoch = train_view(train_cov, val_cov, cov_root, labels, 168, 167_058)
    code, code_f1, code_epoch = train_view(train_bone, val_bone, bone_root, labels, 190, 171_282, codebook)
    bone_logits, targets = predict(bone, val_bone, bone_root, labels, 126)
    cov_logits, cov_targets = predict(cov, val_cov, cov_root, labels, 168)
    code_logits, code_targets = predict(code, val_bone, bone_root, labels, 190, codebook)
    if not np.array_equal(targets, cov_targets) or not np.array_equal(targets, code_targets):
        raise ValueError("Objetivos LOSO incompatibles")
    fused = macro_f1(targets, ((bone_logits + cov_logits + code_logits) / 3.0).argmax(axis=1))
    if save_folds:
        fold_dir = save_folds / holdout
        fold_dir.mkdir(parents=True, exist_ok=True)
        metadata = {"holdout": holdout, "train_signers": sorted({row["signer_id"] for row in train_bone}), "labels": labels, "s08_read": False, "s09_read": False, "models_trained_from_scratch": True}
        for name, model, feature_dim in (("bone", bone, 126), ("cov", cov, 168), ("code", code, 190)):
            torch.save({"state_dict": model.state_dict(), "feature_dim": feature_dim, **metadata}, fold_dir / f"{name}.pt")
        np.save(fold_dir / "codebook32.npy", codebook, allow_pickle=False)
        (fold_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summary = {"signer": holdout, "train_samples": 1260, "validation_samples": 210, "codebook_fit_signers": sorted({row["signer_id"] for row in train_bone}), "codebook_excludes_holdout": holdout not in {row["signer_id"] for row in train_bone}, "bone": {"macro_f1": bone_f1, "epoch": bone_epoch}, "cov": {"macro_f1": cov_f1, "epoch": cov_epoch}, "code": {"macro_f1": code_f1, "epoch": code_epoch}, "fusion_uniform": {"macro_f1": fused, "weights": [1 / 3, 1 / 3, 1 / 3]}, "delta_fusion_minus_bone": fused - bone_f1}
    oof = {"sample_ids": np.asarray([row["sample_id"] for row in val_bone]), "signers": np.asarray([holdout] * len(val_bone)), "targets": targets.astype(np.int64, copy=False), "bone_logits": bone_logits.astype(np.float32, copy=False), "cov_logits": cov_logits.astype(np.float32, copy=False), "code_logits": code_logits.astype(np.float32, copy=False)}
    return summary, oof


def main() -> None:
    global SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--bone-manifest", type=Path, required=True)
    parser.add_argument("--bone-cache-root", type=Path, required=True)
    parser.add_argument("--cov-manifest", type=Path, required=True)
    parser.add_argument("--cov-cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-oof", type=Path)
    parser.add_argument("--save-folds", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--only-signer", choices=SIGNERS, default=None)
    args = parser.parse_args()
    SEED = args.seed
    bone_rows, cov_rows = load_rows(args.bone_manifest), load_rows(args.cov_manifest)
    if [(row["sample_id"], row["label_lsm"], row["signer_id"]) for row in bone_rows] != [(row["sample_id"], row["label_lsm"], row["signer_id"]) for row in cov_rows]:
        raise ValueError("Vistas bone/cov no comparten S01-S07")
    signers = (args.only_signer,) if args.only_signer else SIGNERS
    folds = []
    oof_parts: list[dict[str, np.ndarray]] = []
    for signer in signers:
        fold, oof = run_fold(bone_rows, cov_rows, signer, args.bone_cache_root, args.cov_cache_root, args.save_folds)
        folds.append(fold)
        oof_parts.append(oof)
        print(json.dumps(fold), flush=True)
    report: dict[str, object] = {"protocol": f"LOSO S01-S07; seed {SEED}; 40 epochs; patience 8; codebook K-means train-six-only; fusión uniforme de logits", "seed": SEED, "formula": "(bone+cov+code)/3", "s08_read": False, "s09_read": False, "s09_evaluated": False, "models_trained_from_scratch": True, "folds": folds, "complete": len(folds) == 7}
    if len(folds) == 7:
        deltas = np.asarray([float(fold["delta_fusion_minus_bone"]) for fold in folds], dtype=np.float64)
        rng = np.random.default_rng(2026)
        bootstrap = rng.choice(deltas, size=(10_000, len(deltas)), replace=True).mean(axis=1)
        report["delta_macro_f1"] = {"mean": float(deltas.mean()), "per_signer": {str(fold["signer"]): float(fold["delta_fusion_minus_bone"]) for fold in folds}, "bootstrap_seed": 2026, "bootstrap_samples": 10_000, "ci95": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.out_oof:
        if len(oof_parts) != 7:
            raise ValueError("La auditoría OOF solo se escribe para LOSO completo")
        args.out_oof.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out_oof, **{key: np.concatenate([part[key] for part in oof_parts], axis=0) for key in oof_parts[0]})
    print(json.dumps(report))


if __name__ == "__main__":
    main()