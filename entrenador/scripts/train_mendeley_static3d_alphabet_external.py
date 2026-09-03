"""Entrena un PointNet compacto desde cero para el alfabeto LSM 3D normalizado."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


EXPECTED_POINTS = 512
EXPECTED_CLASSES = 21


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, int]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"sample_id", "label", "capture_index", "split_external", "feature_path", "points", "channels", "capture_index_is_verified_participant"}
    if len(rows) != 315 or not rows or not required.issubset(rows[0]):
        raise ValueError("Manifiesto 3D estático inválido")
    splits = {split: [row for row in rows if row["split_external"] == split] for split in ("train", "validation", "test")}
    if {key: len(value) for key, value in splits.items()} != {"train": 231, "validation": 42, "test": 42}:
        raise ValueError("Split 3D estático inválido")
    labels = {label: index for index, label in enumerate(sorted({row["label"] for row in splits["train"]}))}
    if len(labels) != EXPECTED_CLASSES or any({row["label"] for row in values} != set(labels) for values in splits.values()):
        raise ValueError("Cobertura de letras inválida")
    if any(row["capture_index_is_verified_participant"] != "False" for row in rows):
        raise ValueError("No se permiten afirmaciones de firmante no recuperables")
    return splits, labels


class PointCloudRows(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, str]], labels: dict[str, int], source_root: Path) -> None:
        self.rows, self.labels, self.source_root = rows, labels, source_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        values = np.loadtxt(self.source_root / row["feature_path"], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 2 or not np.isfinite(values).all():
            raise ValueError(f"Nube inválida: {row['feature_path']}")
        sample = values[np.linspace(0, values.shape[0] - 1, EXPECTED_POINTS, dtype=np.int64)]
        return torch.from_numpy(sample), torch.tensor(self.labels[row["label"]], dtype=torch.long)


class CompactPointNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.points = nn.Sequential(nn.Conv1d(3, 64, 1), nn.GELU(), nn.Conv1d(64, 128, 1), nn.GELU(), nn.Conv1d(128, 256, 1), nn.GELU())
        self.head = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.20), nn.Linear(128, EXPECTED_CLASSES))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (EXPECTED_POINTS, 3):
            raise ValueError(f"Forma PointNet inválida: {tuple(values.shape)}")
        features = self.points(values.transpose(1, 2))
        return self.head(torch.cat((features.amax(dim=2), features.mean(dim=2)), dim=1))


def evaluate(model: CompactPointNet, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval(); predictions: list[np.ndarray] = []; targets_out: list[np.ndarray] = []
    with torch.inference_mode():
        for values, targets in loader:
            predictions.append(model(values.to(device)).argmax(dim=1).cpu().numpy()); targets_out.append(targets.numpy())
    targets_arr, predictions_arr = np.concatenate(targets_out), np.concatenate(predictions)
    return float(f1_score(targets_arr, predictions_arr, average="macro", zero_division=0)), float(accuracy_score(targets_arr, predictions_arr))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--source-root", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--epochs", type=int, default=80); parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--lr", type=float, default=0.002); parser.add_argument("--weight-decay", type=float, default=0.0001); parser.add_argument("--patience", type=int, default=15); parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluate-checkpoint", type=Path)
    args = parser.parse_args()
    if args.epochs != 80 or args.seed != 42 or args.lr != 0.002 or args.weight_decay != 0.0001 or args.patience != 15:
        raise ValueError("Hiperparámetros externos divergentes del preregistro")
    seed_everything(args.seed)
    splits, labels = load_rows(args.manifest)
    device = torch.device(args.device)
    validation_loader = DataLoader(PointCloudRows(splits["validation"], labels, args.source_root), batch_size=args.batch_size, shuffle=False, num_workers=0)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.evaluate_checkpoint is not None:
        checkpoint = torch.load(args.evaluate_checkpoint, map_location=device, weights_only=False)
        if checkpoint.get("labels") != labels or checkpoint.get("test_evaluated") is not False:
            raise ValueError("Checkpoint externo no congelado o incompatible")
        model = CompactPointNet().to(device); model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        test_loader = DataLoader(PointCloudRows(splits["test"], labels, args.source_root), batch_size=args.batch_size, shuffle=False, num_workers=0)
        macro_f1, accuracy = evaluate(model, test_loader, device)
        report = {"kind": "mendeley_static3d_alphabet_pointnet_test", "checkpoint_sha256": sha256(args.evaluate_checkpoint), "test_samples": len(splits["test"]), "test_macro_f1": macro_f1, "test_accuracy": accuracy, "test_evaluated": True, "participant_level_claim_permitted": False, "s08_read": False, "s09_read": False}
        (args.out / "test_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False)); return
    train_loader = DataLoader(PointCloudRows(splits["train"], labels, args.source_root), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    model = CompactPointNet().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay); criterion = nn.CrossEntropyLoss()
    best, best_epoch, stale, history = -1.0, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0; count = 0
        for values, targets in train_loader:
            optimizer.zero_grad(set_to_none=True); loss = criterion(model(values.to(device)), targets.to(device))
            if not torch.isfinite(loss): raise FloatingPointError("Pérdida PointNet no finita")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); total_loss += float(loss.detach()) * targets.size(0); count += targets.size(0)
        validation_f1, validation_accuracy = evaluate(model, validation_loader, device)
        record = {"epoch": epoch, "train_loss": total_loss / count, "validation_macro_f1": validation_f1, "validation_accuracy": validation_accuracy, "test_used": False}; history.append(record); print(json.dumps(record), flush=True)
        if validation_f1 > best:
            best, best_epoch, stale = validation_f1, epoch, 0
            torch.save({"kind": "mendeley_static3d_alphabet_pointnet", "model_state_dict": model.state_dict(), "labels": labels, "epoch": epoch, "seed": args.seed, "training_from_scratch": True, "external_pretrained_weights_loaded": False, "test_evaluated": False, "participant_level_claim_permitted": False, "s08_read": False, "s09_read": False}, args.out / "best.pt")
        else:
            stale += 1
            if stale >= args.patience: break
    report = {"kind": "mendeley_static3d_alphabet_pointnet_train", "seed": args.seed, "training_from_scratch": True, "external_pretrained_weights_loaded": False, "train_samples": len(splits["train"]), "validation_samples": len(splits["validation"]), "test_samples_closed": len(splits["test"]), "best_validation_macro_f1": best, "best_epoch": best_epoch, "test_evaluated": False, "participant_level_claim_permitted": False, "s08_read": False, "s09_read": False, "manifest_sha256": sha256(args.manifest), "history": history}
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps({"best_validation_macro_f1": best, "best_epoch": best_epoch, "test_evaluated": False, "s09_read": False}))


if __name__ == "__main__":
    main()