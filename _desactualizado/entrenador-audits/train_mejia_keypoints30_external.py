"""Entrenador aislado desde cero para 30 señas LSM RGB-D de Mejía et al."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lsm.models.tcn import TemporalTCN, parameter_count


TASK = "mejia_keypoints30_external"
EXPECTED_SHAPE = (20, 201)
EXPECTED_COUNTS = {"train": 2040, "validation": 510, "test": 450}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def macro_f1(predictions: np.ndarray, targets: np.ndarray, classes: int) -> float:
    scores: list[float] = []
    for label in range(classes):
        true_positive = int(np.sum((predictions == label) & (targets == label)))
        false_positive = int(np.sum((predictions == label) & (targets != label)))
        false_negative = int(np.sum((predictions != label) & (targets == label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
    return float(np.mean(scores))


def load_partitions(manifest_path: Path) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"sample_id", "label", "sample_index", "source_partition", "split_external", "feature_path", "frames", "channels", "participant_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Manifiesto Mejía sin columnas requeridas")
    labels = sorted({row["label"] for row in rows})
    partitions = {split: [row for row in rows if row["split_external"] == split] for split in EXPECTED_COUNTS}
    if len(rows) != 3000 or len(labels) != 30 or {split: len(items) for split, items in partitions.items()} != EXPECTED_COUNTS:
        raise ValueError("Cobertura del corpus Mejía incompatible con el protocolo")
    for split, expected_per_label in (("train", 68), ("validation", 17), ("test", 15)):
        counts = {label: sum(row["label"] == label for row in partitions[split]) for label in labels}
        if set(counts.values()) != {expected_per_label}:
            raise ValueError(f"Cobertura por clase incorrecta en {split}: {counts}")
    if any(row["participant_id"] != "unavailable_in_release" for row in rows):
        raise ValueError("El release Mejía no permite declarar participantes recuperables")
    return partitions, labels


@dataclass(frozen=True)
class Record:
    feature_path: str
    label: int


class KeypointDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, str]], label_map: dict[str, int], source_root: Path) -> None:
        self.records = [Record(row["feature_path"], label_map[row["label"]]) for row in rows]
        self.source_root = source_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        values = np.loadtxt(self.source_root / record.feature_path, delimiter=",", skiprows=1, dtype=np.float32)
        values = values[:, 1:]  # índice CSV; los 201 valores restantes son keypoints 3D publicados.
        if values.shape != EXPECTED_SHAPE or not np.isfinite(values).all():
            raise ValueError(f"Keypoints Mejía inválidos: {record.feature_path}")
        return torch.from_numpy(values), torch.tensor(record.label, dtype=torch.long)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer | None, classes: int) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, total = 0.0, 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for features, labels in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach()) * labels.shape[0]
        total += labels.shape[0]
        predictions.append(logits.argmax(dim=1).detach().numpy())
        targets.append(labels.numpy())
    predicted, expected = np.concatenate(predictions), np.concatenate(targets)
    return {"loss": total_loss / total, "accuracy": float(np.mean(predicted == expected)), "macro_f1": macro_f1(predicted, expected, classes)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-checkpoint", type=Path, help="Consulta una vez Testing con checkpoint congelado")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.lr <= 0:
        raise ValueError("Hiperparámetros inválidos")
    seed_everything(args.seed)
    partitions, label_list = load_partitions(args.manifest)
    label_map = {label: index for index, label in enumerate(label_list)}
    datasets = {split: KeypointDataset(rows, label_map, args.source_root) for split, rows in partitions.items()}
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"), num_workers=0) for split, dataset in datasets.items()}
    model = TemporalTCN(feature_dim=201, classes=len(label_list), frames=20)
    criterion = nn.CrossEntropyLoss()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.evaluate_checkpoint is not None:
        checkpoint = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("labels") != label_list or checkpoint.get("feature_shape") != list(EXPECTED_SHAPE):
            raise ValueError("Checkpoint incompatible con el corpus Mejía")
        model.load_state_dict(checkpoint["model_state_dict"])
        with torch.inference_mode():
            test_metrics = run_epoch(model, loaders["test"], criterion, None, len(label_list))
        report = {"kind": TASK, "evaluation_mode": "frozen_checkpoint_published_test", "checkpoint": str(args.evaluate_checkpoint), "counts": EXPECTED_COUNTS, "test": test_metrics, "training_from_scratch": True, "external_pretrained_weights_loaded": False, "participant_level_claim_permitted": False, "benchmark_210_words_touched": False, "s08_metrics_evaluated": False, "s09_metrics_evaluated": False}
        (args.out / "external_test_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"done": True, "evaluation_mode": report["evaluation_mode"], "test": test_metrics}), flush=True)
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score, stale, history, started = -1.0, 0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, loaders["train"], criterion, optimizer, len(label_list))
        with torch.inference_mode():
            validation = run_epoch(model, loaders["validation"], criterion, None, len(label_list))
        record = {"epoch": epoch, "train": train, "validation": validation, "elapsed_seconds": time.time() - started}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["macro_f1"] > best_score:
            best_score, stale = validation["macro_f1"], 0
            torch.save({"model_state_dict": model.state_dict(), "labels": label_list, "feature_shape": list(EXPECTED_SHAPE), "seed": args.seed, "training_from_scratch": True, "external_pretrained_weights_loaded": False}, args.out / "best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break
    report = {"kind": TASK, "counts": EXPECTED_COUNTS, "labels": label_list, "parameters": parameter_count(model), "best_validation_macro_f1": best_score, "test_evaluated": False, "history": history, "training_from_scratch": True, "external_pretrained_weights_loaded": False, "participant_level_claim_permitted": False, "benchmark_210_words_touched": False, "s08_metrics_evaluated": False, "s09_metrics_evaluated": False}
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"done": True, "best_validation_macro_f1": best_score, "test_evaluated": False}), flush=True)


if __name__ == "__main__":
    main()