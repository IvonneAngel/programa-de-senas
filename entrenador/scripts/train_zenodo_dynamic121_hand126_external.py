"""Entrenador aislado desde cero para las 121 glosas dinámicas Zenodo."""
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


TASK = "zenodo_dynamic121_lsm_hand126_external"
EXPECTED_CLASSES = 121
EXPECTED_SHAPE = (30, 126)
EXPECTED_SIGNERS = {
    "train": frozenset(f"S{value:02d}" for value in range(1, 9)),
    "validation": frozenset({"S09", "S10"}),
    "test": frozenset({"S11", "S12"}),
}


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


def load_and_partition(manifest_path: Path) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = {"sample_id", "label_lsm", "participant_id", "split_external", "feature_status", "feature_path"}
    if not rows or not columns.issubset(rows[0]):
        raise ValueError("Manifiesto Zenodo dinámico 121 sin columnas requeridas")
    rows = [row for row in rows if row["feature_status"] == "ok"]
    labels = sorted({row["label_lsm"] for row in rows})
    if len(labels) != EXPECTED_CLASSES:
        raise ValueError(f"Se esperaban {EXPECTED_CLASSES} glosas, se recibieron {len(labels)}")
    partitions = {split: [row for row in rows if row["split_external"] == split] for split in EXPECTED_SIGNERS}
    if any(not split_rows for split_rows in partitions.values()):
        raise ValueError("Falta un split externo de Zenodo dinámico")
    signer_sets = {split: {row["participant_id"] for row in split_rows} for split, split_rows in partitions.items()}
    if signer_sets != EXPECTED_SIGNERS:
        raise ValueError("Participantes inesperados o fuga entre splits externos Zenodo")
    if any(signer_sets[left] & signer_sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Fuga de participantes entre splits Zenodo dinámicos")
    for split in ("train", "validation"):
        if {row["label_lsm"] for row in partitions[split]} != set(labels):
            raise ValueError(f"El split {split} debe cubrir las 121 glosas")
    if not {row["label_lsm"] for row in partitions["test"]}.issubset(set(labels)):
        raise ValueError("El test externo contiene una glosa fuera del inventario")
    return partitions, labels


@dataclass(frozen=True)
class Record:
    feature_path: str
    label: int


class LandmarkDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, str]], labels: dict[str, int], cache_root: Path) -> None:
        self.records = [Record(row["feature_path"], labels[row["label_lsm"]]) for row in rows]
        self.cache_root = cache_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        values = np.load(self.cache_root / record.feature_path, allow_pickle=False)
        if values.shape != EXPECTED_SHAPE or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"Landmarks Zenodo dinámicos inválidos: {record.feature_path}")
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
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument("--evaluate-checkpoint", type=Path, help="Evalúa una vez un checkpoint congelado sobre el test externo")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.lr <= 0:
        raise ValueError("Hiperparámetros de entrenamiento inválidos")
    seed_everything(args.seed)
    partitions, label_list = load_and_partition(args.manifest)
    labels = {label: index for index, label in enumerate(label_list)}
    datasets = {split: LandmarkDataset(rows, labels, args.cache_root) for split, rows in partitions.items()}
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"), num_workers=0) for split, dataset in datasets.items()}
    model = TemporalTCN(feature_dim=126, classes=len(label_list), frames=30)
    criterion = nn.CrossEntropyLoss()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.evaluate_checkpoint is not None:
        if args.skip_test_evaluation:
            raise ValueError("--evaluate-checkpoint no admite --skip-test-evaluation")
        checkpoint = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("labels") != label_list or checkpoint.get("feature_shape") != list(EXPECTED_SHAPE):
            raise ValueError("Checkpoint externo incompatible con el manifiesto dinámico 121")
        model.load_state_dict(checkpoint["model_state_dict"])
        with torch.inference_mode():
            test_metrics = run_epoch(model, loaders["test"], criterion, None, len(label_list))
        report = {"kind": TASK, "evaluation_mode": "frozen_checkpoint_external_test", "checkpoint": str(args.evaluate_checkpoint), "counts": {split: len(rows) for split, rows in partitions.items()}, "test_observed_labels": len({row["label_lsm"] for row in partitions["test"]}), "metric_class_count": len(label_list), "test": test_metrics, "training_from_scratch": True, "external_pretrained_weights_loaded": False, "benchmark_210_words_touched": False, "s08_metrics_evaluated": False, "s09_metrics_evaluated": False}
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
    test_metrics = None
    if not args.skip_test_evaluation:
        checkpoint = torch.load(args.out / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        with torch.inference_mode():
            test_metrics = run_epoch(model, loaders["test"], criterion, None, len(label_list))
    report = {"kind": TASK, "counts": {split: len(rows) for split, rows in partitions.items()}, "signers": {split: sorted({row["participant_id"] for row in rows}) for split, rows in partitions.items()}, "labels": label_list, "parameters": parameter_count(model), "best_validation_macro_f1": best_score, "test": test_metrics, "test_evaluated": not args.skip_test_evaluation, "history": history, "training_from_scratch": True, "external_pretrained_weights_loaded": False, "benchmark_210_words_touched": False, "s08_metrics_evaluated": False, "s09_metrics_evaluated": False}
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"done": True, "best_validation_macro_f1": best_score, "test_evaluated": report["test_evaluated"]}), flush=True)


if __name__ == "__main__":
    main()