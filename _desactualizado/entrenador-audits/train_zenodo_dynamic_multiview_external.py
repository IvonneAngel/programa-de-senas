"""Entrena desde cero un control externo multivista de seis letras dinámicas LSM."""
from __future__ import annotations

import argparse
import csv
import hashlib
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


TASK = "zenodo_dynamic_lsm_multiview_external"
EXPECTED_SHAPE = (30, 126)
EXPECTED_LABELS = ("J", "K", "Q", "X", "Z", "Ñ")
EXPECTED_COUNTS = {"train": 960, "validation": 120, "test": 120}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def macro_f1(predictions: np.ndarray, targets: np.ndarray, classes: int) -> float:
    scores: list[float] = []
    for label in range(classes):
        tp = int(np.sum((predictions == label) & (targets == label)))
        fp = int(np.sum((predictions == label) & (targets != label)))
        fn = int(np.sum((predictions != label) & (targets == label)))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    return float(np.mean(scores))


def load_partitions(manifest: Path) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "pair_id", "participant_id", "letter_lsm", "view", "split_external", "feature_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Manifiesto multivista sin columnas requeridas")
    labels = sorted({row["letter_lsm"] for row in rows})
    if tuple(labels) != EXPECTED_LABELS:
        raise ValueError(f"Etiquetas inesperadas: {labels}")
    partitions = {split: [row for row in rows if row["split_external"] == split] for split in EXPECTED_COUNTS}
    counts = {split: len(values) for split, values in partitions.items()}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Conteos multivista inesperados: {counts}")
    signer_sets = {split: {row["participant_id"] for row in values} for split, values in partitions.items()}
    if any(signer_sets[left] & signer_sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Fuga de firmantes entre particiones multivista")
    for split, values in partitions.items():
        pairs: dict[str, set[str]] = {}
        for row in values:
            pairs.setdefault(row["pair_id"], set()).add(row["view"])
        if not pairs or any(views != {"frontal", "profile"} for views in pairs.values()):
            raise ValueError(f"Pares multivista incompletos en {split}")
        if {row["letter_lsm"] for row in values} != set(labels):
            raise ValueError(f"Cobertura de letras incompleta en {split}")
    return partitions, labels


@dataclass(frozen=True)
class Record:
    feature_path: str
    label: int
    view: str


class LandmarkDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(self, rows: list[dict[str, str]], labels: dict[str, int], frontal_cache: Path, profile_cache: Path) -> None:
        self.records = [Record(row["feature_path"], labels[row["letter_lsm"]], row["view"]) for row in rows]
        self.roots = {"frontal": frontal_cache, "profile": profile_cache}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        values = np.load(self.roots[record.view] / record.feature_path, allow_pickle=False)
        if values.shape != EXPECTED_SHAPE or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"Landmarks multivista inválidos: {record.view}/{record.feature_path}")
        return torch.from_numpy(values), torch.tensor(record.label, dtype=torch.long), record.view


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer | None, classes: int) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    total_loss, total = 0.0, 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    views: list[str] = []
    for features, labels, batch_views in loader:
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
        views.extend(batch_views)
    predicted, expected = np.concatenate(predictions), np.concatenate(targets)
    metric = {"loss": total_loss / total, "accuracy": float(np.mean(predicted == expected)), "macro_f1": macro_f1(predicted, expected, classes)}
    by_view = {}
    view_array = np.asarray(views)
    for view in ("frontal", "profile"):
        mask = view_array == view
        by_view[view] = {"samples": int(mask.sum()), "accuracy": float(np.mean(predicted[mask] == expected[mask])), "macro_f1": macro_f1(predicted[mask], expected[mask], classes)}
    return {"aggregate": metric, "by_view": by_view}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frontal-cache", type=Path, required=True)
    parser.add_argument("--profile-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument("--evaluate-checkpoint", type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.lr <= 0:
        raise ValueError("Hiperparámetros inválidos")
    if args.evaluate_checkpoint is not None and args.skip_test_evaluation:
        raise ValueError("--evaluate-checkpoint no admite --skip-test-evaluation")
    seed_everything(args.seed)
    partitions, label_list = load_partitions(args.manifest)
    labels = {label: index for index, label in enumerate(label_list)}
    datasets = {split: LandmarkDataset(rows, labels, args.frontal_cache, args.profile_cache) for split, rows in partitions.items()}
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"), num_workers=0) for split, dataset in datasets.items()}
    model = TemporalTCN(feature_dim=126, classes=len(label_list), frames=30)
    criterion = nn.CrossEntropyLoss()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.evaluate_checkpoint is not None:
        checkpoint = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("labels") != label_list or checkpoint.get("feature_shape") != list(EXPECTED_SHAPE):
            raise ValueError("Checkpoint multivista incompatible")
        model.load_state_dict(checkpoint["model_state_dict"])
        with torch.inference_mode():
            test = run_epoch(model, loaders["test"], criterion, None, len(label_list))
        report = {"kind": TASK, "evaluation_mode": "frozen_checkpoint_official_test_once", "checkpoint_sha256": sha256(args.evaluate_checkpoint), "counts": EXPECTED_COUNTS, "test": test, "training_from_scratch": True, "external_pretrained_weights_loaded": False, "view_fusion_used": False, "benchmark_210_words_touched": False, "s08_read": False, "s09_read": False}
        (args.out / "external_test_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"done": True, "mode": report["evaluation_mode"], "test": test}), flush=True)
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, stale, history, started = -1.0, 0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, loaders["train"], criterion, optimizer, len(label_list))
        with torch.inference_mode():
            validation = run_epoch(model, loaders["validation"], criterion, None, len(label_list))
        record = {"epoch": epoch, "train": train, "validation": validation, "elapsed_seconds": time.time() - started}
        history.append(record)
        print(json.dumps(record), flush=True)
        score = float(validation["aggregate"]["macro_f1"])
        if score > best:
            best, stale = score, 0
            torch.save({"model_state_dict": model.state_dict(), "labels": label_list, "feature_shape": list(EXPECTED_SHAPE), "seed": args.seed, "training_from_scratch": True, "external_pretrained_weights_loaded": False, "view_fusion_used": False}, args.out / "best.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break
    report = {"kind": TASK, "counts": EXPECTED_COUNTS, "labels": label_list, "parameters": parameter_count(model), "best_validation_macro_f1": best, "test_evaluated": False, "history": history, "training_from_scratch": True, "external_pretrained_weights_loaded": False, "view_fusion_used": False, "benchmark_210_words_touched": False, "s08_read": False, "s09_read": False}
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"done": True, "best_validation_macro_f1": best, "test_evaluated": False}), flush=True)


if __name__ == "__main__":
    main()