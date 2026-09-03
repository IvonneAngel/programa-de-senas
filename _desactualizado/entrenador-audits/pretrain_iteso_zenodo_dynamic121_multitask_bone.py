"""Preentrenamiento from-scratch ITESO + Zenodo-6 + Zenodo-121 para GJS."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lsm.models.tcn import TemporalTCN


EPOCHS = 40
LR = 0.002
WEIGHT_DECAY = 0.0001
PROTOCOL = "iteso_zenodo_dynamic121_multitask_bone126_from_scratch"
SOURCE121_TRAIN_SIGNERS = frozenset(f"S{value:02d}" for value in range(1, 9))


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True, warn_only=True)


class CachedRows(Dataset):
    def __init__(self, rows: list[dict[str, str]], label_field: str, labels: dict[str, int], root: Path) -> None:
        if not rows: raise ValueError("Dataset fuente vacío")
        self.rows, self.label_field, self.labels, self.root = rows, label_field, labels, root

    def __len__(self) -> int: return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]; values = np.load(self.root / row["feature_path"], allow_pickle=False).astype(np.float32, copy=False)
        if values.shape != (30, 126) or not np.isfinite(values).all(): raise ValueError(f"Feature fuente inválido: {row['feature_path']}")
        return torch.from_numpy(values), torch.tensor(self.labels[row[self.label_field]], dtype=torch.long)


class ThreeDynamicSourceMultiTaskBone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = TemporalTCN(feature_dim=126, classes=41, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20)
        self.dynamic6_head = TemporalTCN(feature_dim=126, classes=6, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20).head
        self.dynamic121_head = TemporalTCN(feature_dim=126, classes=121, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20).head

    def iteso(self, values: torch.Tensor) -> torch.Tensor: return self.encoder(values)
    def dynamic6(self, values: torch.Tensor) -> torch.Tensor: return self.dynamic6_head(self.encoder.forward_features(values))
    def dynamic121(self, values: torch.Tensor) -> torch.Tensor: return self.dynamic121_head(self.encoder.forward_features(values))


def load_iteso(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, int]]:
    with path.open(encoding="utf-8", newline="") as handle: rows = [row for row in csv.DictReader(handle) if row.get("feature_status") == "ok"]
    train, validation = [row for row in rows if row["split_model"] == "train"], [row for row in rows if row["split_model"] == "validation"]
    labels = {label: index for index, label in enumerate(sorted({row["label_lsm"] for row in train}))}
    if len(train) != 1684 or len(validation) != 203 or len(labels) != 41 or set(labels) != {row["label_lsm"] for row in validation}: raise ValueError("Manifiesto ITESO inválido")
    return train, validation, labels


def load_dynamic6(path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    with path.open(encoding="utf-8", newline="") as handle: rows = [row for row in csv.DictReader(handle) if row.get("feature_status") == "ok"]
    labels = {label: index for index, label in enumerate(sorted({row["letter_lsm"] for row in rows}))}
    if len(rows) != 600 or len(labels) != 6: raise ValueError("Manifiesto Zenodo dinámico de seis letras inválido")
    return rows, labels


def load_dynamic121_train(path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    with path.open(encoding="utf-8", newline="") as handle: rows = [row for row in csv.DictReader(handle) if row.get("feature_status") == "ok"]
    train = [row for row in rows if row.get("split_external") == "train"]
    labels = {label: index for index, label in enumerate(sorted({row["label_lsm"] for row in train}))}
    if len(train) != 957 or len(labels) != 121 or {row["participant_id"] for row in train} != SOURCE121_TRAIN_SIGNERS:
        raise ValueError("Fuente Zenodo dinámica 121 train inválida o con fuga de participantes")
    if any(row.get("participant_id") not in SOURCE121_TRAIN_SIGNERS for row in train): raise ValueError("La fuente 121 train contiene firmante no autorizado")
    return train, labels


def evaluate_iteso(model: ThreeDynamicSourceMultiTaskBone, loader: DataLoader, device: torch.device) -> float:
    model.eval(); predictions: list[np.ndarray] = []; targets_out: list[np.ndarray] = []
    with torch.inference_mode():
        for values, targets in loader:
            predictions.append(model.iteso(values.to(device)).argmax(dim=1).cpu().numpy()); targets_out.append(targets.numpy())
    return float(f1_score(np.concatenate(targets_out), np.concatenate(predictions), average="macro", zero_division=0))


def cyclic_next(iterator: object, loader: DataLoader) -> tuple[object, tuple[torch.Tensor, torch.Tensor]]:
    try: return iterator, next(iterator)  # type: ignore[arg-type]
    except StopIteration: iterator = iter(loader); return iterator, next(iterator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteso-manifest", type=Path, required=True); parser.add_argument("--iteso-cache-root", type=Path, required=True)
    parser.add_argument("--dynamic6-manifest", type=Path, required=True); parser.add_argument("--dynamic6-cache-root", type=Path, required=True)
    parser.add_argument("--dynamic121-manifest", type=Path, required=True); parser.add_argument("--dynamic121-cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs != EPOCHS: raise ValueError("40 épocas son parte de la prerregistración")
    seed_everything(args.seed)
    iteso_train_rows, iteso_validation_rows, iteso_labels = load_iteso(args.iteso_manifest)
    dynamic6_rows, dynamic6_labels = load_dynamic6(args.dynamic6_manifest)
    dynamic121_rows, dynamic121_labels = load_dynamic121_train(args.dynamic121_manifest)
    iteso_train = CachedRows(iteso_train_rows, "label_lsm", iteso_labels, args.iteso_cache_root)
    iteso_validation = CachedRows(iteso_validation_rows, "label_lsm", iteso_labels, args.iteso_cache_root)
    dynamic6_train = CachedRows(dynamic6_rows, "letter_lsm", dynamic6_labels, args.dynamic6_cache_root)
    dynamic121_train = CachedRows(dynamic121_rows, "label_lsm", dynamic121_labels, args.dynamic121_cache_root)
    iteso_loader = DataLoader(iteso_train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    dynamic6_loader = DataLoader(dynamic6_train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed + 1), num_workers=0)
    dynamic121_loader = DataLoader(dynamic121_train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed + 2), num_workers=0)
    validation_loader = DataLoader(iteso_validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device); model = ThreeDynamicSourceMultiTaskBone().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY); criterion = nn.CrossEntropyLoss(); args.out.mkdir(parents=True, exist_ok=True)
    best, best_epoch, history = -1.0, 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train(); dynamic6_iter, dynamic121_iter = iter(dynamic6_loader), iter(dynamic121_loader); losses = {"iteso": 0.0, "dynamic6": 0.0, "dynamic121": 0.0}; steps = 0
        for iteso_values, iteso_targets in iteso_loader:
            dynamic6_iter, (dynamic6_values, dynamic6_targets) = cyclic_next(dynamic6_iter, dynamic6_loader); dynamic121_iter, (dynamic121_values, dynamic121_targets) = cyclic_next(dynamic121_iter, dynamic121_loader)
            optimizer.zero_grad(set_to_none=True)
            iteso_loss = criterion(model.iteso(iteso_values.to(device)), iteso_targets.to(device)); dynamic6_loss = criterion(model.dynamic6(dynamic6_values.to(device)), dynamic6_targets.to(device)); dynamic121_loss = criterion(model.dynamic121(dynamic121_values.to(device)), dynamic121_targets.to(device)); loss = iteso_loss + dynamic6_loss + dynamic121_loss
            if not torch.isfinite(loss): raise FloatingPointError("Pérdida multitarea dinámica no finita")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step(); losses["iteso"] += float(iteso_loss.detach()); losses["dynamic6"] += float(dynamic6_loss.detach()); losses["dynamic121"] += float(dynamic121_loss.detach()); steps += 1
        validation_f1 = evaluate_iteso(model, validation_loader, device); record = {"epoch": epoch, "iteso_train_loss": losses["iteso"] / steps, "dynamic6_train_loss": losses["dynamic6"] / steps, "dynamic121_train_loss": losses["dynamic121"] / steps, "iteso_validation_macro_f1": validation_f1}; history.append(record); print(json.dumps(record), flush=True)
        if validation_f1 > best:
            best, best_epoch = validation_f1, epoch
            torch.save({"encoder_state_dict": model.encoder.state_dict(), "multitask_state_dict": model.state_dict(), "iteso_labels": iteso_labels, "dynamic6_labels": dynamic6_labels, "dynamic121_labels": dynamic121_labels, "epoch": epoch, "training_from_scratch": True, "external_pretrained_weights_loaded": False}, args.out / "best.pt")
    report = {"kind": "iteso_zenodo_dynamic121_multitask_bone_pretrain", "split_protocol": PROTOCOL, "seed": args.seed, "training_from_scratch": True, "external_pretrained_weights_loaded": False, "iteso_train_samples": len(iteso_train), "iteso_validation_samples": len(iteso_validation), "dynamic6_train_samples": len(dynamic6_train), "dynamic6_test_metrics_used": False, "dynamic121_train_samples": len(dynamic121_train), "dynamic121_train_signers": sorted(SOURCE121_TRAIN_SIGNERS), "dynamic121_validation_or_test_metrics_used": False, "best_iteso_validation_macro_f1": best, "best_epoch": best_epoch, "history": history, "benchmark_210_words_touched": False, "s08_read": False, "s09_read": False}
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps({"best_iteso_validation_macro_f1": best, "best_epoch": best_epoch, "s09_read": False}, ensure_ascii=False))


if __name__ == "__main__": main()