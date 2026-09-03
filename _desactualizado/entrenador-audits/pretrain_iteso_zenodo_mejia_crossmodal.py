"""Preentrenamiento heteromodal desde cero: bone126 y RGB-D201 comparten solo dinámica temporal."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lsm.models.tcn import TCNResidualBlock
from pretrain_iteso_zenodo_multitask_bone import CachedRows, load_iteso, load_zenodo


EPOCHS = 40
LR = 0.002
WEIGHT_DECAY = 0.0001
PROTOCOL = "iteso_zenodo_mejia_crossmodal_shared_temporal_blocks_from_scratch"
MEJIA_SHAPE = (20, 201)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def head(classes: int) -> nn.Sequential:
    return nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 128), nn.GELU(), nn.Dropout(0.20), nn.Linear(128, classes))


class HeteroMultiTaskTemporal(nn.Module):
    """Dos adaptadores de entrada, dinámica TCN y cabezas estrictamente disjuntas."""

    def __init__(self) -> None:
        super().__init__()
        self.bone_stem = nn.Conv1d(126, 64, kernel_size=3, padding=1)
        self.rgbd_stem = nn.Conv1d(201, 64, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[TCNResidualBlock(64, dilation, 0.20) for dilation in (1, 2, 4, 8)])
        self.iteso_head = head(41)
        self.zenodo_head = head(6)
        self.mejia_head = head(30)

    def encode_bone(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (30, 126):
            raise ValueError(f"bone126 inválido: {tuple(values.shape)}")
        return self.blocks(F.gelu(self.bone_stem(values.transpose(1, 2))))

    def encode_rgbd(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != MEJIA_SHAPE:
            raise ValueError(f"rgbd201 inválido: {tuple(values.shape)}")
        return self.blocks(F.gelu(self.rgbd_stem(values.transpose(1, 2))))

    def iteso(self, values: torch.Tensor) -> torch.Tensor:
        return self.iteso_head(self.encode_bone(values))

    def zenodo(self, values: torch.Tensor) -> torch.Tensor:
        return self.zenodo_head(self.encode_bone(values))

    def mejia(self, values: torch.Tensor) -> torch.Tensor:
        return self.mejia_head(self.encode_rgbd(values))

    def bone_transfer_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        allowed = {name: value.detach().cpu() for name, value in state.items() if name.startswith("bone_stem.") or name.startswith("blocks.")}
        if not allowed or any(name.startswith(("rgbd_stem.", "iteso_head.", "zenodo_head.", "mejia_head.")) for name in allowed):
            raise AssertionError("La transferencia debe contener solo bone_stem y bloques")
        return allowed


class MejiaTrainRows(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, str]], labels: dict[str, int], source_root: Path) -> None:
        self.rows, self.labels, self.source_root = rows, labels, source_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        values = np.loadtxt(self.source_root / row["feature_path"], delimiter=",", skiprows=1, dtype=np.float32)[:, 1:]
        if values.shape != MEJIA_SHAPE or not np.isfinite(values).all():
            raise ValueError(f"Keypoints Mejía inválidos: {row['feature_path']}")
        return torch.from_numpy(values), torch.tensor(self.labels[row["label"]], dtype=torch.long)


def load_mejia_train(path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"label", "sample_index", "split_external", "feature_path", "frames", "channels", "participant_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Manifiesto Mejía inválido")
    train = [row for row in rows if row["split_external"] == "train"]
    labels = {label: index for index, label in enumerate(sorted({row["label"] for row in train}))}
    if len(train) != 2040 or len(labels) != 30 or any(row["sample_index"] not in {str(value) for value in range(1, 69)} for row in train):
        raise ValueError("Solo se admiten los 2,040 ejemplos train de Mejía")
    if any(row["participant_id"] != "unavailable_in_release" for row in rows):
        raise ValueError("No se permiten afirmaciones de firmante para Mejía")
    return train, labels


def evaluate_iteso(model: HeteroMultiTaskTemporal, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    predictions: list[np.ndarray] = []
    targets_out: list[np.ndarray] = []
    with torch.inference_mode():
        for values, targets in loader:
            predictions.append(model.iteso(values.to(device)).argmax(dim=1).cpu().numpy())
            targets_out.append(targets.numpy())
    return float(f1_score(np.concatenate(targets_out), np.concatenate(predictions), average="macro", zero_division=0))


def next_batch(iterator: object, loader: DataLoader) -> tuple[object, tuple[torch.Tensor, torch.Tensor]]:
    try:
        return iterator, next(iterator)  # type: ignore[arg-type]
    except StopIteration:
        iterator = iter(loader)
        return iterator, next(iterator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteso-manifest", type=Path, required=True); parser.add_argument("--iteso-cache-root", type=Path, required=True)
    parser.add_argument("--zenodo-manifest", type=Path, required=True); parser.add_argument("--zenodo-cache-root", type=Path, required=True)
    parser.add_argument("--mejia-manifest", type=Path, required=True); parser.add_argument("--mejia-source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs != EPOCHS:
        raise ValueError("40 épocas son parte del protocolo preregistrado")
    seed_everything(args.seed)
    iteso_train_rows, iteso_validation_rows, iteso_labels = load_iteso(args.iteso_manifest)
    zenodo_rows, zenodo_labels = load_zenodo(args.zenodo_manifest)
    mejia_rows, mejia_labels = load_mejia_train(args.mejia_manifest)
    iteso_loader = DataLoader(CachedRows(iteso_train_rows, "label_lsm", iteso_labels, args.iteso_cache_root), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    zenodo_loader = DataLoader(CachedRows(zenodo_rows, "letter_lsm", zenodo_labels, args.zenodo_cache_root), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed + 1), num_workers=0)
    mejia_loader = DataLoader(MejiaTrainRows(mejia_rows, mejia_labels, args.mejia_source_root), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed + 2), num_workers=0)
    validation_loader = DataLoader(CachedRows(iteso_validation_rows, "label_lsm", iteso_labels, args.iteso_cache_root), batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model = HeteroMultiTaskTemporal().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    args.out.mkdir(parents=True, exist_ok=True)
    best, best_epoch, history = -1.0, 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train(); zenodo_iter, mejia_iter = iter(zenodo_loader), iter(mejia_loader); losses = {"iteso": 0.0, "zenodo": 0.0, "mejia": 0.0}; steps = 0
        for iteso_values, iteso_targets in iteso_loader:
            zenodo_iter, (zenodo_values, zenodo_targets) = next_batch(zenodo_iter, zenodo_loader)
            mejia_iter, (mejia_values, mejia_targets) = next_batch(mejia_iter, mejia_loader)
            optimizer.zero_grad(set_to_none=True)
            loss_iteso = criterion(model.iteso(iteso_values.to(device)), iteso_targets.to(device))
            loss_zenodo = criterion(model.zenodo(zenodo_values.to(device)), zenodo_targets.to(device))
            loss_mejia = criterion(model.mejia(mejia_values.to(device)), mejia_targets.to(device))
            loss = loss_iteso + loss_zenodo + loss_mejia
            if not torch.isfinite(loss):
                raise FloatingPointError("Pérdida heteromodal no finita")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
            for name, value in (("iteso", loss_iteso), ("zenodo", loss_zenodo), ("mejia", loss_mejia)):
                losses[name] += float(value.detach())
            steps += 1
        validation_f1 = evaluate_iteso(model, validation_loader, device)
        record = {"epoch": epoch, "iteso_train_loss": losses["iteso"] / steps, "zenodo_train_loss": losses["zenodo"] / steps, "mejia_train_loss": losses["mejia"] / steps, "iteso_validation_macro_f1": validation_f1, "mejia_validation_used": False, "mejia_test_used": False}
        history.append(record); print(json.dumps(record), flush=True)
        if validation_f1 > best:
            best, best_epoch = validation_f1, epoch
            transfer = model.bone_transfer_state_dict()
            torch.save({"kind": PROTOCOL, "full_state_dict": model.state_dict(), "bone_transfer_state_dict": transfer, "iteso_labels": iteso_labels, "zenodo_labels": zenodo_labels, "mejia_labels": mejia_labels, "epoch": epoch, "training_from_scratch": True, "external_pretrained_weights_loaded": False, "mejia_validation_used": False, "mejia_test_used": False, "s08_used": False, "s09_read": False}, args.out / "best.pt")
    report = {"kind": PROTOCOL, "seed": args.seed, "training_from_scratch": True, "external_pretrained_weights_loaded": False, "iteso_train_samples": len(iteso_train_rows), "iteso_validation_samples": len(iteso_validation_rows), "zenodo_train_samples": len(zenodo_rows), "mejia_train_samples": len(mejia_rows), "mejia_validation_used": False, "mejia_test_used": False, "s08_used": False, "s09_read": False, "best_iteso_validation_macro_f1": best, "best_epoch": best_epoch, "mejia_manifest_sha256": hashlib.sha256(args.mejia_manifest.read_bytes()).hexdigest(), "history": history}
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_iteso_validation_macro_f1": best, "best_epoch": best_epoch, "mejia_validation_used": False, "mejia_test_used": False, "s09_read": False}), flush=True)


if __name__ == "__main__":
    main()