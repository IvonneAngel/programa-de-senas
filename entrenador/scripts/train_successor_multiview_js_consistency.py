"""Entrena tres vistas LSM con consistencia GJS exclusiva de S01--S07."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lsm.models.tcn import TemporalTCN, parameter_count


TOTAL_EPOCHS = 40
LR = 0.002
WEIGHT_DECAY = 0.0001
GJS_LAMBDA = 0.10
VIEW_SPECS = (("bone", 126), ("cov", 168), ("code", 190))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["feature_status"] == "ok"]
    counts = {split: sum(row["split_model"] == split for row in rows) for split in ("train", "validation", "test")}
    if len(rows) != 1890 or counts != {"train": 1470, "validation": 210, "test": 210}:
        raise ValueError(f"Manifiesto multivista inválido: {len(rows)} filas / {counts}")
    if len({row["label_lsm"] for row in rows}) != 210:
        raise ValueError("La consistencia multivista requiere exactamente 210 etiquetas")
    return sorted(rows, key=lambda row: row["sample_id"])


def aligned_split(rows_by_view: dict[str, list[dict[str, str]]], split: str) -> list[tuple[dict[str, str], dict[str, str], dict[str, str]]]:
    selected = {name: [row for row in rows if row["split_model"] == split] for name, rows in rows_by_view.items()}
    reference = [(row["sample_id"], row["label_lsm"], row["signer_id"]) for row in selected["bone"]]
    if split == "train" and {signer for _, _, signer in reference} != {f"S{index:02d}" for index in range(1, 8)}:
        raise ValueError("Train debe contener exactamente S01–S07")
    if split == "validation" and {signer for _, _, signer in reference} != {"S08"}:
        raise ValueError("Validation debe contener únicamente S08")
    for name in ("cov", "code"):
        if [(row["sample_id"], row["label_lsm"], row["signer_id"]) for row in selected[name]] != reference:
            raise ValueError(f"La vista {name} no está alineada por sample_id/etiqueta/firmante")
    return list(zip(selected["bone"], selected["cov"], selected["code"], strict=True))


class AlignedMultiViewDataset(Dataset):
    def __init__(self, triples: list[tuple[dict[str, str], dict[str, str], dict[str, str]]], cache_roots: dict[str, Path], labels: dict[str, int]):
        if not triples:
            raise ValueError("La partición multivista no puede estar vacía")
        self.triples = triples
        self.cache_roots = cache_roots
        self.labels = labels

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = self.triples[index]
        values: list[torch.Tensor] = []
        for (name, dimensions), row in zip(VIEW_SPECS, rows, strict=True):
            feature = np.load(self.cache_roots[name] / row["feature_path"], allow_pickle=False).astype(np.float32, copy=False)
            if feature.shape != (30, dimensions) or not np.isfinite(feature).all():
                raise ValueError(f"{name}/{row['sample_id']}: característica inválida {feature.shape}")
            values.append(torch.from_numpy(feature))
        return values[0], values[1], values[2], torch.tensor(self.labels[rows[0]["label_lsm"]], dtype=torch.long)


def generalized_js(logits_by_view: list[torch.Tensor]) -> torch.Tensor:
    if len(logits_by_view) != 3:
        raise ValueError("GJS requiere exactamente tres vistas")
    first = logits_by_view[0]
    if first.ndim != 2 or first.shape[1] != 210 or any(logits.shape != first.shape for logits in logits_by_view):
        raise ValueError("Logits GJS incompatibles con 210 clases")
    log_probabilities = [torch.log_softmax(logits, dim=1) for logits in logits_by_view]
    probabilities = [values.exp() for values in log_probabilities]
    mean_probability = torch.stack(probabilities, dim=0).mean(dim=0)
    return torch.stack([(probability * (log_probability - mean_probability.log())).sum(dim=1).mean() for probability, log_probability in zip(probabilities, log_probabilities, strict=True)]).mean()


def uniform_logits(logits_by_view: list[torch.Tensor]) -> torch.Tensor:
    if len(logits_by_view) != 3 or any(logits.shape != logits_by_view[0].shape for logits in logits_by_view):
        raise ValueError("La fusión uniforme exige tres logits compatibles")
    return torch.stack(logits_by_view, dim=0).mean(dim=0)


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    lexical_loss: float
    gjs_loss: float
    macro_f1: float


def run_epoch(models: dict[str, TemporalTCN], loader: DataLoader, criterion: nn.Module, device: torch.device, optimizers: dict[str, torch.optim.Optimizer] | None = None) -> EpochMetrics:
    training = optimizers is not None
    for model in models.values():
        model.train(training)
    total_loss = 0.0
    total_lexical = 0.0
    total_gjs = 0.0
    count = 0
    prediction_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    for bone, cov, code, targets in loader:
        features = {"bone": bone.to(device), "cov": cov.to(device), "code": code.to(device)}
        targets = targets.to(device)
        if training:
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
        logits_by_view = [models[name](features[name]) for name, _ in VIEW_SPECS]
        lexical_loss = sum(criterion(logits, targets) for logits in logits_by_view)
        gjs_loss = generalized_js(logits_by_view) if training else torch.zeros((), device=device)
        loss = lexical_loss + GJS_LAMBDA * gjs_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Pérdida multivista no finita")
        if training:
            loss.backward()
            for model, optimizer in zip(models.values(), optimizers.values(), strict=True):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total_loss += float(loss.detach()) * targets.size(0)
        total_lexical += float(lexical_loss.detach()) * targets.size(0)
        total_gjs += float(gjs_loss.detach()) * targets.size(0)
        count += targets.size(0)
        prediction_chunks.append(uniform_logits(logits_by_view).argmax(dim=1).detach().cpu().numpy())
        target_chunks.append(targets.detach().cpu().numpy())
    return EpochMetrics(
        loss=total_loss / count,
        lexical_loss=total_lexical / count,
        gjs_loss=total_gjs / count,
        macro_f1=float(f1_score(np.concatenate(target_chunks), np.concatenate(prediction_chunks), average="macro", zero_division=0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    for name, _ in VIEW_SPECS:
        parser.add_argument(f"--{name}-manifest", type=Path, required=True)
        parser.add_argument(f"--{name}-cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument("--bone-transfer-checkpoint", type=Path, help="Checkpoint heteromodal; transfiere exclusivamente bone_stem y bloques")
    args = parser.parse_args()
    if not args.skip_test_evaluation:
        raise ValueError("La candidata multivista exige --skip-test-evaluation para proteger S09")
    if args.epochs != TOTAL_EPOCHS:
        raise ValueError("Las 40 épocas son parte del protocolo preregistrado")
    seed_everything(args.seed)
    manifests = {name: getattr(args, f"{name}_manifest") for name, _ in VIEW_SPECS}
    cache_roots = {name: getattr(args, f"{name}_cache_root") for name, _ in VIEW_SPECS}
    rows_by_view = {name: load_rows(path) for name, path in manifests.items()}
    train_triples = aligned_split(rows_by_view, "train")
    validation_triples = aligned_split(rows_by_view, "validation")
    labels = {label: index for index, label in enumerate(sorted({row[0]["label_lsm"] for row in train_triples}))}
    if set(labels) != {row[0]["label_lsm"] for row in validation_triples}:
        raise ValueError("S08 debe contener las mismas 210 etiquetas sin usar S09")
    train_dataset = AlignedMultiViewDataset(train_triples, cache_roots, labels)
    validation_dataset = AlignedMultiViewDataset(validation_triples, cache_roots, labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    models = {name: TemporalTCN(feature_dim=dimensions, classes=210, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20).to(device) for name, dimensions in VIEW_SPECS}
    expected_counts = {"bone": 158_994, "cov": 167_058, "code": 171_282}
    if {name: parameter_count(model) for name, model in models.items()} != expected_counts:
        raise AssertionError("Presupuesto multivista inesperado")
    transfer = {"enabled": False}
    if args.bone_transfer_checkpoint is not None:
        checkpoint = torch.load(args.bone_transfer_checkpoint, map_location="cpu", weights_only=False)
        source = checkpoint.get("bone_transfer_state_dict")
        if checkpoint.get("kind") != "iteso_zenodo_mejia_crossmodal_shared_temporal_blocks_from_scratch" or not isinstance(source, dict):
            raise ValueError("Checkpoint heteromodal incompatible")
        target = models["bone"].state_dict()
        transferred = {"stem.weight": source.get("bone_stem.weight"), "stem.bias": source.get("bone_stem.bias")}
        transferred.update({name: source.get(name) for name in target if name.startswith("blocks.")})
        if set(transferred) != {name for name in target if name.startswith("stem.") or name.startswith("blocks.")} or any(value is None for value in transferred.values()):
            raise ValueError("Transferencia heteromodal incompleta o fuera de frontera")
        target.update(transferred)
        models["bone"].load_state_dict(target, strict=True)
        transfer = {"enabled": True, "checkpoint": str(args.bone_transfer_checkpoint), "sha256": hashlib.sha256(args.bone_transfer_checkpoint.read_bytes()).hexdigest(), "transferred_prefixes": ["bone_stem→stem", "blocks"]}
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) for name, model in models.items()}
    criterion = nn.CrossEntropyLoss()
    args.out.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, stale = -1.0, 0, 0
    history: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(models, train_loader, criterion, device, optimizers)
        with torch.no_grad():
            validation_metrics = run_epoch(models, validation_loader, criterion, device)
        record = {
            "epoch": epoch, "train_loss": train_metrics.loss, "train_lexical_loss": train_metrics.lexical_loss,
            "train_gjs_loss": train_metrics.gjs_loss, "train_macro_f1_uniform_fusion": train_metrics.macro_f1,
            "validation_loss": validation_metrics.loss, "validation_macro_f1_uniform_fusion": validation_metrics.macro_f1,
            "validation_gjs_not_computed": True,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_metrics.macro_f1 > best_f1:
            best_f1, best_epoch, stale = validation_metrics.macro_f1, epoch, 0
            for name, model in models.items():
                torch.save({"model_state_dict": model.state_dict(), "labels": labels, "epoch": epoch, "view": name}, args.out / f"best_{name}.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    report = {
        "kind": "successor_multiview_js_consistency", "seed": args.seed, "gjs_lambda": GJS_LAMBDA,
        "views": dict(VIEW_SPECS), "fusion": "mean_of_three_logits_equal_weights", "training_performed": True,
        "train_samples": len(train_dataset), "validation_samples": len(validation_dataset), "test_samples_closed": 210,
        "s08_gjs_used": False, "s09_read": False, "s09_evaluated": False, "best_validation_macro_f1": best_f1,
        "best_epoch": best_epoch, "parameter_count_by_view": expected_counts, "total_parameter_count": sum(expected_counts.values()),
        "manifest_sha256": {name: manifest_sha256(path) for name, path in manifests.items()}, "bone_transfer": transfer, "history": history,
        "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform()}, "elapsed_seconds": time.time() - started,
    }
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_validation_macro_f1": best_f1, "best_epoch": best_epoch, "s09_evaluated": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()