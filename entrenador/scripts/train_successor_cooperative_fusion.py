"""Entrena tres vistas con supervisión directa de su promedio uniforme de logits."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

import train_successor_multiview_js_consistency as common
from lsm.models.tcn import TemporalTCN, parameter_count


TOTAL_EPOCHS = 40
VIEW_SPECS = common.VIEW_SPECS


def cooperative_loss(logits_by_view: list[torch.Tensor], targets: torch.Tensor, criterion: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    if len(logits_by_view) != 3 or targets.ndim != 1:
        raise ValueError("La pérdida cooperativa exige tres logits y objetivos unidimensionales")
    fused = common.uniform_logits(logits_by_view)
    individual = sum(criterion(logits, targets) for logits in logits_by_view)
    cooperative = criterion(fused, targets)
    total = individual + cooperative
    if not torch.isfinite(total):
        raise FloatingPointError("Pérdida cooperativa no finita")
    return total, cooperative


def run_epoch(models: dict[str, TemporalTCN], loader: DataLoader, criterion: nn.Module, device: torch.device, optimizers: dict[str, torch.optim.Optimizer] | None = None) -> dict[str, float]:
    training = optimizers is not None
    for model in models.values():
        model.train(training)
    total_loss, total_cooperative, count = 0.0, 0.0, 0
    predictions, all_targets = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for bone, cov, code, targets in loader:
            features = {"bone": bone.to(device), "cov": cov.to(device), "code": code.to(device)}
            targets = targets.to(device)
            if training:
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
            logits_by_view = [models[name](features[name]) for name, _ in VIEW_SPECS]
            total, cooperative = cooperative_loss(logits_by_view, targets, criterion) if training else (sum(criterion(logits, targets) for logits in logits_by_view), torch.zeros((), device=device))
            if training:
                total.backward()
                for model, optimizer in zip(models.values(), optimizers.values(), strict=True):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            total_loss += float(total.detach()) * targets.size(0)
            total_cooperative += float(cooperative.detach()) * targets.size(0)
            count += targets.size(0)
            predictions.append(common.uniform_logits(logits_by_view).argmax(dim=1).detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
    return {"loss": total_loss / count, "cooperative_loss": total_cooperative / count, "macro_f1": float(f1_score(np.concatenate(all_targets), np.concatenate(predictions), average="macro", zero_division=0))}


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
    args = parser.parse_args()
    if not args.skip_test_evaluation:
        raise ValueError("La candidata cooperativa exige --skip-test-evaluation para proteger S09")
    if args.epochs != TOTAL_EPOCHS:
        raise ValueError("Las 40 épocas son parte del protocolo preregistrado")
    common.seed_everything(args.seed)
    manifests = {name: getattr(args, f"{name}_manifest") for name, _ in VIEW_SPECS}
    roots = {name: getattr(args, f"{name}_cache_root") for name, _ in VIEW_SPECS}
    rows = {name: common.load_rows(path) for name, path in manifests.items()}
    train_triples, validation_triples = common.aligned_split(rows, "train"), common.aligned_split(rows, "validation")
    labels = {label: index for index, label in enumerate(sorted({triple[0]["label_lsm"] for triple in train_triples}))}
    if set(labels) != {triple[0]["label_lsm"] for triple in validation_triples}:
        raise ValueError("S08 debe tener las mismas 210 etiquetas")
    train_loader = DataLoader(common.AlignedMultiViewDataset(train_triples, roots, labels), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    validation_loader = DataLoader(common.AlignedMultiViewDataset(validation_triples, roots, labels), batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    models = {name: TemporalTCN(feature_dim=dimensions, classes=210, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20).to(device) for name, dimensions in VIEW_SPECS}
    counts = {"bone": 158_994, "cov": 167_058, "code": 171_282}
    if {name: parameter_count(model) for name, model in models.items()} != counts:
        raise AssertionError("Presupuesto cooperativo inesperado")
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001) for name, model in models.items()}
    criterion = nn.CrossEntropyLoss()
    args.out.mkdir(parents=True, exist_ok=True)
    best, best_epoch, stale, history = -1.0, 0, 0, []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(models, train_loader, criterion, device, optimizers)
        validation = run_epoch(models, validation_loader, criterion, device)
        record = {"epoch": epoch, "train_loss": train["loss"], "train_cooperative_loss": train["cooperative_loss"], "train_macro_f1_uniform_fusion": train["macro_f1"], "validation_loss": validation["loss"], "validation_macro_f1_uniform_fusion": validation["macro_f1"], "validation_cooperative_loss_not_computed": True}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["macro_f1"] > best:
            best, best_epoch, stale = validation["macro_f1"], epoch, 0
            for name, model in models.items():
                torch.save({"model_state_dict": model.state_dict(), "labels": labels, "epoch": epoch, "view": name}, args.out / f"best_{name}.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    report = {"kind": "successor_cooperative_uniform_fusion", "seed": args.seed, "loss": "sum(CE(view_i))+CE(mean_logits)", "gjs_used": False, "weights": [1 / 3, 1 / 3, 1 / 3], "s08_cooperative_loss_used": False, "s09_read": False, "s09_evaluated": False, "train_samples": len(train_triples), "validation_samples": len(validation_triples), "test_samples_closed": 210, "best_validation_macro_f1": best, "best_epoch": best_epoch, "parameter_count_by_view": counts, "total_parameter_count": sum(counts.values()), "manifest_sha256": {name: common.manifest_sha256(path) for name, path in manifests.items()}, "history": history, "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform()}, "elapsed_seconds": time.time() - started}
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_validation_macro_f1": best, "best_epoch": best_epoch, "s09_evaluated": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()