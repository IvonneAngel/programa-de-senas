"""Entrena tres TCN con pooling temporal de covarianza y fusión GJS; S09 cerrado."""
from __future__ import annotations

import argparse
import copy
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
from lsm.models.tcn import TemporalCovariancePoolingTCN, parameter_count


VIEW_SPECS = common.VIEW_SPECS
GJS_LAMBDA = 0.10
EPOCHS = 40


def gjs_loss(logits_by_view: list[torch.Tensor], targets: torch.Tensor, criterion: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    lexical = sum(criterion(logits, targets) for logits in logits_by_view)
    gjs = common.generalized_js(logits_by_view)
    total = lexical + GJS_LAMBDA * gjs
    if not torch.isfinite(total):
        raise FloatingPointError("Pérdida GJS+covarianza no finita")
    return total, gjs


def run_epoch(models: dict[str, TemporalCovariancePoolingTCN], loader: DataLoader, criterion: nn.Module, optimizers: dict[str, torch.optim.Optimizer] | None = None) -> dict[str, float]:
    training = optimizers is not None
    for model in models.values():
        model.train(training)
    total_loss, total_gjs, count = 0.0, 0.0, 0
    predictions, target_values = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for bone, cov, code, targets in loader:
            features = {"bone": bone, "cov": cov, "code": code}
            if training:
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
            logits_by_view = [models[name](features[name]) for name, _ in VIEW_SPECS]
            if training:
                total, gjs = gjs_loss(logits_by_view, targets, criterion)
                total.backward()
                for model, optimizer in zip(models.values(), optimizers.values(), strict=True):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                total_loss += float(total.detach()) * targets.size(0)
                total_gjs += float(gjs.detach()) * targets.size(0)
            count += targets.size(0)
            predictions.append(common.uniform_logits(logits_by_view).argmax(dim=1).detach().cpu().numpy())
            target_values.append(targets.detach().cpu().numpy())
    return {"loss": total_loss / count if training else 0.0, "gjs": total_gjs / count if training else 0.0, "macro_f1": float(f1_score(np.concatenate(target_values), np.concatenate(predictions), average="macro", zero_division=0))}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name, _ in VIEW_SPECS:
        parser.add_argument(f"--{name}-manifest", type=Path, required=True)
        parser.add_argument(f"--{name}-cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.skip_test_evaluation:
        raise ValueError("GJS+covarianza temporal exige --skip-test-evaluation para proteger S09")
    common.seed_everything(args.seed)
    manifests = {name: getattr(args, f"{name}_manifest") for name, _ in VIEW_SPECS}
    roots = {name: getattr(args, f"{name}_cache_root") for name, _ in VIEW_SPECS}
    rows = {name: common.load_rows(path) for name, path in manifests.items()}
    train_triples, validation_triples = common.aligned_split(rows, "train"), common.aligned_split(rows, "validation")
    labels = {label: index for index, label in enumerate(sorted({triple[0]["label_lsm"] for triple in train_triples}))}
    train_loader = DataLoader(common.AlignedMultiViewDataset(train_triples, roots, labels), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    validation_loader = DataLoader(common.AlignedMultiViewDataset(validation_triples, roots, labels), batch_size=args.batch_size, shuffle=False, num_workers=0)
    models = {name: TemporalCovariancePoolingTCN(feature_dim=dimensions, classes=210, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20) for name, dimensions in VIEW_SPECS}
    expected = {"bone": 177_842, "cov": 185_906, "code": 190_130}
    if {name: parameter_count(model) for name, model in models.items()} != expected:
        raise AssertionError("Presupuesto GJS+covarianza temporal inesperado")
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001) for name, model in models.items()}
    criterion = nn.CrossEntropyLoss()
    args.out.mkdir(parents=True, exist_ok=True)
    best, best_epoch, stale, best_state, history = -1.0, 0, 0, None, []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        train = run_epoch(models, train_loader, criterion, optimizers)
        validation = run_epoch(models, validation_loader, criterion)
        record = {"epoch": epoch, "train_loss": train["loss"], "train_gjs_loss": train["gjs"], "train_macro_f1_uniform_fusion": train["macro_f1"], "validation_macro_f1_uniform_fusion": validation["macro_f1"], "validation_gjs_not_computed": True}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["macro_f1"] > best:
            best, best_epoch, stale = validation["macro_f1"], epoch, 0
            best_state = copy.deepcopy({name: model.state_dict() for name, model in models.items()})
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise AssertionError("Sin checkpoint GJS+covarianza temporal")
    for name, model in models.items():
        torch.save({"model_state_dict": best_state[name], "labels": labels, "view": name, "epoch": best_epoch}, args.out / f"best_{name}.pt")
    report = {"kind": "gjs_temporal_covariance_pool", "seed": args.seed, "gjs_lambda": GJS_LAMBDA, "temporal_covariance_channels": 16, "descriptor_dim": 200, "s09_read": False, "s09_evaluated": False, "validation_gjs_computed": False, "best_validation_macro_f1": best, "best_epoch": best_epoch, "parameter_count_by_view": expected, "total_parameter_count": sum(expected.values()), "history": history, "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform()}, "elapsed_seconds": time.time() - started}
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_validation_macro_f1": best, "best_epoch": best_epoch, "s09_evaluated": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()