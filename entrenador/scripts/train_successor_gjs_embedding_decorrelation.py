"""TCN multivista con GJS predictivo y decorrelación cross-view train-only."""
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
from lsm.models.tcn import TemporalTCN, parameter_count


VIEW_SPECS = common.VIEW_SPECS
GJS_LAMBDA = 0.10
DECORRELATION_LAMBDA = 0.10
EMBEDDING_DIM = 128
STANDARDIZATION_EPSILON = 1e-4
EPOCHS = 40


def embedding_decorrelation(embeddings: list[torch.Tensor], epsilon: float = STANDARDIZATION_EPSILON) -> torch.Tensor:
    if len(embeddings) != 3 or epsilon <= 0.0:
        raise ValueError("La decorrelación exige tres embeddings y epsilon positivo")
    if any(value.ndim != 2 or value.shape[1] != EMBEDDING_DIM for value in embeddings):
        raise ValueError("Los embeddings deben tener forma (batch,128)")
    batch = embeddings[0].shape[0]
    if batch < 2 or any(value.shape[0] != batch for value in embeddings):
        raise ValueError("La decorrelación exige batch común de al menos dos ejemplos")
    standardized = [(value - value.mean(dim=0, keepdim=True)) / torch.sqrt(value.var(dim=0, unbiased=False, keepdim=True) + epsilon) for value in embeddings]
    values = []
    for left, right in ((standardized[0], standardized[1]), (standardized[0], standardized[2]), (standardized[1], standardized[2])):
        correlation = left.transpose(0, 1).matmul(right) / float(batch)
        values.append(correlation.square().mean())
    result = sum(values) / 3.0
    if not torch.isfinite(result):
        raise FloatingPointError("Decorrelación no finita")
    return result


def combined_loss(logits_by_view: list[torch.Tensor], embeddings: list[torch.Tensor], targets: torch.Tensor, criterion: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lexical = sum(criterion(logits, targets) for logits in logits_by_view)
    gjs = common.generalized_js(logits_by_view)
    decorrelation = embedding_decorrelation(embeddings)
    total = lexical + GJS_LAMBDA * gjs + DECORRELATION_LAMBDA * decorrelation
    if not torch.isfinite(total):
        raise FloatingPointError("Pérdida GJS+decorrelación no finita")
    return total, gjs, decorrelation


def run_epoch(models: dict[str, TemporalTCN], loader: DataLoader, criterion: nn.Module, device: torch.device, optimizers: dict[str, torch.optim.Optimizer] | None = None) -> dict[str, float]:
    training = optimizers is not None
    for model in models.values():
        model.train(training)
    total_loss, total_gjs, total_decorrelation, count = 0.0, 0.0, 0.0, 0
    all_predictions, all_targets = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for bone, cov, code, targets in loader:
            features = {"bone": bone.to(device), "cov": cov.to(device), "code": code.to(device)}
            targets = targets.to(device)
            if training:
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
            outputs = [models[name].forward_with_embedding(features[name]) for name, _ in VIEW_SPECS]
            logits_by_view, embeddings = [output[0] for output in outputs], [output[1] for output in outputs]
            if training:
                total, gjs, decorrelation = combined_loss(logits_by_view, embeddings, targets, criterion)
                total.backward()
                for model, optimizer in zip(models.values(), optimizers.values(), strict=True):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                total_loss += float(total.detach()) * targets.size(0)
                total_gjs += float(gjs.detach()) * targets.size(0)
                total_decorrelation += float(decorrelation.detach()) * targets.size(0)
            count += targets.size(0)
            all_predictions.append(common.uniform_logits(logits_by_view).argmax(dim=1).detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
    return {"loss": total_loss / count if training else 0.0, "gjs": total_gjs / count if training else 0.0, "decorrelation": total_decorrelation / count if training else 0.0, "macro_f1": float(f1_score(np.concatenate(all_targets), np.concatenate(all_predictions), average="macro", zero_division=0))}


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
        raise ValueError("GJS+decorrelación exige --skip-test-evaluation para proteger S09")
    common.seed_everything(args.seed)
    manifests = {name: getattr(args, f"{name}_manifest") for name, _ in VIEW_SPECS}
    roots = {name: getattr(args, f"{name}_cache_root") for name, _ in VIEW_SPECS}
    rows = {name: common.load_rows(path) for name, path in manifests.items()}
    train_triples, validation_triples = common.aligned_split(rows, "train"), common.aligned_split(rows, "validation")
    labels = {label: index for index, label in enumerate(sorted({triple[0]["label_lsm"] for triple in train_triples}))}
    train_loader = DataLoader(common.AlignedMultiViewDataset(train_triples, roots, labels), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    validation_loader = DataLoader(common.AlignedMultiViewDataset(validation_triples, roots, labels), batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cpu")
    models = {name: TemporalTCN(feature_dim=dimensions, classes=210, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20).to(device) for name, dimensions in VIEW_SPECS}
    expected = {"bone": 158_994, "cov": 167_058, "code": 171_282}
    if {name: parameter_count(model) for name, model in models.items()} != expected:
        raise AssertionError("Presupuesto GJS+decorrelación inesperado")
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001) for name, model in models.items()}
    criterion = nn.CrossEntropyLoss()
    args.out.mkdir(parents=True, exist_ok=True)
    best, best_epoch, stale, best_state, history = -1.0, 0, 0, None, []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        train = run_epoch(models, train_loader, criterion, device, optimizers)
        validation = run_epoch(models, validation_loader, criterion, device)
        record = {"epoch": epoch, "train_loss": train["loss"], "train_gjs_loss": train["gjs"], "train_embedding_decorrelation": train["decorrelation"], "train_macro_f1_uniform_fusion": train["macro_f1"], "validation_macro_f1_uniform_fusion": validation["macro_f1"], "validation_regularizers_not_computed": True}
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
        raise AssertionError("Sin checkpoint GJS+decorrelación")
    for name, model in models.items():
        torch.save({"model_state_dict": best_state[name], "labels": labels, "view": name, "epoch": best_epoch}, args.out / f"best_{name}.pt")
    report = {"kind": "gjs_embedding_decorrelation", "seed": args.seed, "gjs_lambda": GJS_LAMBDA, "decorrelation_lambda": DECORRELATION_LAMBDA, "standardization_epsilon": STANDARDIZATION_EPSILON, "s09_read": False, "s09_evaluated": False, "validation_regularizers_computed": False, "best_validation_macro_f1": best, "best_epoch": best_epoch, "total_parameter_count": sum(expected.values()), "history": history, "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform()}, "elapsed_seconds": time.time() - started}
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_validation_macro_f1": best, "best_epoch": best_epoch, "s09_evaluated": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()