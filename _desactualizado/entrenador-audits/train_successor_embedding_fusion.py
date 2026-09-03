"""Fusión end-to-end de embeddings bone/cov/code; S09 queda cerrado."""
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
EPOCHS = 40
FUSION_DIM = 128


class EmbeddingFusionHead(nn.Module):
    """Concatena tres embeddings de 128 y predice 210 clases sin pesos de vista."""

    def __init__(self, classes: int = 210, dropout: float = 0.20):
        super().__init__()
        self.normalization = nn.LayerNorm(3 * FUSION_DIM)
        self.hidden = nn.Linear(3 * FUSION_DIM, FUSION_DIM)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(FUSION_DIM, classes)

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        if len(embeddings) != 3 or any(value.ndim != 2 or value.shape[1] != FUSION_DIM for value in embeddings):
            raise ValueError("La fusión exige exactamente tres embeddings (batch,128)")
        batch = embeddings[0].shape[0]
        if any(value.shape[0] != batch for value in embeddings):
            raise ValueError("Los embeddings deben compartir batch")
        return self.output(self.dropout(torch.nn.functional.gelu(self.hidden(self.normalization(torch.cat(embeddings, dim=1))))))


def fusion_loss(view_logits: list[torch.Tensor], fused_logits: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    if len(view_logits) != 3:
        raise ValueError("La pérdida de fusión requiere tres vistas")
    individual = sum(criterion(logits, targets) for logits in view_logits) / 3.0
    fused = criterion(fused_logits, targets)
    total = individual + fused
    if not torch.isfinite(total):
        raise FloatingPointError("Pérdida de fusión no finita")
    return total, fused


def run_epoch(models: dict[str, TemporalTCN], fusion_head: EmbeddingFusionHead, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizers: list[torch.optim.Optimizer] | None = None) -> dict[str, float]:
    training = optimizers is not None
    for model in models.values():
        model.train(training)
    fusion_head.train(training)
    total_loss, total_head_ce, count = 0.0, 0.0, 0
    all_predictions, all_targets = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for bone, cov, code, targets in loader:
            features = {"bone": bone.to(device), "cov": cov.to(device), "code": code.to(device)}
            targets = targets.to(device)
            if training:
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
            outputs = [models[name].forward_with_embedding(features[name]) for name, _ in VIEW_SPECS]
            logits_by_view = [output[0] for output in outputs]
            fused_logits = fusion_head([output[1] for output in outputs])
            if training:
                total, head_ce = fusion_loss(logits_by_view, fused_logits, targets, criterion)
                total.backward()
                for model in models.values():
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(fusion_head.parameters(), max_norm=1.0)
                for optimizer in optimizers:
                    optimizer.step()
                total_loss += float(total.detach()) * targets.size(0)
                total_head_ce += float(head_ce.detach()) * targets.size(0)
            else:
# Validación: solo CE del logit fusionado; la pérdida train individual
                head_ce = criterion(fused_logits, targets)
                total_head_ce += float(head_ce.detach()) * targets.size(0)
            count += targets.size(0)
            all_predictions.append(fused_logits.argmax(dim=1).detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
    return {"train_loss": total_loss / count if training else 0.0, "head_ce": total_head_ce / count, "macro_f1": float(f1_score(np.concatenate(all_targets), np.concatenate(all_predictions), average="macro", zero_division=0))}


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
        raise ValueError("La fusión de embeddings exige --skip-test-evaluation para proteger S09")
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
    expected_views = {"bone": 158_994, "cov": 167_058, "code": 171_282}
    if {name: parameter_count(model) for name, model in models.items()} != expected_views:
        raise AssertionError("Presupuesto TCN de fusión inesperado")
    fusion_head = EmbeddingFusionHead().to(device)
    if parameter_count(fusion_head) != 77_138:
        raise AssertionError("Presupuesto de cabeza de fusión inesperado")
    optimizers = [torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001) for model in models.values()] + [torch.optim.AdamW(fusion_head.parameters(), lr=0.002, weight_decay=0.0001)]
    criterion = nn.CrossEntropyLoss()
    args.out.mkdir(parents=True, exist_ok=True)
    best, best_epoch, stale, best_state, history = -1.0, 0, 0, None, []
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        train = run_epoch(models, fusion_head, train_loader, criterion, device, optimizers)
        validation = run_epoch(models, fusion_head, validation_loader, criterion, device)
        record = {"epoch": epoch, "train_loss": train["train_loss"], "train_fusion_head_ce": train["head_ce"], "train_macro_f1_fused_head": train["macro_f1"], "validation_fusion_head_ce": validation["head_ce"], "validation_macro_f1_fused_head": validation["macro_f1"], "validation_training_loss_not_computed": True}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["macro_f1"] > best:
            best, best_epoch, stale = validation["macro_f1"], epoch, 0
            best_state = {"models": copy.deepcopy({name: model.state_dict() for name, model in models.items()}), "fusion_head": copy.deepcopy(fusion_head.state_dict())}
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise AssertionError("No se encontró checkpoint de fusión")
    for name, model in models.items():
        torch.save({"model_state_dict": best_state["models"][name], "labels": labels, "view": name, "epoch": best_epoch}, args.out / f"best_{name}.pt")
    torch.save({"state_dict": best_state["fusion_head"], "classes": 210, "epoch": best_epoch}, args.out / "best_fusion_head.pt")
    report = {"kind": "embedding_fusion_head", "seed": args.seed, "formula": "CE(fusion_head(embeddings))+mean(CE(view_i))", "gjs_used": False, "s09_read": False, "s09_evaluated": False, "train_samples": len(train_triples), "validation_samples": len(validation_triples), "best_validation_macro_f1": best, "best_epoch": best_epoch, "parameter_count_by_view": expected_views, "fusion_head_parameter_count": 77_138, "total_parameter_count": sum(expected_views.values()) + 77_138, "history": history, "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform()}, "elapsed_seconds": time.time() - started}
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_validation_macro_f1": best, "best_epoch": best_epoch, "s09_evaluated": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()