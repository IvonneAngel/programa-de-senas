"""GJS multitarea con mezcla temporal bone intraclase/cross-signer solo en train."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lsm.models.tcn import TemporalTCN, parameter_count
from train_successor_gjs_iteso_bone_transfer import TRANSFER_PREFIXES, checkpoint_sha256, transfer_bone_extractor, verify_source_report
from train_successor_multiview_js_consistency import GJS_LAMBDA, LR, TOTAL_EPOCHS, WEIGHT_DECAY, VIEW_SPECS, AlignedMultiViewDataset, aligned_split, generalized_js, load_rows, manifest_sha256, run_epoch, seed_everything, uniform_logits


MIX_PROBABILITY = 0.50
MIX_ALPHA = 0.20


def build_cross_signer_partner_indices(triples: list[tuple[dict[str, str], dict[str, str], dict[str, str]]]) -> list[list[int]]:
    """Devuelve candidatos train-only con misma glosa y firmante diferente."""
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, rows in enumerate(triples):
        by_label[rows[0]["label_lsm"]].append(index)
    candidates: list[list[int]] = []
    for index, rows in enumerate(triples):
        label, signer = rows[0]["label_lsm"], rows[0]["signer_id"]
        matches = [other for other in by_label[label] if other != index and triples[other][0]["signer_id"] != signer]
        if not matches:
            raise ValueError(f"La ancla {rows[0]['sample_id']} no tiene pareja cross-signer de misma etiqueta")
        candidates.append(matches)
    return candidates


def mix_bone_with_partner(bone: torch.Tensor, partner: torch.Tensor, apply: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    if bone.shape != partner.shape or bone.ndim != 3 or bone.shape[1:] != (30, 126):
        raise ValueError("Mezcla bone con formas incompatibles")
    if apply.shape != (bone.shape[0],) or lambdas.shape != (bone.shape[0],):
        raise ValueError("Máscara o lambda de mezcla incompatibles")
    weights = torch.where(apply, lambdas, torch.ones_like(lambdas)).view(-1, 1, 1)
    return weights * bone + (1.0 - weights) * partner


class TrainOnlyTemporalMixupDataset(Dataset):
    def __init__(self, triples: list[tuple[dict[str, str], dict[str, str], dict[str, str]]], cache_roots: dict[str, Path], labels: dict[str, int], seed: int) -> None:
        self.base = AlignedMultiViewDataset(triples, cache_roots, labels)
        self.triples = triples
        self.partner_indices = build_cross_signer_partner_indices(triples)
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bone, cov, code, label = self.base[index]
        rng = np.random.default_rng(np.uint64(self.seed) * np.uint64(1_000_003) + np.uint64(self.epoch) * np.uint64(10_007) + np.uint64(index))
        partner_index = self.partner_indices[index][int(rng.integers(len(self.partner_indices[index])))]
        partner_bone = self.base[partner_index][0]
        apply = bool(rng.random() < MIX_PROBABILITY)
        if apply:
            sampled = float(rng.beta(MIX_ALPHA, MIX_ALPHA))
            mixing_lambda = max(sampled, 1.0 - sampled)
        else:
            mixing_lambda = 1.0
        return bone, cov, code, partner_bone, torch.tensor(apply, dtype=torch.bool), torch.tensor(mixing_lambda, dtype=torch.float32), label


def run_epoch_mixup(models: dict[str, TemporalTCN], loader: DataLoader, criterion: nn.Module, device: torch.device, optimizers: dict[str, torch.optim.Optimizer] | None = None) -> dict[str, float]:
    training = optimizers is not None
    for model in models.values():
        model.train(training)
    total_loss = total_lexical = total_gjs = 0.0
    count = mixed_count = 0
    predictions: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    for bone, cov, code, partner_bone, apply, lambdas, targets in loader:
        bone, cov, code, partner_bone = bone.to(device), cov.to(device), code.to(device), partner_bone.to(device)
        apply, lambdas, targets = apply.to(device), lambdas.to(device), targets.to(device)
        mixed_bone = mix_bone_with_partner(bone, partner_bone, apply, lambdas) if training else bone
        if training:
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
        logits = [models["bone"](mixed_bone), models["cov"](cov), models["code"](code)]
        lexical = sum(criterion(values, targets) for values in logits)
        gjs = generalized_js(logits) if training else torch.zeros((), device=device)
        loss = lexical + GJS_LAMBDA * gjs
        if not torch.isfinite(loss):
            raise FloatingPointError("Pérdida de mixup temporal no finita")
        if training:
            loss.backward()
            for model, optimizer in zip(models.values(), optimizers.values(), strict=True):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        batch_size = targets.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_lexical += float(lexical.detach()) * batch_size
        total_gjs += float(gjs.detach()) * batch_size
        count += batch_size
        mixed_count += int(apply.sum().detach()) if training else 0
        predictions.append(uniform_logits(logits).argmax(dim=1).detach().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy())
    return {"loss": total_loss / count, "lexical_loss": total_lexical / count, "gjs_loss": total_gjs / count, "macro_f1": float(f1_score(np.concatenate(targets_all), np.concatenate(predictions), average="macro", zero_division=0)), "mixed_fraction": mixed_count / count if training else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name, _ in VIEW_SPECS:
        parser.add_argument(f"--{name}-manifest", type=Path, required=True)
        parser.add_argument(f"--{name}-cache-root", type=Path, required=True)
    parser.add_argument("--iteso-checkpoint", type=Path, required=True)
    parser.add_argument("--iteso-train-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-test-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.skip_test_evaluation:
        raise ValueError("La candidata exige --skip-test-evaluation para proteger S09")
    if args.epochs != TOTAL_EPOCHS:
        raise ValueError("Las 40 épocas son parte del protocolo preregistrado")
    source_report = verify_source_report(args.iteso_train_report, "iteso_zenodo_multitask")
    seed_everything(args.seed)
    manifests = {name: getattr(args, f"{name}_manifest") for name, _ in VIEW_SPECS}
    roots = {name: getattr(args, f"{name}_cache_root") for name, _ in VIEW_SPECS}
    rows_by_view = {name: load_rows(path) for name, path in manifests.items()}
    train_rows, validation_rows = aligned_split(rows_by_view, "train"), aligned_split(rows_by_view, "validation")
    labels = {label: index for index, label in enumerate(sorted({row[0]["label_lsm"] for row in train_rows}))}
    if set(labels) != {row[0]["label_lsm"] for row in validation_rows}:
        raise ValueError("S08 debe conservar las 210 etiquetas")
    train_dataset = TrainOnlyTemporalMixupDataset(train_rows, roots, labels, args.seed)
    validation_dataset = AlignedMultiViewDataset(validation_rows, roots, labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    models = {name: TemporalTCN(feature_dim=dimensions, classes=210, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20).to(device) for name, dimensions in VIEW_SPECS}
    expected_counts = {"bone": 158_994, "cov": 167_058, "code": 171_282}
    if {name: parameter_count(model) for name, model in models.items()} != expected_counts:
        raise AssertionError("Presupuesto GJS inesperado")
    source_checkpoint = torch.load(args.iteso_checkpoint, map_location="cpu", weights_only=False)
    transferred_keys = transfer_bone_extractor(models["bone"], source_checkpoint, prefixes=TRANSFER_PREFIXES)
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) for name, model in models.items()}
    criterion = nn.CrossEntropyLoss()
    args.out.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, stale, history, started = -1.0, 0, 0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        train = run_epoch_mixup(models, train_loader, criterion, device, optimizers)
        with torch.no_grad():
            validation = run_epoch(models, validation_loader, criterion, device)
        record = {"epoch": epoch, "train_loss": train["loss"], "train_lexical_loss": train["lexical_loss"], "train_gjs_loss": train["gjs_loss"], "train_macro_f1_uniform_fusion": train["macro_f1"], "train_mixed_fraction": train["mixed_fraction"], "validation_loss": validation.loss, "validation_macro_f1_uniform_fusion": validation.macro_f1, "validation_gjs_not_computed": True, "validation_mixed_fraction": 0.0}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation.macro_f1 > best_f1:
            best_f1, best_epoch, stale = validation.macro_f1, epoch, 0
            for name, model in models.items():
                torch.save({"model_state_dict": model.state_dict(), "labels": labels, "epoch": epoch, "view": name}, args.out / f"best_{name}.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    report = {"kind": "successor_gjs_multitask_within_class_cross_signer_temporal_mixup", "seed": args.seed, "mixup": {"view": "bone", "probability": MIX_PROBABILITY, "alpha": MIX_ALPHA, "same_label": True, "different_signer": True, "train_only": True, "labels_mixed": False}, "gjs_lambda": GJS_LAMBDA, "views": dict(VIEW_SPECS), "fusion": "mean_of_three_logits_equal_weights", "training_performed": True, "train_samples": len(train_dataset), "validation_samples": len(validation_dataset), "test_samples_closed": 210, "s08_gjs_used": False, "s09_read": False, "s09_evaluated": False, "best_validation_macro_f1": best_f1, "best_epoch": best_epoch, "parameter_count_by_view": expected_counts, "total_parameter_count": sum(expected_counts.values()), "manifest_sha256": {name: manifest_sha256(path) for name, path in manifests.items()}, "iteso_source": {"checkpoint_sha256": checkpoint_sha256(args.iteso_checkpoint), "training_report_sha256": checkpoint_sha256(args.iteso_train_report), "source_protocol": source_report["split_protocol"], "transferred_keys": transferred_keys, "classifier_transferred": False}, "history": history, "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform()}, "elapsed_seconds": time.time() - started}
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_validation_macro_f1": best_f1, "best_epoch": best_epoch, "s09_evaluated": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()