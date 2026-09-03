"""Transferencia selectiva ITESO→GJS bone; S09 no se carga ni evalúa."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from lsm.models.tcn import TemporalTCN, parameter_count
from train_successor_multiview_js_consistency import (
    GJS_LAMBDA,
    LR,
    TOTAL_EPOCHS,
    WEIGHT_DECAY,
    AlignedMultiViewDataset,
    VIEW_SPECS,
    aligned_split,
    generalized_js,
    load_rows,
    manifest_sha256,
    run_epoch,
    seed_everything,
)


TRANSFER_PREFIXES = ("stem.", "blocks.")
SHALLOW_TRANSFER_PREFIXES = ("stem.", "blocks.0.", "blocks.1.")
SOURCE_PROTOCOLS = {"iteso_single": "deterministic_similarity_grouped_clip_unknown_signer", "iteso_zenodo_multitask": "iteso_zenodo_multitask_bone126_from_scratch", "iteso_zenodo_dynamic_static_multitask": "iteso_zenodo_dynamic_static_multitask_bone126_from_scratch", "iteso_zenodo_dynamic121_multitask": "iteso_zenodo_dynamic121_multitask_bone126_from_scratch"}


def checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transfer_bone_extractor(target: TemporalTCN, checkpoint: dict, prefixes: tuple[str, ...] = TRANSFER_PREFIXES) -> list[str]:
    source_state = checkpoint.get("encoder_state_dict", checkpoint.get("model_state_dict"))
    if not isinstance(source_state, dict):
        raise ValueError("Checkpoint ITESO sin model_state_dict")
    target_state = target.state_dict()
    expected = {name for name in target_state if name.startswith(prefixes)}
    selected = {name: value for name, value in source_state.items() if name in expected and target_state[name].shape == value.shape}
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        raise ValueError(f"Checkpoint ITESO incompatible; faltan capas extractor: {missing}")
    target.load_state_dict(selected, strict=False)
    return sorted(selected)


def verify_source_report(path: Path, source_kind: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("split_protocol") != SOURCE_PROTOCOLS[source_kind]:
        raise ValueError("El checkpoint fuente no corresponde al protocolo preregistrado")
    if report.get("external_pretrained_weights_loaded") is not False or report.get("training_from_scratch") is not True:
        raise ValueError("La fuente ITESO debe declarar entrenamiento desde cero sin pesos externos")
    if source_kind == "iteso_single" and report.get("test") not in (None,):
        raise ValueError("La transferencia no acepta un reporte ITESO con test integrado")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name, _ in VIEW_SPECS:
        parser.add_argument(f"--{name}-manifest", type=Path, required=True)
        parser.add_argument(f"--{name}-cache-root", type=Path, required=True)
    parser.add_argument("--iteso-checkpoint", type=Path, required=True)
    parser.add_argument("--iteso-train-report", type=Path, required=True)
    parser.add_argument("--source-kind", choices=tuple(SOURCE_PROTOCOLS), default="iteso_single")
    parser.add_argument("--transfer-depth", choices=("all", "shallow"), default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-test-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.skip_test_evaluation:
        raise ValueError("La transferencia exige --skip-test-evaluation para proteger S09")
    if args.epochs != TOTAL_EPOCHS:
        raise ValueError("Las 40 épocas son parte del protocolo preregistrado")
    source_report = verify_source_report(args.iteso_train_report, args.source_kind)
    seed_everything(args.seed)
    manifests = {name: getattr(args, f"{name}_manifest") for name, _ in VIEW_SPECS}
    roots = {name: getattr(args, f"{name}_cache_root") for name, _ in VIEW_SPECS}
    rows_by_view = {name: load_rows(path) for name, path in manifests.items()}
    train_rows, validation_rows = aligned_split(rows_by_view, "train"), aligned_split(rows_by_view, "validation")
    labels = {label: index for index, label in enumerate(sorted({row[0]["label_lsm"] for row in train_rows}))}
    if set(labels) != {row[0]["label_lsm"] for row in validation_rows}:
        raise ValueError("S08 debe conservar las 210 etiquetas")
    train_dataset = AlignedMultiViewDataset(train_rows, roots, labels)
    validation_dataset = AlignedMultiViewDataset(validation_rows, roots, labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    models = {name: TemporalTCN(feature_dim=dimensions, classes=210, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20).to(device) for name, dimensions in VIEW_SPECS}
    expected_counts = {"bone": 158_994, "cov": 167_058, "code": 171_282}
    if {name: parameter_count(model) for name, model in models.items()} != expected_counts:
        raise AssertionError("Presupuesto GJS inesperado")
    source_checkpoint = torch.load(args.iteso_checkpoint, map_location="cpu", weights_only=False)
    transfer_prefixes = TRANSFER_PREFIXES if args.transfer_depth == "all" else SHALLOW_TRANSFER_PREFIXES
    transferred_keys = transfer_bone_extractor(models["bone"], source_checkpoint, prefixes=transfer_prefixes)
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
        record = {"epoch": epoch, "train_loss": train_metrics.loss, "train_lexical_loss": train_metrics.lexical_loss, "train_gjs_loss": train_metrics.gjs_loss, "train_macro_f1_uniform_fusion": train_metrics.macro_f1, "validation_loss": validation_metrics.loss, "validation_macro_f1_uniform_fusion": validation_metrics.macro_f1, "validation_gjs_not_computed": True}
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
    report = {"kind": "successor_gjs_multisource_shallow_transfer" if args.transfer_depth == "shallow" else "successor_gjs_iteso_bone_transfer", "seed": args.seed, "gjs_lambda": GJS_LAMBDA, "views": dict(VIEW_SPECS), "fusion": "mean_of_three_logits_equal_weights", "training_performed": True, "train_samples": len(train_dataset), "validation_samples": len(validation_dataset), "test_samples_closed": 210, "s08_gjs_used": False, "s09_read": False, "s09_evaluated": False, "best_validation_macro_f1": best_f1, "best_epoch": best_epoch, "parameter_count_by_view": expected_counts, "total_parameter_count": sum(expected_counts.values()), "manifest_sha256": {name: manifest_sha256(path) for name, path in manifests.items()}, "iteso_source": {"source_kind": args.source_kind, "transfer_depth": args.transfer_depth, "checkpoint_sha256": checkpoint_sha256(args.iteso_checkpoint), "training_report_sha256": checkpoint_sha256(args.iteso_train_report), "source_protocol": source_report["split_protocol"], "source_test_metrics_used": False, "transferred_keys": transferred_keys, "classifier_transferred": False}, "history": history, "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform()}, "elapsed_seconds": time.time() - started}
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_validation_macro_f1": best_f1, "best_epoch": best_epoch, "s09_evaluated": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()