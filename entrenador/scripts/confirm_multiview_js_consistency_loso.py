"""Confirmación LOSO S01--S07 de consistencia GJS contra OOF uniforme congelado."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import confirm_uniform_fusion_loso as common
import train_successor_multiview_js_consistency as candidate
from bone_code190 import transform_sequence as code_transform
from lsm.models.tcn import TemporalTCN, parameter_count


class FoldMultiViewDataset(Dataset):
    def __init__(self, bone_rows: list[dict[str, str]], cov_rows: list[dict[str, str]], bone_root: Path, cov_root: Path, labels: dict[str, int], codebook: np.ndarray):
        if [(row["sample_id"], row["label_lsm"], row["signer_id"]) for row in bone_rows] != [(row["sample_id"], row["label_lsm"], row["signer_id"]) for row in cov_rows]:
            raise ValueError("Filas bone/cov del fold no alineadas")
        self.bone_rows, self.cov_rows = bone_rows, cov_rows
        self.bone_root, self.cov_root, self.labels, self.codebook = bone_root, cov_root, labels, codebook

    def __len__(self) -> int:
        return len(self.bone_rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bone_row, cov_row = self.bone_rows[index], self.cov_rows[index]
        bone = np.load(self.bone_root / bone_row["feature_path"], allow_pickle=False).astype(np.float32, copy=False)
        cov = np.load(self.cov_root / cov_row["feature_path"], allow_pickle=False).astype(np.float32, copy=False)
        code, _ = code_transform(bone, self.codebook)
        if bone.shape != (30, 126) or cov.shape != (30, 168) or code.shape != (30, 190):
            raise ValueError(f"Características incompatibles en {bone_row['sample_id']}")
        if not np.isfinite(bone).all() or not np.isfinite(cov).all() or not np.isfinite(code).all():
            raise ValueError(f"Características no finitas en {bone_row['sample_id']}")
        return torch.from_numpy(bone), torch.from_numpy(cov), torch.from_numpy(code.astype(np.float32, copy=False)), torch.tensor(self.labels[bone_row["label_lsm"]], dtype=torch.long)


def one_epoch(models: dict[str, TemporalTCN], loader: DataLoader, criterion: nn.Module, optimizers: dict[str, torch.optim.Optimizer] | None = None) -> tuple[float, float, np.ndarray, np.ndarray]:
    training = optimizers is not None
    for model in models.values():
        model.train(training)
    total_loss, total_gjs, count = 0.0, 0.0, 0
    all_logits, all_targets = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for bone, cov, code, targets in loader:
            features = {"bone": bone, "cov": cov, "code": code}
            if training:
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
            logits_by_view = [models[name](features[name]) for name, _ in candidate.VIEW_SPECS]
            lexical = sum(criterion(logits, targets) for logits in logits_by_view)
            gjs = candidate.generalized_js(logits_by_view) if training else torch.zeros((), dtype=lexical.dtype)
            loss = lexical + candidate.GJS_LAMBDA * gjs
            if not torch.isfinite(loss):
                raise FloatingPointError("Pérdida GJS LOSO no finita")
            if training:
                loss.backward()
                for model, optimizer in zip(models.values(), optimizers.values(), strict=True):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            total_loss += float(loss.detach()) * targets.size(0)
            total_gjs += float(gjs.detach()) * targets.size(0)
            count += targets.size(0)
            all_logits.append(candidate.uniform_logits(logits_by_view).detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
    return total_loss / count, total_gjs / count, np.concatenate(all_logits), np.concatenate(all_targets)


def train_fold(train_bone: list[dict[str, str]], train_cov: list[dict[str, str]], val_bone: list[dict[str, str]], val_cov: list[dict[str, str]], bone_root: Path, cov_root: Path, labels: dict[str, int], codebook: np.ndarray) -> tuple[dict[str, TemporalTCN], float, int, float]:
    common.seed_everything()
    train_dataset = FoldMultiViewDataset(train_bone, train_cov, bone_root, cov_root, labels, codebook)
    val_dataset = FoldMultiViewDataset(val_bone, val_cov, bone_root, cov_root, labels, codebook)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(common.SEED), num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    models = {name: TemporalTCN(feature_dim=dimensions, classes=210, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20) for name, dimensions in candidate.VIEW_SPECS}
    expected = {"bone": 158_994, "cov": 167_058, "code": 171_282}
    if {name: parameter_count(model) for name, model in models.items()} != expected:
        raise AssertionError("Presupuesto LOSO GJS inesperado")
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001) for name, model in models.items()}
    criterion = nn.CrossEntropyLoss()
    best, best_epoch, stale, best_gjs, state = -1.0, 0, 0, 0.0, None
    for epoch in range(1, common.EPOCHS + 1):
        one_epoch(models, train_loader, criterion, optimizers)
        _, _, logits, targets = one_epoch(models, val_loader, criterion)
        score = common.macro_f1(targets, logits.argmax(axis=1))
        if score > best:
            best, best_epoch, stale, state = score, epoch, 0, copy.deepcopy({name: model.state_dict() for name, model in models.items()})
        else:
            stale += 1
            if stale >= common.PATIENCE:
                break
    if state is None:
        raise AssertionError("LOSO GJS sin checkpoint")
    for name, model in models.items():
        model.load_state_dict(state[name], strict=True)
        model.eval()
    return models, best, best_epoch, best_gjs


def load_control_oof(path: Path) -> dict[str, np.ndarray]:
    values = np.load(path, allow_pickle=False)
    expected = {"sample_ids", "signers", "targets", "bone_logits", "cov_logits", "code_logits"}
    if set(values.files) != expected:
        raise ValueError("OOF uniforme congelado incompatible")
    return {key: values[key] for key in expected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bone-manifest", type=Path, required=True)
    parser.add_argument("--bone-cache-root", type=Path, required=True)
    parser.add_argument("--cov-manifest", type=Path, required=True)
    parser.add_argument("--cov-cache-root", type=Path, required=True)
    parser.add_argument("--control-oof", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-oof", type=Path, required=True)
    args = parser.parse_args()
    common.SEED = 42
    bone_rows, cov_rows = common.load_rows(args.bone_manifest), common.load_rows(args.cov_manifest)
    if [(row["sample_id"], row["label_lsm"], row["signer_id"]) for row in bone_rows] != [(row["sample_id"], row["label_lsm"], row["signer_id"]) for row in cov_rows]:
        raise ValueError("Manifiestos bone/cov incompatibles")
    control = load_control_oof(args.control_oof)
    labels = {label: index for index, label in enumerate(sorted({row["label_lsm"] for row in bone_rows}))}
    reports, oof_parts, deltas = [], [], []
    for outer in common.SIGNERS:
        train_bone = [row for row in bone_rows if row["signer_id"] != outer]
        val_bone = [row for row in bone_rows if row["signer_id"] == outer]
        train_cov = [row for row in cov_rows if row["signer_id"] != outer]
        val_cov = [row for row in cov_rows if row["signer_id"] == outer]
        if len(train_bone) != 1260 or len(val_bone) != 210:
            raise ValueError(f"Fold {outer} inválido")
        codebook = common.fit_codebook(train_bone, args.bone_cache_root)
        models, inner_score, inner_epoch, _ = train_fold(train_bone, train_cov, val_bone, val_cov, args.bone_cache_root, args.cov_cache_root, labels, codebook)
        _, _, candidate_logits, targets = one_epoch(models, DataLoader(FoldMultiViewDataset(val_bone, val_cov, args.bone_cache_root, args.cov_cache_root, labels, codebook), batch_size=64, shuffle=False, num_workers=0), nn.CrossEntropyLoss())
        mask = control["signers"] == outer
        if not np.array_equal(control["sample_ids"][mask], np.asarray([row["sample_id"] for row in val_bone])) or not np.array_equal(control["targets"][mask], targets):
            raise ValueError(f"OOF de control no alineado con fold {outer}")
        control_logits = (control["bone_logits"][mask] + control["cov_logits"][mask] + control["code_logits"][mask]) / 3.0
        candidate_score = common.macro_f1(targets, candidate_logits.argmax(axis=1))
        control_score = common.macro_f1(targets, control_logits.argmax(axis=1))
        delta = candidate_score - control_score
        deltas.append(delta)
        record = {"outer_holdout": outer, "train_samples": 1260, "outer_samples": 210, "codebook_fit_signers": sorted({row["signer_id"] for row in train_bone}), "codebook_excludes_outer": outer not in {row["signer_id"] for row in train_bone}, "inner_uniform_fusion_macro_f1": inner_score, "inner_best_epoch": inner_epoch, "candidate_macro_f1": candidate_score, "control_uniform_macro_f1": control_score, "delta_candidate_minus_control": delta}
        reports.append(record)
        oof_parts.append({"sample_ids": np.asarray([row["sample_id"] for row in val_bone]), "signers": np.asarray([outer] * len(val_bone)), "targets": targets, "candidate_logits": candidate_logits.astype(np.float32, copy=False), "control_logits": control_logits.astype(np.float32, copy=False)})
        print(json.dumps(record), flush=True)
    deltas_array = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(2026)
    bootstrap = rng.choice(deltas_array, size=(10_000, len(deltas_array)), replace=True).mean(axis=1)
    report = {"protocol": "LOSO S01-S07; seed 42; six-signer train-only codebook; 40 epochs; patience 8; GJS lambda 0.10; OOF uniform control frozen", "s08_read": False, "s09_read": False, "s09_evaluated": False, "models_trained_from_scratch": True, "folds": reports, "delta_macro_f1": {"mean": float(deltas_array.mean()), "per_outer_signer": {signer: float(delta) for signer, delta in zip(common.SIGNERS, deltas_array, strict=True)}, "bootstrap_seed": 2026, "bootstrap_samples": 10_000, "ci95": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))]}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.out_oof, **{key: np.concatenate([part[key] for part in oof_parts], axis=0) for key in oof_parts[0]})
    print(json.dumps(report))


if __name__ == "__main__":
    main()