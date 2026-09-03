"""Confirmación LOSO anidada sin S08/S09 del bagging cross-signer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import confirm_uniform_fusion_loso as common


def aligned(rows: list[dict[str, str]], signer: str) -> list[dict[str, str]]:
    return [row for row in rows if row["signer_id"] == signer]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bone-manifest", type=Path, required=True)
    parser.add_argument("--bone-cache-root", type=Path, required=True)
    parser.add_argument("--cov-manifest", type=Path, required=True)
    parser.add_argument("--cov-cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    common.SEED = 42
    bone_rows, cov_rows = common.load_rows(args.bone_manifest), common.load_rows(args.cov_manifest)
    if [(row["sample_id"], row["label_lsm"], row["signer_id"]) for row in bone_rows] != [(row["sample_id"], row["label_lsm"], row["signer_id"]) for row in cov_rows]:
        raise ValueError("Manifiestos bone/cov incompatibles")
    labels = {label: index for index, label in enumerate(sorted({row["label_lsm"] for row in bone_rows}))}
    folds: list[dict[str, object]] = []
    for outer in common.SIGNERS:
        outer_bone, outer_cov = aligned(bone_rows, outer), aligned(cov_rows, outer)
        inner_signers = tuple(signer for signer in common.SIGNERS if signer != outer)
        if len(outer_bone) != 210 or len(outer_cov) != 210:
            raise ValueError(f"Outer {outer} inválido")
        fold_fusions: list[np.ndarray] = []
        fold_bones: list[np.ndarray] = []
        target_reference: np.ndarray | None = None
        inner_records: list[dict[str, object]] = []
        for inner in inner_signers:
            train_bone = [row for row in bone_rows if row["signer_id"] not in {outer, inner}]
            val_bone = aligned(bone_rows, inner)
            train_cov = [row for row in cov_rows if row["signer_id"] not in {outer, inner}]
            val_cov = aligned(cov_rows, inner)
            if len(train_bone) != 1050 or len(val_bone) != 210:
                raise ValueError(f"Nested {outer}/{inner} inválido")
            codebook = common.fit_codebook(train_bone, args.bone_cache_root)
            bone, bone_f1, bone_epoch = common.train_view(train_bone, val_bone, args.bone_cache_root, labels, 126, 158_994)
            cov, cov_f1, cov_epoch = common.train_view(train_cov, val_cov, args.cov_cache_root, labels, 168, 167_058)
            code, code_f1, code_epoch = common.train_view(train_bone, val_bone, args.bone_cache_root, labels, 190, 171_282, codebook)
            bone_logits, targets = common.predict(bone, outer_bone, args.bone_cache_root, labels, 126)
            cov_logits, cov_targets = common.predict(cov, outer_cov, args.cov_cache_root, labels, 168)
            code_logits, code_targets = common.predict(code, outer_bone, args.bone_cache_root, labels, 190, codebook)
            if not np.array_equal(targets, cov_targets) or not np.array_equal(targets, code_targets):
                raise ValueError(f"Targets incompatibles {outer}/{inner}")
            if target_reference is None:
                target_reference = targets
            elif not np.array_equal(target_reference, targets):
                raise ValueError(f"Targets outer inconsistentes {outer}")
            fold_bones.append(bone_logits)
            fold_fusions.append((bone_logits + cov_logits + code_logits) / 3.0)
            inner_records.append({"inner_holdout": inner, "train_signers": [signer for signer in common.SIGNERS if signer not in {outer, inner}], "outer_excluded_from_train": outer not in {row["signer_id"] for row in train_bone}, "outer_excluded_from_codebook": outer not in {row["signer_id"] for row in train_bone}, "bone": {"macro_f1_inner": bone_f1, "epoch": bone_epoch}, "cov": {"macro_f1_inner": cov_f1, "epoch": cov_epoch}, "code": {"macro_f1_inner": code_f1, "epoch": code_epoch}})
        bag_bone = np.mean(np.stack(fold_bones), axis=0)
        bag_fusion = np.mean(np.stack(fold_fusions), axis=0)
        bone_score = common.macro_f1(target_reference, bag_bone.argmax(axis=1))
        fusion_score = common.macro_f1(target_reference, bag_fusion.argmax(axis=1))
        fold = {"outer_holdout": outer, "outer_samples": 210, "inner_folds": inner_records, "bagged_bone_macro_f1": bone_score, "bagged_fusion_macro_f1": fusion_score, "delta_fusion_minus_bone_bagging": fusion_score - bone_score}
        folds.append(fold)
        print(json.dumps(fold), flush=True)
    deltas = np.asarray([float(fold["delta_fusion_minus_bone_bagging"]) for fold in folds], dtype=np.float64)
    rng = np.random.default_rng(2026)
    bootstrap = rng.choice(deltas, size=(10_000, len(deltas)), replace=True).mean(axis=1)
    report = {"protocol": "Nested LOSO S01-S07; outer signer excluded from all inner training/codebooks; seed 42; 40 epochs; patience 8", "formula_candidate": "mean_inner((bone+cov+code)/3)", "formula_control": "mean_inner(bone)", "s08_read": False, "s09_read": False, "s09_evaluated": False, "models_trained_from_scratch": True, "outer_folds": folds, "delta_macro_f1": {"mean": float(deltas.mean()), "per_outer_signer": {str(fold["outer_holdout"]): float(fold["delta_fusion_minus_bone_bagging"]) for fold in folds}, "bootstrap_seed": 2026, "bootstrap_samples": 10_000, "ci95": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))]}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()