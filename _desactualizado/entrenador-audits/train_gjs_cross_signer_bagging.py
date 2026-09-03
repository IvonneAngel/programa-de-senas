"""Entrena siete ensembles GJS por exclusión de firmante, sin S08/S09."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import confirm_multiview_js_consistency_loso as gjs_loso
import confirm_uniform_fusion_loso as common


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bone-manifest", type=Path, required=True)
    parser.add_argument("--bone-cache-root", type=Path, required=True)
    parser.add_argument("--cov-manifest", type=Path, required=True)
    parser.add_argument("--cov-cache-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--hashes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.skip_test_evaluation:
        raise ValueError("El bagging GJS exige --skip-test-evaluation para proteger S09")
    common.SEED = args.seed
    bone_rows = common.load_rows(args.bone_manifest)
    cov_rows = common.load_rows(args.cov_manifest)
    if [(row["sample_id"], row["label_lsm"], row["signer_id"]) for row in bone_rows] != [(row["sample_id"], row["label_lsm"], row["signer_id"]) for row in cov_rows]:
        raise ValueError("Manifiestos bone/cov incompatibles")
    labels = {label: index for index, label in enumerate(sorted({row["label_lsm"] for row in bone_rows}))}
    folds: list[dict[str, object]] = []
    hash_paths: list[Path] = []
    for holdout in common.SIGNERS:
        train_bone = [row for row in bone_rows if row["signer_id"] != holdout]
        val_bone = [row for row in bone_rows if row["signer_id"] == holdout]
        train_cov = [row for row in cov_rows if row["signer_id"] != holdout]
        val_cov = [row for row in cov_rows if row["signer_id"] == holdout]
        if len(train_bone) != 1260 or len(val_bone) != 210:
            raise ValueError(f"Fold cross-signer inválido: {holdout}")
        codebook = common.fit_codebook(train_bone, args.bone_cache_root)
        models, validation_f1, best_epoch, _ = gjs_loso.train_fold(train_bone, train_cov, val_bone, val_cov, args.bone_cache_root, args.cov_cache_root, labels, codebook)
        folder = args.fold_root / holdout
        folder.mkdir(parents=True, exist_ok=True)
        metadata = {"holdout_signer": holdout, "train_signers": sorted({row["signer_id"] for row in train_bone}), "labels": labels, "seed": args.seed, "gjs_lambda": 0.10, "s08_read": False, "s09_read": False, "s09_evaluated": False, "trained_from_scratch": True, "best_inner_macro_f1": validation_f1, "best_epoch": best_epoch}
        for name, dimensions in (("bone", 126), ("cov", 168), ("code", 190)):
            checkpoint = folder / f"{name}.pt"
            torch.save({"state_dict": models[name].state_dict(), "feature_dim": dimensions, **metadata}, checkpoint)
            hash_paths.append(checkpoint.resolve())
        codebook_path = folder / "codebook32.npy"
        np.save(codebook_path, codebook, allow_pickle=False)
        hash_paths.append(codebook_path.resolve())
        metadata_path = folder / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        hash_paths.append(metadata_path.resolve())
        record = {"holdout": holdout, "train_samples": len(train_bone), "validation_samples": len(val_bone), "codebook_excludes_holdout": holdout not in {row["signer_id"] for row in train_bone}, "best_inner_macro_f1": validation_f1, "best_epoch": best_epoch}
        folds.append(record)
        print(json.dumps(record), flush=True)
    if len(hash_paths) != 35:
        raise AssertionError("El ensamble GJS debe producir 35 huellas")
    args.hashes.parent.mkdir(parents=True, exist_ok=True)
    args.hashes.write_text("".join(f"{sha256(path)}  {path}\n" for path in hash_paths), encoding="utf-8")
    report = {"kind": "gjs_cross_signer_bagging_training", "seed": args.seed, "gjs_lambda": 0.10, "folds": folds, "fold_count": len(folds), "hash_count": len(hash_paths), "s08_read": False, "s09_read": False, "s09_evaluated": False, "trained_from_scratch": True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()