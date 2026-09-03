"""Benchmark offline CPU de bagging cross-signer sobre una muestra S01 real."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from bone_code190 import transform_sequence as code_transform
from evaluate_successor_cross_signer_bagging import load_model


FOLDS = tuple(f"S{value:02d}" for value in range(1, 8))


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bone-manifest", type=Path, required=True)
    parser.add_argument("--bone-cache-root", type=Path, required=True)
    parser.add_argument("--cov-manifest", type=Path, required=True)
    parser.add_argument("--cov-cache-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=200)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with args.bone_manifest.open(encoding="utf-8", newline="") as handle:
        bone_row = next(row for row in csv.DictReader(handle) if row["feature_status"] == "ok" and row["signer_id"] == "S01")
    with args.cov_manifest.open(encoding="utf-8", newline="") as handle:
        cov_row = next(row for row in csv.DictReader(handle) if row["sample_id"] == bone_row["sample_id"])
    bone_np = np.load(args.bone_cache_root / bone_row["feature_path"], allow_pickle=False).astype(np.float32)
    cov_np = np.load(args.cov_cache_root / cov_row["feature_path"], allow_pickle=False).astype(np.float32)
    if bone_np.shape != (30, 126) or cov_np.shape != (30, 168):
        raise ValueError("Muestra de benchmark inválida")
    models, codebooks, labels = [], [], None
    for fold in FOLDS:
        folder = args.fold_root / fold
        bone, current = load_model(folder / "bone.pt", 126)
        cov, cov_labels = load_model(folder / "cov.pt", 168)
        code, code_labels = load_model(folder / "code.pt", 190)
        if current != cov_labels or current != code_labels or (labels is not None and labels != current):
            raise ValueError("Etiquetas incompatibles")
        labels = current
        models.append((bone, cov, code))
        codebooks.append(np.load(folder / "codebook32.npy", allow_pickle=False))
    with torch.no_grad():
        for (bone, cov, code), codebook in zip(models, codebooks):
            code_np, _ = code_transform(bone_np, codebook)
            _ = bone(torch.from_numpy(bone_np[None])) + cov(torch.from_numpy(cov_np[None])) + code(torch.from_numpy(code_np[None]))
    formation_ms, inference_ms, argmaxes = [], [], []
    with torch.no_grad():
        for _ in range(args.repetitions):
            start = time.perf_counter_ns()
            code_sequences = [code_transform(bone_np, codebook)[0].astype(np.float32, copy=False) for codebook in codebooks]
            middle = time.perf_counter_ns()
            logits = []
            for (bone, cov, code), code_np in zip(models, code_sequences):
                logits.append((bone(torch.from_numpy(bone_np[None])) + cov(torch.from_numpy(cov_np[None])) + code(torch.from_numpy(code_np[None]))) / 3.0)
            fused = torch.stack(logits).mean(dim=0)
            end = time.perf_counter_ns()
            formation_ms.append((middle - start) / 1_000_000)
            inference_ms.append((end - middle) / 1_000_000)
            argmaxes.append(int(fused.argmax(dim=1).item()))
    if len(set(argmaxes)) != 1:
        raise AssertionError("Argmax inestable en benchmark determinista")
    report = {"protocol": "CPU sandbox; muestra S01 real; 7 folds × 3 TCN; sin cámara", "repetitions": args.repetitions, "s08_read": False, "s09_read": False, "formation_code190_ms": {"p50": percentile(formation_ms, 50), "p95": percentile(formation_ms, 95)}, "inference_21_tcn_ms": {"p50": percentile(inference_ms, 50), "p95": percentile(inference_ms, 95)}, "argmax_stable": True, "argmax_index": argmaxes[0], "mobile_equivalent": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()