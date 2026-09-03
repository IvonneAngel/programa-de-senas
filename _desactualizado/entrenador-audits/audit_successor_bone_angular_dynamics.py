"""Audita dinámica angular sobre bone_vector126 usando exclusivamente S01–S07."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def angular_deltas(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve 40 ángulos inter-frame (20 por mano) y su máscara factual."""
    values = np.asarray(sequence, dtype=np.float32)
    if values.shape != (30, 126) or not np.isfinite(values).all():
        raise ValueError(f"Secuencia inválida: {values.shape}")
    bones = values.reshape(30, 2, 21, 3)[:, :, :20, :]
    norms = np.linalg.norm(bones, axis=-1)
    valid = norms > 1e-6
    units = np.divide(bones, norms[..., None], out=np.zeros_like(bones), where=valid[..., None])
    dot = np.sum(units[1:] * units[:-1], axis=-1)
    pair_valid = valid[1:] & valid[:-1]
    angles = np.where(pair_valid, np.arccos(np.clip(dot, -1.0, 1.0)), 0.0).astype(np.float32)
    output = np.zeros((30, 40), dtype=np.float32)
    output[1:] = angles.reshape(29, 40)
    mask = np.zeros((30, 40), dtype=bool)
    mask[1:] = pair_valid.reshape(29, 40)
    return output, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [row for row in rows if row["split_model"] == "train" and row["feature_status"] == "ok"]
    if len(train) != 1470 or any(row["signer_id"] not in {f"S{index:02d}" for index in range(1, 8)} for row in train):
        raise ValueError("La auditoría angular exige exactamente S01–S07 de train")
    if any(row["split_model"] in {"validation", "test"} for row in train):
        raise AssertionError("Fuga de split")
    values, masks, labels, signers = [], [], [], []
    for row in train:
        sequence = np.load(args.cache_root / row["feature_path"], allow_pickle=False)
        angular, mask = angular_deltas(sequence)
        values.append(angular)
        masks.append(mask)
        labels.append(row["label_lsm"])
        signers.append(row["signer_id"])
    stack, mask_stack = np.stack(values), np.stack(masks)
    per_clip = mask_stack.mean(axis=(1, 2))
    per_class = {label: float(np.mean([per_clip[index] for index, value in enumerate(labels) if value == label])) for label in sorted(set(labels))}
    report = {
        "kind": "successor_bone_angular_dynamics_train_only_audit",
        "split_read": "train_only_S01_to_S07",
        "s08_read": False,
        "s09_read": False,
        "cache_written": False,
        "clips": len(train),
        "classes": len(set(labels)),
        "signers": sorted(set(signers)),
        "shape_per_clip": [30, 40],
        "finite": bool(np.isfinite(stack).all()),
        "angular_valid_fraction": float(mask_stack.mean()),
        "clip_valid_fraction": {"min": float(per_clip.min()), "median": float(np.median(per_clip)), "p05": float(np.percentile(per_clip, 5)), "p95": float(np.percentile(per_clip, 95))},
        "nonzero_angle_fraction": float((stack[mask_stack] > 1e-6).mean()) if mask_stack.any() else 0.0,
        "angle_radians": {"median": float(np.median(stack[mask_stack])) if mask_stack.any() else 0.0, "p95": float(np.percentile(stack[mask_stack], 95)) if mask_stack.any() else 0.0},
        "classes_with_zero_valid_fraction": sorted(label for label, fraction in per_class.items() if fraction == 0.0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()