"""Auditoría train-only de volúmenes tetraédricos locales de mano."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FINGERS = np.asarray(((1, 2, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)), dtype=np.int64)
MCP = np.asarray((5, 9, 13, 17), dtype=np.int64)


def tetra_volumes(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(sequence, dtype=np.float32)
    if values.shape != (30, 126) or not np.isfinite(values).all():
        raise ValueError(f"positions126 inválido: {values.shape}")
    output = np.zeros((30, 10), dtype=np.float32)
    valid = np.zeros((30, 10), dtype=bool)
    for frame_index, frame in enumerate(values):
        for hand_index, flat in enumerate((frame[:63], frame[63:])):
            hand = flat.reshape(21, 3)
            if not hand.any():
                continue
            scale = np.linalg.norm(hand[MCP], axis=1).mean()
            if not np.isfinite(scale) or scale <= 1e-6:
                continue
            vectors = hand[FINGERS] / scale
            output[frame_index, hand_index * 5:(hand_index + 1) * 5] = np.linalg.det(vectors).astype(np.float32)
            valid[frame_index, hand_index * 5:(hand_index + 1) * 5] = True
    if not np.isfinite(output).all():
        raise AssertionError("Firma tetraédrica no finita")
    return output, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [row for row in rows if row["task"] == "successor_positions126" and row["feature_status"] == "ok" and row["split_model"] == "train"]
    if len(train) != 1470 or any(row["signer_id"] not in {f"S{index:02d}" for index in range(1, 8)} for row in train):
        raise ValueError("La auditoría exige exactamente S01–S07 de train")
    values, masks, labels = [], [], []
    for row in train:
        tetra, valid = tetra_volumes(np.load(args.cache_root / row["feature_path"], allow_pickle=False))
        values.append(tetra)
        masks.append(valid)
        labels.append(row["label_lsm"])
    stack, mask_stack = np.stack(values), np.stack(masks)
    per_clip = mask_stack.mean(axis=(1, 2))
    per_class = {label: float(np.mean([per_clip[index] for index, value in enumerate(labels) if value == label])) for label in sorted(set(labels))}
    masked = stack[mask_stack]
    report = {
        "kind": "successor_hand_tetra_volume_train_only_audit",
        "split_read": "train_only_S01_to_S07",
        "s08_read": False,
        "s09_read": False,
        "cache_written": False,
        "clips": len(train),
        "classes": len(set(labels)),
        "shape_per_clip": [30, 10],
        "finite": bool(np.isfinite(stack).all()),
        "valid_fraction": float(mask_stack.mean()),
        "clip_valid_fraction": {"min": float(per_clip.min()), "median": float(np.median(per_clip)), "p05": float(np.percentile(per_clip, 5)), "p95": float(np.percentile(per_clip, 95))},
        "nonzero_fraction": float((np.abs(masked) > 1e-6).mean()) if masked.size else 0.0,
        "absolute_volume": {"median": float(np.median(np.abs(masked))) if masked.size else 0.0, "p95": float(np.percentile(np.abs(masked), 95)) if masked.size else 0.0},
        "classes_with_zero_valid_fraction": sorted(label for label, fraction in per_class.items() if fraction == 0.0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()