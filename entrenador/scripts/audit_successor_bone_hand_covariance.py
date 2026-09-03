"""Audita covarianza temporal compacta de bone_vector126 solo en S01-S07."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CHAINS = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15), (16, 17, 18, 19))
UPPER = np.triu_indices(6)


def hand_covariance(hand: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(hand, dtype=np.float32)
    if values.shape != (30, 21, 3) or not np.isfinite(values).all():
        raise ValueError(f"Mano inválida: {values.shape}")
    present = np.linalg.norm(values, axis=(1, 2)) > 1e-6
    observed = values[present]
    if observed.shape[0] < 2:
        return np.zeros(21, dtype=np.float32), int(observed.shape[0])
    chains = np.asarray([np.linalg.norm(observed[:, chain, :], axis=2).sum(axis=1) for chain in CHAINS], dtype=np.float32).T
    palm_norm = np.linalg.norm(observed[:, 20, :], axis=1, keepdims=True)
    traits = np.concatenate((chains, palm_norm), axis=1)
    centered = traits - traits.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / float(observed.shape[0] - 1)
    return covariance[UPPER].astype(np.float32), int(observed.shape[0])


def transform(values: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    base = np.asarray(values, dtype=np.float32)
    if base.shape != (30, 126) or not np.isfinite(base).all():
        raise ValueError(f"Secuencia inválida: {base.shape}")
    hands = base.reshape(30, 2, 21, 3)
    left, left_count = hand_covariance(hands[:, 0])
    right, right_count = hand_covariance(hands[:, 1])
    context = np.concatenate((left, right), axis=0)
    result = np.concatenate((base, np.broadcast_to(context, (30, 42))), axis=1).astype(np.float32, copy=False)
    return result, (left_count, right_count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [row for row in rows if row["split_model"] == "train"]
    if len(train) != 1470 or any(row["signer_id"] not in {f"S{i:02d}" for i in range(1, 8)} for row in train):
        raise ValueError("La auditoría exige 1,470 clips S01-S07")
    classes: dict[str, int] = defaultdict(int)
    norms: list[float] = []
    sparse_hands = 0
    for row in train:
        base = np.load(args.cache_root / row["feature_path"], allow_pickle=False)
        transformed, counts = transform(base)
        if transformed.shape != (30, 168) or not np.isfinite(transformed).all():
            raise ValueError(f"Covarianza inválida: {row['sample_id']}")
        if not np.array_equal(transformed[:, :126], base.astype(np.float32)):
            raise AssertionError("El prefijo bone_vector126 cambió")
        context = transformed[0, 126:]
        norms.append(float(np.linalg.norm(context)))
        if min(counts) < 2:
            sparse_hands += 1
        if np.linalg.norm(context) > 1e-8:
            classes[row["label_lsm"]] += 1
    report = {
        "kind": "successor_bone_hand_covariance_train_only_audit",
        "split_read": "train_only_S01_to_S07",
        "s08_read": False,
        "s09_read": False,
        "cache_written": False,
        "samples": len(train),
        "classes": len({row["label_lsm"] for row in train}),
        "candidate_shape": [30, 168],
        "context_norm_mean": float(np.mean(norms)),
        "context_norm_p95": float(np.percentile(norms, 95)),
        "classes_with_nonzero_covariance": len(classes),
        "clips_with_hand_under_two_observations": sparse_hands,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()