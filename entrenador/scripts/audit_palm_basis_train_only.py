"""Audita la estabilidad de una base de palma usando únicamente tensores S01–S07."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EPSILON = 1e-6


def basis(hand: np.ndarray) -> np.ndarray | None:
    if hand.shape != (21, 3) or not np.isfinite(hand).all() or not hand.any():
        return None
    forward = hand[9]
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= EPSILON:
        return None
    first = forward / forward_norm
    lateral_raw = hand[5] - np.dot(hand[5], first) * first
    lateral_norm = float(np.linalg.norm(lateral_raw))
    if lateral_norm <= EPSILON:
        return None
    second = lateral_raw / lateral_norm
    third = np.cross(first, second)
    if float(np.linalg.norm(third)) <= EPSILON:
        return None
    return np.stack((first, second, third), axis=1)


def rotation_angle(left: np.ndarray, right: np.ndarray) -> float:
    relative = left.T @ right
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [row for row in rows if row["split_project"] == "train" and row.get("feature_status") == "ok"]
    if len(train) != 1470:
        raise ValueError(f"Se esperaban 1,470 muestras train válidas, se obtuvieron {len(train)}")
    rotations: list[float] = []
    frames_seen = 0
    hands_seen = 0
    bases_valid = 0
    for row in train:
        sequence = np.load(args.cache_root / row["feature_path"], allow_pickle=False)
        if sequence.shape != (30, 126) or not np.isfinite(sequence).all():
            raise ValueError(f"Tensor inválido: {row['sample_id']}")
        for offset in (0, 63):
            prior: np.ndarray | None = None
            for frame in sequence[:, offset:offset + 63].reshape(30, 21, 3):
                frames_seen += 1
                if frame.any():
                    hands_seen += 1
                current = basis(frame)
                if current is None:
                    prior = None
                    continue
                bases_valid += 1
                if prior is not None:
                    rotations.append(rotation_angle(prior, current))
                prior = current
    values = np.asarray(rotations, dtype=np.float64)
    report = {
        "data_scope": {"splits_read": ["train"], "s08_tensor_read": False, "s09_tensor_read": False, "samples": len(train)},
        "basis_definition": {"forward_joint": 9, "lateral_joint": 5, "epsilon": EPSILON},
        "frames_total": frames_seen,
        "hands_detected": hands_seen,
        "bases_valid": bases_valid,
        "bases_degenerate_or_absent": frames_seen - bases_valid,
        "basis_valid_share_among_detected_hands": float(bases_valid / hands_seen) if hands_seen else 0.0,
        "consecutive_rotation_radians": {
            "count": int(values.size),
            "median": float(np.median(values)) if values.size else None,
            "p90": float(np.quantile(values, 0.9)) if values.size else None,
            "p95": float(np.quantile(values, 0.95)) if values.size else None,
            "over_1_rad": int(np.sum(values > 1.0)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()