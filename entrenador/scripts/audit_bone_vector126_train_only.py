"""Audita la escala palmar de positions126 solo sobre entrenamiento S01-S07."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MCP = np.asarray([5, 9, 13, 17], dtype=np.int64)


def palm_scale(hand: np.ndarray) -> float | None:
    if hand.shape != (63,):
        raise ValueError(hand.shape)
    if not np.any(hand):
        return None
    points = hand.reshape(21, 3)
    return float(np.linalg.norm(points[MCP], axis=1).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split_project"] == "train" and row["feature_status"] == "ok"]
    if len(rows) != 1470:
        raise ValueError(f"Se esperaban 1470 muestras train, no {len(rows)}")
    scales: list[float] = []
    nonzero_hands = 0
    degenerate: list[dict[str, str | int | float]] = []
    for row in rows:
        sequence = np.load(args.cache_root / row["feature_path"], allow_pickle=False)
        if sequence.shape != (30, 126) or not np.isfinite(sequence).all():
            raise ValueError(f"Tensor inválido: {row['sample_id']}")
        for frame_index, frame in enumerate(sequence):
            for side, hand in (("left", frame[:63]), ("right", frame[63:])):
                scale = palm_scale(hand)
                if scale is None:
                    continue
                nonzero_hands += 1
                scales.append(scale)
                if scale <= 1e-6:
                    degenerate.append({"sample_id": row["sample_id"], "frame": frame_index, "side": side, "scale": scale})
    report = {
        "split_read": "train_only",
        "clips": len(rows),
        "nonzero_hands": nonzero_hands,
        "degenerate_scales_at_or_below_1e-6": len(degenerate),
        "scale": {"min": min(scales), "median": float(np.median(scales)), "p05": float(np.percentile(scales, 5)), "p95": float(np.percentile(scales, 95)), "max": max(scales)},
        "degenerate_examples": degenerate[:20],
        "s08_read": False,
        "s09_read": False,
        "cache_written": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()