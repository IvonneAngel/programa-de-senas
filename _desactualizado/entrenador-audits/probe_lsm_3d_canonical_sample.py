"""Sonda determinista de precisión para una nube 3D original; no escribe datos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()
    points64 = np.loadtxt(args.path, dtype=np.float64)
    centroid64 = points64.mean(axis=0, keepdims=True)
    canonical64 = (points64 - centroid64) / np.linalg.norm(points64 - centroid64, axis=1).max()
    canonical32 = canonical64.astype(np.float32)
    report = {
        "path": str(args.path),
        "points": int(points64.shape[0]),
        "canonical64_centroid_norm": float(np.linalg.norm(canonical64.mean(axis=0))),
        "canonical64_max_radius": float(np.linalg.norm(canonical64, axis=1).max()),
        "canonical32_centroid_norm": float(np.linalg.norm(canonical32.mean(axis=0))),
        "canonical32_max_radius": float(np.linalg.norm(canonical32, axis=1).max()),
        "finite": bool(np.isfinite(canonical32).all()),
    }
    print(json.dumps(report))


if __name__ == "__main__":
    main()