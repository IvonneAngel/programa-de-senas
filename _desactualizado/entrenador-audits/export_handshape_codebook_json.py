"""Convierte el codebook NPZ train-only a JSON para su uso local en React Native."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    with np.load(args.input, allow_pickle=False) as artifact:
        centers = np.asarray(artifact["centers"], dtype=np.float32)
        clusters = int(artifact["clusters"])
        feature_dim = int(artifact["feature_dim"])
        fitted_frames = int(artifact["fitted_frames"])
    if centers.shape != (32, 63) or clusters != 32 or feature_dim != 63 or not np.isfinite(centers).all():
        raise ValueError("Codebook incompatible para exportación móvil")
    payload = {"kind": "bone_code190_train_only_handshape_codebook", "sha256": digest, "clusters": clusters, "feature_dim": feature_dim, "fitted_frames": fitted_frames, "centers": centers.tolist(), "s08_read": False, "s09_read": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "sha256": digest, "shape": list(centers.shape)}))


if __name__ == "__main__":
    main()