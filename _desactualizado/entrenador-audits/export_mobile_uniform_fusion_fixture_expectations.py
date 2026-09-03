"""Genera expectativas Python para paridad móvil sobre fixture anonimizado de 30 frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bone_code190 import load_centers, transform_sequence as code_transform
from bone_cov168 import transform_sequence as cov_transform
from bone_vector126 import transform_sequence as bone_transform


def parse_hand(raw: str) -> np.ndarray:
    values = np.asarray([float(value) for value in raw.split(",") if value], dtype=np.float32).reshape(21, 3)
    return values - values[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frames = []
    for line in args.fixture.read_text(encoding="utf-8").splitlines():
        right_flag, left_flag, _pose_flag, right_raw, left_raw, _pose_raw = line.split("|")
        left = parse_hand(left_raw).reshape(-1) if left_flag == "1" else np.zeros(63, dtype=np.float32)
        right = parse_hand(right_raw).reshape(-1) if right_flag == "1" else np.zeros(63, dtype=np.float32)
        frames.append(np.concatenate((left, right)))
    raw = np.stack(frames).astype(np.float32)
    bone, degeneracies = bone_transform(raw)
    cov, cov_observations = cov_transform(bone)
    code, code_observations = code_transform(bone, load_centers(args.codebook))
    codes = []
    for frame in range(30):
        row = []
        for start in (126, 158):
            active = np.flatnonzero(code[frame, start:start + 32] == 1.0)
            row.append(int(active[0]) if active.size else -1)
        codes.append(row)
    payload = {
        "kind": "mobile_uniform_fusion_real_fixture_expectations",
        "frames": 30,
        "bone_degeneracies": degeneracies,
        "cov_hand_observations": list(cov_observations),
        "code_hand_observations": list(code_observations),
        "cov_context": cov[0, 126:].astype(float).tolist(),
        "code_indices_per_frame": codes,
        "s08_read": False,
        "s09_read": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "cov_observations": list(cov_observations), "code_observations": list(code_observations)}))


if __name__ == "__main__":
    main()