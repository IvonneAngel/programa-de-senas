"""Procesa un fixture crudo anonimizado con la referencia Python body_anchor51."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from extract_successor_body_anchor51_recovery import HAND_POINTS, LEFT_SHOULDER, RIGHT_SHOULDER, body_motion


def numbers(raw: str, expected: int) -> np.ndarray:
    values = np.asarray([float(value) for value in raw.split(",") if value], dtype=np.float32)
    if values.shape != (expected,) or not np.isfinite(values).all():
        raise ValueError(f"Se esperaban {expected} valores finitos")
    return values


def position(fields: list[str]) -> np.ndarray:
    if len(fields) != 6:
        raise ValueError("Cada frame debe tener seis campos")
    right_flag, left_flag, pose_flag, right_values, left_values, pose_values = fields
    output = np.zeros(51, dtype=np.float32)
    if pose_flag != "1":
        return output
    pose = numbers(pose_values, 132).reshape(33, 4)
    left_shoulder, right_shoulder = pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER]
    if min(float(left_shoulder[3]), float(right_shoulder[3])) < 0.5:
        return output
    center = (left_shoulder[:2] + right_shoulder[:2]) * 0.5
    scale = float(np.hypot(*(left_shoulder[:2] - right_shoulder[:2])))
    if not np.isfinite(scale) or scale <= 1e-6:
        return output
    output[50] = 1.0
    for slot, (flag, values) in enumerate(((right_flag, right_values), (left_flag, left_values))):
        if flag != "1":
            continue
        hand = numbers(values, 63).reshape(21, 3)
        start = slot * 12
        for offset, point_index in enumerate(HAND_POINTS):
            output[start + offset * 2:start + offset * 2 + 2] = (hand[point_index, :2] - center) / scale
        output[48 + slot] = 1.0
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    lines = [line for line in args.fixture.read_text(encoding="utf-8").splitlines() if line]
    positioned = np.stack([position(line.split("|")) for line in lines])
    if positioned.shape != (30, 51):
        raise ValueError(f"El fixture debe formar (30,51), recibió {positioned.shape}")
    for row in body_motion(positioned):
        print(" ".join(f"{float(value):.9f}" for value in row))


if __name__ == "__main__":
    main()