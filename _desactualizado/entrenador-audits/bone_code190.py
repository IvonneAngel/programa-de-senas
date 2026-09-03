"""Código discreto de configuraciones manuales sobre bone_vector126."""
from __future__ import annotations

from pathlib import Path

import numpy as np


CLUSTERS = 32


def load_centers(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as artifact:
        centers = np.asarray(artifact["centers"], dtype=np.float32)
        clusters = int(artifact["clusters"])
        feature_dim = int(artifact["feature_dim"])
    if clusters != CLUSTERS or feature_dim != 63 or centers.shape != (CLUSTERS, 63) or not np.isfinite(centers).all():
        raise ValueError("Codebook incompatible con bone_code190")
    return centers


def encode_hand(hand: np.ndarray, centers: np.ndarray) -> np.ndarray:
    values = np.asarray(hand, dtype=np.float32)
    if values.shape != (30, 63) or centers.shape != (CLUSTERS, 63):
        raise ValueError("Mano o codebook incompatibles")
    output = np.zeros((30, CLUSTERS), dtype=np.float32)
    present = np.linalg.norm(values, axis=1) > 1e-6
    if np.any(present):
        distances = np.square(values[present, None, :] - centers[None, :, :]).sum(axis=2)
        output[np.flatnonzero(present), distances.argmin(axis=1)] = 1.0
    return output


def transform_sequence(sequence: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    base = np.asarray(sequence, dtype=np.float32)
    if base.shape != (30, 126) or not np.isfinite(base).all():
        raise ValueError(f"Secuencia bone inválida: {base.shape}")
    left = encode_hand(base[:, :63], centers)
    right = encode_hand(base[:, 63:], centers)
    output = np.concatenate((base, left, right), axis=1).astype(np.float32, copy=False)
    return output, (int(left.sum()), int(right.sum()))