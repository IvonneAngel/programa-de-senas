"""Contexto de covarianza temporal compacto derivado de bone_vector126."""
from __future__ import annotations

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
    chain_lengths = np.asarray([np.linalg.norm(observed[:, chain, :], axis=2).sum(axis=1) for chain in CHAINS], dtype=np.float32).T
    palm_length = np.linalg.norm(observed[:, 20, :], axis=1, keepdims=True)
    traits = np.concatenate((chain_lengths, palm_length), axis=1)
    centered = traits - traits.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / float(observed.shape[0] - 1)
    return covariance[UPPER].astype(np.float32), int(observed.shape[0])


def transform_sequence(sequence: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    base = np.asarray(sequence, dtype=np.float32)
    if base.shape != (30, 126) or not np.isfinite(base).all():
        raise ValueError(f"Secuencia bone inválida: {base.shape}")
    hands = base.reshape(30, 2, 21, 3)
    left, left_frames = hand_covariance(hands[:, 0])
    right, right_frames = hand_covariance(hands[:, 1])
    context = np.concatenate((left, right), axis=0)
    return np.concatenate((base, np.broadcast_to(context, (30, 42))), axis=1).astype(np.float32, copy=False), (left_frames, right_frames)