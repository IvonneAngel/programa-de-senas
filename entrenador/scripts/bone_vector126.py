"""Representación cinemática de manos para positions126 con orientación de cámara preservada."""
from __future__ import annotations

import numpy as np


EPSILON = 1e-6
PARENTS = np.asarray((0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19), dtype=np.int64)
CHILDREN = np.arange(1, 21, dtype=np.int64)
MCP = np.asarray((5, 9, 13, 17), dtype=np.int64)


def palm_scale(hand: np.ndarray) -> float | None:
    values = np.asarray(hand, dtype=np.float32)
    if values.shape != (21, 3) or not np.isfinite(values).all():
        raise ValueError(f"Mano inválida: {values.shape}")
    if not values.any():
        return None
    return float(np.linalg.norm(values[MCP], axis=1).mean())


def transform_hand(hand: np.ndarray) -> tuple[np.ndarray, bool]:
    """Devuelve 21×3 (20 huesos + palma) y si se detectó escala degenerada."""
    values = np.asarray(hand, dtype=np.float32)
    scale = palm_scale(values)
    if scale is None:
        return np.zeros((21, 3), dtype=np.float32), False
    if scale <= EPSILON:
        return np.zeros((21, 3), dtype=np.float32), True
    output = np.empty((21, 3), dtype=np.float32)
    output[:20] = (values[CHILDREN] - values[PARENTS]) / scale
    output[20] = values[MCP].mean(axis=0) / scale
    if not np.isfinite(output).all():
        raise ValueError("bone_vector no finito")
    return output, False


def transform_sequence(sequence: np.ndarray) -> tuple[np.ndarray, int]:
    """Transforma `(30,126)` y devuelve cantidad de manos degeneradas no sustituidas."""
    values = np.asarray(sequence, dtype=np.float32)
    if values.shape != (30, 126) or not np.isfinite(values).all():
        raise ValueError(f"Secuencia inválida: {values.shape}")
    output = np.empty_like(values)
    degeneracies = 0
    for index, frame in enumerate(values):
        left, left_degenerate = transform_hand(frame[:63].reshape(21, 3))
        right, right_degenerate = transform_hand(frame[63:].reshape(21, 3))
        output[index, :63] = left.reshape(63)
        output[index, 63:] = right.reshape(63)
        degeneracies += int(left_degenerate) + int(right_degenerate)
    if output.shape != (30, 126) or not np.isfinite(output).all():
        raise AssertionError("Contrato bone_vector126 inválido")
    return output, degeneracies