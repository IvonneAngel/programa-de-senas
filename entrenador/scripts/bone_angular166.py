"""Dinámica angular factual de 20 vectores óseos por mano."""
from __future__ import annotations

import numpy as np


EPSILON = 1e-6


def angular_dynamics(sequence: np.ndarray) -> np.ndarray:
    """Deriva `(30,40)` de bone_vector126; ceros preservan ausencia o primer frame."""
    values = np.asarray(sequence, dtype=np.float32)
    if values.shape != (30, 126) or not np.isfinite(values).all():
        raise ValueError(f"bone_vector126 inválido: {values.shape}")
    bones = values.reshape(30, 2, 21, 3)[:, :, :20, :]
    norms = np.linalg.norm(bones, axis=-1)
    valid = norms > EPSILON
    units = np.divide(bones, norms[..., None], out=np.zeros_like(bones), where=valid[..., None])
    dot = np.sum(units[1:] * units[:-1], axis=-1)
    pair_valid = valid[1:] & valid[:-1]
    result = np.zeros((30, 40), dtype=np.float32)
    result[1:] = np.where(pair_valid, np.arccos(np.clip(dot, -1.0, 1.0)), 0.0).reshape(29, 40)
    if not np.isfinite(result).all() or np.any(result < 0.0) or np.any(result > np.pi + 1e-6):
        raise AssertionError("Dinámica angular inválida")
    return result


def transform_sequence(sequence: np.ndarray) -> np.ndarray:
    """Concatena el prefijo bone_vector126 inalterado y 40 ángulos factuales."""
    values = np.asarray(sequence, dtype=np.float32)
    angular = angular_dynamics(values)
    output = np.concatenate((values, angular), axis=1).astype(np.float32, copy=False)
    if output.shape != (30, 166) or not np.isfinite(output).all() or not np.array_equal(output[:, :126], values):
        raise AssertionError("Contrato bone_angular166 inválido")
    return output