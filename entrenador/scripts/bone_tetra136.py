"""Firma tetraédrica local de dedos para complementar bone_vector126."""
from __future__ import annotations

import numpy as np


EPSILON = 1e-6
FINGERS = np.asarray(((1, 2, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)), dtype=np.int64)
MCP = np.asarray((5, 9, 13, 17), dtype=np.int64)


def tetra_volumes(sequence: np.ndarray) -> np.ndarray:
    """Devuelve `(30,10)` con cinco volúmenes firmados por mano y ceros factuales."""
    values = np.asarray(sequence, dtype=np.float32)
    if values.shape != (30, 126) or not np.isfinite(values).all():
        raise ValueError(f"positions126 inválido: {values.shape}")
    output = np.zeros((30, 10), dtype=np.float32)
    for frame_index, frame in enumerate(values):
        for hand_index, flat in enumerate((frame[:63], frame[63:])):
            hand = flat.reshape(21, 3)
            if not hand.any():
                continue
            scale = np.linalg.norm(hand[MCP], axis=1).mean()
            if not np.isfinite(scale) or scale <= EPSILON:
                continue
            output[frame_index, hand_index * 5:(hand_index + 1) * 5] = np.linalg.det(hand[FINGERS] / scale).astype(np.float32)
    if not np.isfinite(output).all():
        raise AssertionError("Volúmenes tetraédricos no finitos")
    return output


def transform_sequence(bones: np.ndarray, positions: np.ndarray) -> np.ndarray:
    bone_values = np.asarray(bones, dtype=np.float32)
    if bone_values.shape != (30, 126) or not np.isfinite(bone_values).all():
        raise ValueError(f"bone_vector126 inválido: {bone_values.shape}")
    output = np.concatenate((bone_values, tetra_volumes(positions)), axis=1).astype(np.float32, copy=False)
    if output.shape != (30, 136) or not np.isfinite(output).all() or not np.array_equal(output[:, :126], bone_values):
        raise AssertionError("Contrato bone_tetra136 inválido")
    return output