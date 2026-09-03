"""Marco palmar determinista para una mano `(21,3)` con muñeca ya centrada."""
from __future__ import annotations

import numpy as np


EPSILON = 1e-6


def palm_basis(hand: np.ndarray) -> np.ndarray | None:
    """Devuelve una base derecha columna a columna: medio, índice y normal."""
    if hand.shape != (21, 3) or not np.isfinite(hand).all() or not hand.any():
        return None
    forward = hand[9]
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= EPSILON:
        return None
    first = forward / forward_norm
    lateral_raw = hand[5] - np.dot(hand[5], first) * first
    lateral_norm = float(np.linalg.norm(lateral_raw))
    if lateral_norm <= EPSILON:
        return None
    second = lateral_raw / lateral_norm
    third = np.cross(first, second)
    third_norm = float(np.linalg.norm(third))
    if third_norm <= EPSILON:
        return None
    return np.stack((first, second, third / third_norm), axis=1).astype(np.float32)


def rotation_vector(matrix: np.ndarray) -> np.ndarray:
    """Convierte una matriz SO(3) a vector eje-ángulo sin dependencias externas."""
    if matrix.shape != (3, 3):
        raise ValueError(f"Se esperaba rotación 3x3, no {matrix.shape}")
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle <= EPSILON:
        return np.zeros(3, dtype=np.float32)
    if np.pi - angle <= 1e-4:
        eigenvalues, eigenvectors = np.linalg.eigh((matrix + np.eye(3, dtype=np.float32)) * 0.5)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        skew = np.asarray((matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]))
        if float(np.dot(axis, skew)) < 0.0:
            axis = -axis
    else:
        sine = 2.0 * float(np.sin(angle))
        axis = np.asarray((matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1])) / sine
    norm = float(np.linalg.norm(axis))
    if norm <= EPSILON or not np.isfinite(axis).all():
        raise ValueError("No se pudo obtener eje de rotación finito")
    return (axis / norm * angle).astype(np.float32)


def transform_hand(hand: np.ndarray) -> np.ndarray:
    """Canoniza dedos en la base palmar y reserva la muñeca para orientación explícita."""
    frame = np.asarray(hand, dtype=np.float32)
    basis = palm_basis(frame)
    if basis is None:
        return np.zeros((21, 3), dtype=np.float32)
    canonical = frame @ basis
    canonical[0] = rotation_vector(basis)
    if not np.isfinite(canonical).all():
        raise ValueError("Marco palmar no finito")
    return canonical.astype(np.float32, copy=False)


def transform_sequence(sequence: np.ndarray) -> np.ndarray:
    """Transforma posiciones126 a `(30,126)` conservando slots izquierda/derecha y ceros."""
    values = np.asarray(sequence, dtype=np.float32)
    if values.shape != (30, 126) or not np.isfinite(values).all():
        raise ValueError(f"Secuencia inválida: {values.shape}")
    output = np.empty_like(values)
    for index, frame in enumerate(values):
        output[index, :63] = transform_hand(frame[:63].reshape(21, 3)).reshape(63)
        output[index, 63:] = transform_hand(frame[63:].reshape(21, 3)).reshape(63)
    if output.shape != (30, 126) or not np.isfinite(output).all():
        raise AssertionError("Contrato palm_frame126 inválido")
    return output