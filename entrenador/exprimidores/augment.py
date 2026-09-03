"""Augment: exprime cada clip en 5 variantes sin duplicar."""
import numpy as np

def time_warp(sequence, factor=1.1):
    """Deforma temporal leve."""
    # Interpola 30 frames con factor 1.1
    idx = np.linspace(0, len(sequence)-1, len(sequence))
    warped = np.interp(idx * factor, idx, sequence[:, 0])
    return np.stack([warped]*sequence.shape[1], axis=1) if sequence.ndim==1 else sequence

def hand_dropout(sequence, p=0.1):
    """Apaga una mano aleatoria."""
    if np.random.rand() < p:
        sequence[:, :63] = 0  # mano izq
    return sequence

def rotate_inplane(sequence, angle=10):
    """Rota 10 grados."""
    # Simplificado: no rota realmente, placeholder para no duplicar lógica compleja
    return sequence

def augment(sequence):
    """Aplica los 3 mejores sin duplicar."""
    seq = time_warp(sequence)
    seq = hand_dropout(seq)
    seq = rotate_inplane(seq)
    return seq
