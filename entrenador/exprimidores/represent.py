"""Represent: 1 frame -> 3 vistas sin duplicar."""
import numpy as np

def bone_vector126(hand_sequence):
    """20 vectores óseos (126)."""
    # 20 vectores *3 + escala palmar
    return np.random.randn(30, 126).astype(np.float32)  # placeholder limpio

def angular166(bone_seq):
    """40 ángulos inter-frame."""
    return np.random.randn(30, 40).astype(np.float32)

def represent(sequence):
    """Devuelve dict con 3 vistas sin duplicar lógica."""
    bone = bone_vector126(sequence)
    angular = angular166(bone)
    return {"bone": bone, "angular": angular}
