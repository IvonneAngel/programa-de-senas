
"""Stub for successor augmentation."""
import numpy as np
def augment_successor_positions126(arr, seed=0):
    # simple identity or tiny noise - for train_augmented task we could add jitter
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.01, arr.shape).astype(np.float32)
    return arr + noise
