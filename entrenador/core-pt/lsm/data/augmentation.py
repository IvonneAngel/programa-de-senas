
"""Stub for lsm.data.augmentation - only needed for specialized tasks, plain successor uses none."""
import numpy as np

def append_w33_original_duration(arr, duration): return arr
def append_w33_relative_time_coordinates(arr): return arr
def canonicalize_w33_dominant_hand(arr): return arr
def canonicalize_w33_velocity_magnitude(arr): return arr
def energy_density_reparameterize_w33(arr, mass): return arr
def hand_branch_structural_dropout(arr, prob): return arr
def load_w63_train_feature_stats(): return (np.zeros(352), np.ones(352))
def monotonic_time_warp(arr, power, base_feature_dim=226, derivative_feature_dim=126): return arr
def rotate_w33_inplane(arr, angle): return arr
def standardize_w33_featurewise(arr, mean, std): return (arr - mean) / std
