"""Entrenador PyTorch v2 para features LSM ya cacheadas."""
from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

from lsm.data.augmentation import append_w33_original_duration, append_w33_relative_time_coordinates, canonicalize_w33_dominant_hand, canonicalize_w33_velocity_magnitude, energy_density_reparameterize_w33, hand_branch_structural_dropout, load_w63_train_feature_stats, monotonic_time_warp, rotate_w33_inplane, standardize_w33_featurewise
from lsm.data.successor_augmentation import augment_successor_positions126
from lsm.models.tcn import build_model, parameter_count

try:
    from .train_helpers import setup_training, run_epoch_split, finalize_training
    from .train_helpers import run_epoch_forward, run_epoch_loss, run_epoch_backward
except ImportError:
    try:
        from train_helpers import setup_training, run_epoch_split, finalize_training
        from train_helpers import run_epoch_forward, run_epoch_loss, run_epoch_backward
    except ImportError:
        pass


TASK_CONFIG = {
    "static_letter": {"frames": 1, "features": 93},
    "dynamic_letter": {"frames": 30, "features": 93},
    "recovery_positions": {"frames": 30, "features": 126},
    "successor_positions126": {"frames": 30, "features": 126},
    "successor_positions126_train_augmented": {"frames": 30, "features": 126},
    "successor_episodic_real_reference": {"frames": 30, "features": 126},
    "successor_temporal_relation_pairs": {"frames": 30, "features": 126},
    "successor_selective_core_relation": {"frames": 30, "features": 126},
    "successor_intramanual_bone166": {"frames": 30, "features": 166},
    "successor_intramanual_kinematic196": {"frames": 30, "features": 196},
    "successor_signer_stratified_batch": {"frames": 30, "features": 126},
    "successor_soft_presence_weight": {"frames": 30, "features": 126},
    "successor_masked_hand_reconstruction": {"frames": 30, "features": 126},
    "successor_intraclip_style_normalization": {"frames": 30, "features": 126},
    "successor_temporal_pyramid_pooling": {"frames": 30, "features": 126},
    "successor_logavgexp_temporal_pooling": {"frames": 30, "features": 126},
    "successor_uniform_label_smoothing": {"frames": 30, "features": 126},
    "successor_train_only_swa": {"frames": 30, "features": 126},
    "successor_shared_bilateral_tcn": {"frames": 30, "features": 126},
    "successor_ecoc_auxiliary_head": {"frames": 30, "features": 126},
    "successor_signer_vrex": {"frames": 30, "features": 126},
    "successor_fixed_hand_graph_tcn": {"frames": 30, "features": 126},
    "successor_arc_length_frame_reindexing": {"frames": 30, "features": 126},
    "successor_bidirectional_gru": {"frames": 30, "features": 126},
    "successor_cosine_classifier": {"frames": 30, "features": 126},
    "successor_spectral_tcn": {"frames": 30, "features": 126},
    "successor_global_wrist132": {"frames": 30, "features": 132},
    "successor_wrist_velocity132": {"frames": 30, "features": 132},
    "recovery_path_signature": {"frames": 30, "features": 366},
    "dynamic_alphabet_zenodo": {"frames": 30, "features": 22},
    "ickmejia_jkq": {"frames": 20, "features": 201},
    "isolated_word": {"frames": 30, "features": 226},
    "isolated_word_chronological": {"frames": 30, "features": 226},
    "isolated_word_residual_chronological": {"frames": 30, "features": 226},
    "isolated_word_zero_init_residual": {"frames": 30, "features": 226},
    "isolated_word_signer_invariant": {"frames": 30, "features": 226},
    "isolated_word_trajectory_residual": {"frames": 30, "features": 364},
    "isolated_word_kinematic_channel_dropout": {"frames": 30, "features": 478},
    "isolated_word_temporal_prediction_consistency": {"frames": 30, "features": 352},
    "isolated_word_cross_signer_feature_mixup": {"frames": 30, "features": 352},
    "isolated_word_cross_signer_temporal_consistency": {"frames": 30, "features": 352},
    "isolated_word_contrastive_soft_dtw_alignment": {"frames": 30, "features": 352},
    "isolated_word_hopfield_prototype_memory": {"frames": 30, "features": 352},
    "isolated_word_log_euclidean_covariance_consistency": {"frames": 30, "features": 352},
    "isolated_word_uncertainty_balanced_covariance": {"frames": 30, "features": 352},
    "isolated_word_sharpness_aware_w3": {"frames": 30, "features": 352},
    "isolated_word_group_dro_signer": {"frames": 30, "features": 352},
    "isolated_word_masked_temporal_pretraining": {"frames": 30, "features": 352},
    "isolated_word_pretrained_temporal_consistency": {"frames": 30, "features": 352},
    "isolated_word_temporal_order_pretraining": {"frames": 30, "features": 352},
    "isolated_word_mean_teacher_temporal_consistency": {"frames": 30, "features": 352},
    "isolated_word_position_velocity_representation_consistency": {"frames": 30, "features": 352},
    "isolated_word_ema_weight_average_inference": {"frames": 30, "features": 352},
    "isolated_word_ldam_deferred_reweighting": {"frames": 30, "features": 352},
    "isolated_word_manual_factorial_ldam": {"frames": 30, "features": 352},
    "isolated_word_body_anchor_residual_ldam": {"frames": 30, "features": 403},
    "isolated_word_quality_gated_body_anchor_ldam": {"frames": 30, "features": 403},
    "isolated_word_palm_axis_residual_ldam": {"frames": 30, "features": 436},
    "isolated_word_dense_hand_spectral_signature_ldam": {"frames": 30, "features": 392},
    "isolated_word_latent_style_mix_ldam": {"frames": 30, "features": 352},
    "isolated_word_stochastic_dropout_consistency_ldam": {"frames": 30, "features": 352},
    "isolated_word_canonical_motion_vat_ldam": {"frames": 30, "features": 352},
    "isolated_word_motion_biased_attentive_pooling_ldam": {"frames": 30, "features": 352},
    "isolated_word_multiscale_temporal_difference_residual_ldam": {"frames": 30, "features": 352},
    "isolated_word_dct_spectral_residual_ldam": {"frames": 30, "features": 352},
    "isolated_word_cosine_classifier_ldam": {"frames": 30, "features": 352},
    "isolated_word_train_prior_logit_adjusted_ldam": {"frames": 30, "features": 352},
    "isolated_word_fixed_hand_graph_residual_ldam": {"frames": 30, "features": 352},
    "isolated_word_classifier_coherence_ldam": {"frames": 30, "features": 352},
    "isolated_word_focal_ldam_deferred_reweighting": {"frames": 30, "features": 352},
    "isolated_word_class_balanced_sampling_ldam": {"frames": 30, "features": 352},
    "isolated_word_energy_phase_residual_ldam": {"frames": 30, "features": 352},
    "isolated_word_channel_recalibration_ldam": {"frames": 30, "features": 352},
    "isolated_word_hand_branch_structural_dropout_ldam": {"frames": 30, "features": 352},
    "isolated_word_dominant_hand_canonicalization_ldam": {"frames": 30, "features": 352},
    "isolated_word_cross_signer_supervised_contrast_ldam": {"frames": 30, "features": 352},
    "isolated_word_global_inplane_rotation_ldam": {"frames": 30, "features": 352},
    "isolated_word_decoupled_classifier_retraining_ldam": {"frames": 30, "features": 352},
    "isolated_word_velocity_magnitude_canonicalization_ldam": {"frames": 30, "features": 353},
    "isolated_word_original_duration_aware_ldam": {"frames": 30, "features": 353},
    "isolated_word_relative_time_coordinates_ldam": {"frames": 30, "features": 354},
    "isolated_word_train_only_feature_standardization_ldam": {"frames": 30, "features": 352},
    "isolated_word_parallel_receptive_field_tcn_ldam": {"frames": 30, "features": 352},
    "isolated_word_parameter_free_temporal_shift_ldam": {"frames": 30, "features": 352},
    "isolated_word_temporal_weight_standardization_ldam": {"frames": 30, "features": 352},
    "isolated_word_depthwise_separable_temporal_tcn_ldam": {"frames": 30, "features": 352},
    "isolated_word_linear_stochastic_depth_ldam": {"frames": 30, "features": 352},
    "isolated_word_rezero_temporal_residual_ldam": {"frames": 30, "features": 352},
    "isolated_word_energy_density_reparameterized_ldam": {"frames": 30, "features": 352},
    "isolated_word_confusion_spectral_ldam": {"frames": 30, "features": 352},
    "isolated_word_adaptive_temporal_prototype_ldam": {"frames": 30, "features": 352},
    "isolated_word_compositional_temporal_prototype_ldam": {"frames": 30, "features": 352},
    "isolated_word_shape_motion_bilinear_ldam": {"frames": 30, "features": 352},
    "isolated_word_signer_covariance_alignment_ldam": {"frames": 30, "features": 352},
    "isolated_word_w80_interior_hand_reconstruction_ldam": {"frames": 30, "features": 352},
    "isolated_word_w81_activity_boundary_ldam": {"frames": 30, "features": 352},
    "isolated_word_w83_world_hand_geometry_residual_ldam": {"frames": 30, "features": 478},
    "isolated_word_w84_explicit_hand_presence_ldam": {"frames": 30, "features": 354},
    "isolated_word_w85_train_signer_group_dro_ldam": {"frames": 30, "features": 352},
    "isolated_word_w89_class_conditional_hand_quality_curriculum_ldam": {"frames": 30, "features": 352},
    "isolated_word_w93_motion_adaptive_temporal_coherence_ldam": {"frames": 30, "features": 352},
    "isolated_word_path_signature_early_fusion_ldam": {"frames": 30, "features": 424},
    "isolated_word_dual_view_residual": {"frames": 30, "features": 704},
    "isolated_word_gated_dual_view_residual": {"frames": 30, "features": 704},
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_manifest(path: Path, task: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    split_column = "split_model" if any(row.get("split_model") for row in rows) else "split_project"
    accepted = {"train", "validation", "test"}
    output = []
    manifest_task = "isolated_word" if task in {"isolated_word_chronological", "isolated_word_residual_chronological", "isolated_word_zero_init_residual", "isolated_word_signer_invariant", "isolated_word_trajectory_residual", "isolated_word_kinematic_channel_dropout", "isolated_word_temporal_prediction_consistency", "isolated_word_cross_signer_feature_mixup", "isolated_word_cross_signer_temporal_consistency", "isolated_word_contrastive_soft_dtw_alignment", "isolated_word_hopfield_prototype_memory", "isolated_word_log_euclidean_covariance_consistency", "isolated_word_uncertainty_balanced_covariance", "isolated_word_sharpness_aware_w3", "isolated_word_group_dro_signer", "isolated_word_masked_temporal_pretraining", "isolated_word_pretrained_temporal_consistency", "isolated_word_temporal_order_pretraining", "isolated_word_mean_teacher_temporal_consistency", "isolated_word_ema_weight_average_inference", "isolated_word_position_velocity_representation_consistency", "isolated_word_ldam_deferred_reweighting", "isolated_word_manual_factorial_ldam", "isolated_word_body_anchor_residual_ldam", "isolated_word_quality_gated_body_anchor_ldam", "isolated_word_palm_axis_residual_ldam", "isolated_word_latent_style_mix_ldam", "isolated_word_stochastic_dropout_consistency_ldam", "isolated_word_canonical_motion_vat_ldam", "isolated_word_motion_biased_attentive_pooling_ldam", "isolated_word_energy_phase_residual_ldam", "isolated_word_channel_recalibration_ldam", "isolated_word_hand_branch_structural_dropout_ldam", "isolated_word_dominant_hand_canonicalization_ldam", "isolated_word_multiscale_temporal_difference_residual_ldam", "isolated_word_dct_spectral_residual_ldam", "isolated_word_cosine_classifier_ldam", "isolated_word_train_prior_logit_adjusted_ldam", "isolated_word_fixed_hand_graph_residual_ldam", "isolated_word_classifier_coherence_ldam", "isolated_word_focal_ldam_deferred_reweighting", "isolated_word_class_balanced_sampling_ldam", "isolated_word_path_signature_early_fusion_ldam", "isolated_word_dual_view_residual", "isolated_word_gated_dual_view_residual"} else task
    if task in {"isolated_word_cross_signer_supervised_contrast_ldam", "isolated_word_global_inplane_rotation_ldam", "isolated_word_decoupled_classifier_retraining_ldam", "isolated_word_velocity_magnitude_canonicalization_ldam", "isolated_word_original_duration_aware_ldam", "isolated_word_relative_time_coordinates_ldam", "isolated_word_train_only_feature_standardization_ldam", "isolated_word_parallel_receptive_field_tcn_ldam", "isolated_word_parameter_free_temporal_shift_ldam", "isolated_word_depthwise_separable_temporal_tcn_ldam", "isolated_word_linear_stochastic_depth_ldam", "isolated_word_rezero_temporal_residual_ldam", "isolated_word_energy_density_reparameterized_ldam", "isolated_word_confusion_spectral_ldam", "isolated_word_adaptive_temporal_prototype_ldam", "isolated_word_compositional_temporal_prototype_ldam", "isolated_word_shape_motion_bilinear_ldam", "isolated_word_dense_hand_spectral_signature_ldam", "isolated_word_w80_interior_hand_reconstruction_ldam", "isolated_word_signer_covariance_alignment_ldam"}:
        manifest_task = "isolated_word"
    if task == "isolated_word_w81_activity_boundary_ldam":
        manifest_task = "isolated_word"
    if task == "isolated_word_w83_world_hand_geometry_residual_ldam":
        manifest_task = "isolated_word"
    if task == "isolated_word_w84_explicit_hand_presence_ldam":
        manifest_task = "isolated_word"
    if task == "isolated_word_w85_train_signer_group_dro_ldam":
        manifest_task = "isolated_word"
    if task == "isolated_word_w89_class_conditional_hand_quality_curriculum_ldam":
        manifest_task = "isolated_word"
    if task == "isolated_word_w93_motion_adaptive_temporal_coherence_ldam":
        manifest_task = "isolated_word"
    if task == "successor_positions126_train_augmented":
        manifest_task = "successor_positions126"
    if task == "successor_episodic_real_reference":
        manifest_task = "successor_positions126"
    if task == "successor_temporal_relation_pairs":
        manifest_task = "successor_positions126"
    if task == "successor_selective_core_relation":
        manifest_task = "successor_positions126"
    if task == "successor_signer_stratified_batch":
        manifest_task = "successor_positions126"
    if task == "successor_soft_presence_weight":
        manifest_task = "successor_positions126"
    if task == "successor_masked_hand_reconstruction":
        manifest_task = "successor_positions126"
    if task == "successor_intraclip_style_normalization":
        manifest_task = "successor_positions126"
    if task == "successor_temporal_pyramid_pooling":
        manifest_task = "successor_positions126"
    if task == "successor_logavgexp_temporal_pooling":
        manifest_task = "successor_positions126"
    if task == "successor_uniform_label_smoothing":
        manifest_task = "successor_positions126"
    if task == "successor_train_only_swa":
        manifest_task = "successor_positions126"
    if task == "successor_shared_bilateral_tcn":
        manifest_task = "successor_positions126"
    if task == "successor_ecoc_auxiliary_head":
        manifest_task = "successor_positions126"
    if task == "successor_signer_vrex":
        manifest_task = "successor_positions126"
    if task == "successor_fixed_hand_graph_tcn":
        manifest_task = "successor_positions126"
    if task == "successor_arc_length_frame_reindexing":
        manifest_task = "successor_positions126"
    if task == "successor_bidirectional_gru":
        manifest_task = "successor_positions126"
    if task == "successor_cosine_classifier":
        manifest_task = "successor_positions126"
    if task == "successor_spectral_tcn":
        manifest_task = "successor_positions126"
    for row in rows:
        if row.get("task") != manifest_task or row.get("feature_status") not in {"", "ok"}:
            continue
        active_split = row.get(split_column, "")
        if active_split not in accepted:
            continue
        row["split_active"] = active_split
        output.append(row)
    return output


def feature_path(row: dict, cache_root: Path) -> Path:
    explicit = str(row.get("feature_path", "")).strip()
    if explicit:
        return cache_root / explicit
    return cache_root / f"{row['sample_id']}.npy"


def load_array(path: Path, expected: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.load(path, allow_pickle=False).astype(np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if tuple(array.shape) != expected:
        raise ValueError(f"{path}: se esperaba {expected}, se recibió {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: contiene NaN o Inf")
    return array


class CachedSequenceDataset(Dataset):
    def __init__(self, rows: list[dict], cache_root: Path, labels: dict[str, int], expected: tuple[int, int], temporal_warp_range: tuple[float, float] | None = None, signer_labels: dict[str, int] | None = None, temporal_consistency_range: tuple[float, float] | None = None, cross_signer_pairing: bool = False, combined_cross_signer_consistency: bool = False, alignment_triplets: bool = False, covariance_pairing: bool = False, hand_branch_dropout_probability: float | None = None, dominant_hand_canonicalization: bool = False, global_inplane_rotation_range: tuple[float, float] | None = None, velocity_magnitude_canonicalization: bool = False, original_duration_aware: bool = False, relative_time_coordinates: bool = False, feature_standardization_stats: tuple[np.ndarray, np.ndarray] | None = None, energy_density_uniform_mass: float | None = None, successor_train_augmentation: bool = False, augmentation_seed: int = 0, successor_relation_pairing: bool = False, soft_presence_weight: bool = False):
        self.rows = rows
        self.cache_root = cache_root
        self.labels = labels
        self.expected = expected
        self.temporal_warp_range = temporal_warp_range
        self.signer_labels = signer_labels
        self.temporal_consistency_range = temporal_consistency_range
        self.cross_signer_pairing = cross_signer_pairing
        self.combined_cross_signer_consistency = combined_cross_signer_consistency
        self.alignment_triplets = alignment_triplets
        self.covariance_pairing = covariance_pairing
        self.hand_branch_dropout_probability = hand_branch_dropout_probability
        self.dominant_hand_canonicalization = dominant_hand_canonicalization
        self.global_inplane_rotation_range = global_inplane_rotation_range
        self.velocity_magnitude_canonicalization = velocity_magnitude_canonicalization
        self.original_duration_aware = original_duration_aware
        self.relative_time_coordinates = relative_time_coordinates
        self.feature_standardization_stats = feature_standardization_stats
        self.energy_density_uniform_mass = energy_density_uniform_mass
        self.successor_train_augmentation = successor_train_augmentation
        self.augmentation_seed = int(augmentation_seed)
        self.augmentation_epoch = 0
        self.successor_relation_pairing = successor_relation_pairing
        self.soft_presence_weight = soft_presence_weight
        self.cache_expected = (30, 352) if velocity_magnitude_canonicalization or original_duration_aware or relative_time_coordinates else expected
        if temporal_consistency_range is not None and expected != (30, 352):
            raise ValueError("La consistencia temporal requiere secuencias W3 de (30,352)")
        if energy_density_uniform_mass is not None:
            if expected != (30, 352) or not 0.0 <= float(energy_density_uniform_mass) <= 1.0:
                raise ValueError("W71 requiere secuencias W33 (30,352) y cuota uniforme en [0,1]")
            if any(value is not None for value in (temporal_warp_range, temporal_consistency_range, hand_branch_dropout_probability, global_inplane_rotation_range, feature_standardization_stats)) or cross_signer_pairing or combined_cross_signer_consistency or alignment_triplets or covariance_pairing or signer_labels is not None or dominant_hand_canonicalization or velocity_magnitude_canonicalization or original_duration_aware or relative_time_coordinates:
                raise ValueError("W71 no se combina con otras transformaciones, pares o pérdidas auxiliares")
        if temporal_consistency_range is not None and signer_labels is not None:
            raise ValueError("No se combinan consistencia temporal y etiquetas adversarias en el mismo experimento")
        if cross_signer_pairing and expected != (30, 352):
            raise ValueError("La mezcla cross-signer requiere secuencias W3 de (30,352)")
        if combined_cross_signer_consistency and (not cross_signer_pairing or temporal_consistency_range is None):
            raise ValueError("W20 requiere pareja cross-signer y consistencia temporal a la vez")
        if combined_cross_signer_consistency and expected != (30, 352):
            raise ValueError("W20 requiere secuencias W3 de (30,352)")
        if cross_signer_pairing and (signer_labels is not None or temporal_warp_range is not None or (temporal_consistency_range is not None and not combined_cross_signer_consistency)):
            raise ValueError("La mezcla cross-signer no se combina con otras augmentaciones o pérdidas auxiliares")
        if alignment_triplets and expected != (30, 352):
            raise ValueError("W21 requiere secuencias W3 de (30,352)")
        if alignment_triplets and (cross_signer_pairing or combined_cross_signer_consistency or temporal_consistency_range is not None or signer_labels is not None or temporal_warp_range is not None):
            raise ValueError("W21 no se combina con otras augmentaciones, pares o pérdidas auxiliares")
        if covariance_pairing and expected != (30, 352):
            raise ValueError("W23 requiere secuencias W3 de (30,352)")
        if covariance_pairing and (cross_signer_pairing or combined_cross_signer_consistency or alignment_triplets or temporal_consistency_range is not None or signer_labels is not None or temporal_warp_range is not None):
            raise ValueError("W23 no se combina con otras augmentaciones, pares o pérdidas auxiliares")
        if hand_branch_dropout_probability is not None:
            if expected != (30, 352) or not 0.0 <= hand_branch_dropout_probability <= 1.0:
                raise ValueError("W55 requiere secuencias W33 (30,352) y probabilidad en [0,1]")
            if temporal_warp_range is not None or temporal_consistency_range is not None or cross_signer_pairing or combined_cross_signer_consistency or alignment_triplets or covariance_pairing or signer_labels is not None:
                raise ValueError("W55 no se combina con otras augmentaciones, pares o pérdidas auxiliares")
        if dominant_hand_canonicalization:
            if expected != (30, 352):
                raise ValueError("W56 requiere secuencias W33 (30,352)")
            if temporal_warp_range is not None or temporal_consistency_range is not None or cross_signer_pairing or combined_cross_signer_consistency or alignment_triplets or covariance_pairing or signer_labels is not None or hand_branch_dropout_probability is not None:
                raise ValueError("W56 no se combina con otras augmentaciones, pares o pérdidas auxiliares")
        if global_inplane_rotation_range is not None:
            low, high = global_inplane_rotation_range
            if expected != (30, 352) or not np.isfinite([low, high]).all() or low > high:
                raise ValueError("W58 requiere W33 (30,352) y rango de rotación finito ordenado")
            if temporal_warp_range is not None or temporal_consistency_range is not None or cross_signer_pairing or combined_cross_signer_consistency or alignment_triplets or covariance_pairing or signer_labels is not None or hand_branch_dropout_probability is not None or dominant_hand_canonicalization:
                raise ValueError("W58 no se combina con otras augmentaciones, pares o pérdidas auxiliares")
        if velocity_magnitude_canonicalization:
            if expected != (30, 353):
                raise ValueError("W60 produce secuencias (30,353) desde caché W33")
            if temporal_warp_range is not None or temporal_consistency_range is not None or cross_signer_pairing or combined_cross_signer_consistency or alignment_triplets or covariance_pairing or signer_labels is not None or hand_branch_dropout_probability is not None or dominant_hand_canonicalization or global_inplane_rotation_range is not None:
                raise ValueError("W60 no se combina con augmentaciones, pares o pérdidas auxiliares")
        if original_duration_aware:
            if expected != (30, 353):
                raise ValueError("W61 produce secuencias (30,353) desde caché W33")
            if temporal_warp_range is not None or temporal_consistency_range is not None or cross_signer_pairing or combined_cross_signer_consistency or alignment_triplets or covariance_pairing or signer_labels is not None or hand_branch_dropout_probability is not None or dominant_hand_canonicalization or global_inplane_rotation_range is not None or velocity_magnitude_canonicalization:
                raise ValueError("W61 no se combina con augmentaciones, pares o pérdidas auxiliares")
        if relative_time_coordinates:
            if expected != (30, 354):
                raise ValueError("W62 produce secuencias (30,354) desde caché W33")
            if temporal_warp_range is not None or temporal_consistency_range is not None or cross_signer_pairing or combined_cross_signer_consistency or alignment_triplets or covariance_pairing or signer_labels is not None or hand_branch_dropout_probability is not None or dominant_hand_canonicalization or global_inplane_rotation_range is not None or velocity_magnitude_canonicalization or original_duration_aware:
                raise ValueError("W62 no se combina con augmentaciones, pares o pérdidas auxiliares")
        if feature_standardization_stats is not None:
            mean, std = feature_standardization_stats
            if expected != (30, 352):
                raise ValueError("W63 conserva secuencias W33 (30,352)")
            standardize_w33_featurewise(np.zeros((30, 352), dtype=np.float32), mean, std)
            if temporal_warp_range is not None or temporal_consistency_range is not None or cross_signer_pairing or combined_cross_signer_consistency or alignment_triplets or covariance_pairing or signer_labels is not None or hand_branch_dropout_probability is not None or dominant_hand_canonicalization or global_inplane_rotation_range is not None or velocity_magnitude_canonicalization or original_duration_aware or relative_time_coordinates:
                raise ValueError("W63 no se combina con augmentaciones, pares o pérdidas auxiliares")
        if successor_train_augmentation:
            if expected != (30, 126):
                raise ValueError("El aumento sucesor requiere secuencias (30,126)")
            if any(value is not None for value in (temporal_warp_range, temporal_consistency_range, hand_branch_dropout_probability, global_inplane_rotation_range, feature_standardization_stats, energy_density_uniform_mass)) or cross_signer_pairing or combined_cross_signer_consistency or alignment_triplets or covariance_pairing or signer_labels is not None or dominant_hand_canonicalization or velocity_magnitude_canonicalization or original_duration_aware or relative_time_coordinates:
                raise ValueError("El aumento sucesor no se combina con otras rutas de aumento o pérdidas auxiliares")
        if successor_relation_pairing:
            if expected != (30, 126):
                raise ValueError("Los pares relacionales sucesores requieren secuencias (30,126)")
            if any(value is not None for value in (temporal_warp_range, temporal_consistency_range, hand_branch_dropout_probability, global_inplane_rotation_range, feature_standardization_stats, energy_density_uniform_mass)) or cross_signer_pairing or combined_cross_signer_consistency or alignment_triplets or covariance_pairing or signer_labels is not None or dominant_hand_canonicalization or velocity_magnitude_canonicalization or original_duration_aware or relative_time_coordinates or successor_train_augmentation:
                raise ValueError("Los pares relacionales sucesores no se combinan con otras rutas auxiliares")
        self.cross_signer_candidates: dict[int, list[int]] = {}
        if cross_signer_pairing or covariance_pairing:
            for index, row in enumerate(rows):
                label, signer = row.get("label_lsm", ""), row.get("signer_id", "")
                candidates = [
                    peer_index
                    for peer_index, peer in enumerate(rows)
                    if peer.get("label_lsm") == label and peer.get("signer_id") != signer
                ]
                if not candidates:
                    raise ValueError(f"W19 no tiene pareja de firmante distinto para {label!r} / {signer!r}")
                self.cross_signer_candidates[index] = candidates
        self.alignment_positive_candidates: dict[int, list[int]] = {}
        self.alignment_negative_candidates: dict[int, list[int]] = {}
        if alignment_triplets:
            for index, row in enumerate(rows):
                label, signer = row.get("label_lsm", ""), row.get("signer_id", "")
                positives = [
                    peer_index
                    for peer_index, peer in enumerate(rows)
                    if peer.get("label_lsm") == label and peer.get("signer_id") != signer
                ]
                negatives = [
                    peer_index
                    for peer_index, peer in enumerate(rows)
                    if peer.get("label_lsm") != label and peer.get("signer_id") == signer
                ]
                if not positives or not negatives:
                    raise ValueError(f"W21 no tiene triplete válido para {label!r} / {signer!r}")
                self.alignment_positive_candidates[index] = positives
                self.alignment_negative_candidates[index] = negatives
        self.successor_relation_candidates: dict[int, list[int]] = {}
        if successor_relation_pairing:
            for index, row in enumerate(rows):
                candidates = [peer_index for peer_index, peer in enumerate(rows) if peer.get("label_lsm") == row.get("label_lsm") and peer.get("signer_id") != row.get("signer_id")]
                if not candidates:
                    raise ValueError(f"Sin pareja real cross-signer para {row.get('sample_id', index)!r}")
                self.successor_relation_candidates[index] = candidates

    def __len__(self) -> int:
        return len(self.rows)

    def set_augmentation_epoch(self, epoch: int) -> None:
        self.augmentation_epoch = int(epoch)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = feature_path(row, self.cache_root)
        array = load_array(path, self.cache_expected)
        if self.successor_train_augmentation:
            sample_seed = (self.augmentation_seed * 1_000_003 + self.augmentation_epoch * 10_007 + int(index)) & 0xFFFFFFFF
            array = augment_successor_positions126(array, seed=sample_seed)
        if self.temporal_warp_range is not None:
            low, high = self.temporal_warp_range
            power = float(np.random.uniform(low, high))
            array = monotonic_time_warp(array, power, base_feature_dim=226, derivative_feature_dim=126)
        if self.hand_branch_dropout_probability is not None:
            array = hand_branch_structural_dropout(array, self.hand_branch_dropout_probability)
        if self.dominant_hand_canonicalization:
            array = canonicalize_w33_dominant_hand(array)
        if self.global_inplane_rotation_range is not None:
            low, high = self.global_inplane_rotation_range
            array = rotate_w33_inplane(array, float(np.random.uniform(low, high)))
        if self.velocity_magnitude_canonicalization:
            array = canonicalize_w33_velocity_magnitude(array)
        if self.original_duration_aware:
            duration = row.get("frames_original") or row.get("frame_count") or 0
            array = append_w33_original_duration(array, float(duration))
        if self.relative_time_coordinates:
            array = append_w33_relative_time_coordinates(array)
        if self.feature_standardization_stats is not None:
            array = standardize_w33_featurewise(array, *self.feature_standardization_stats)
        if self.energy_density_uniform_mass is not None:
            array = energy_density_reparameterize_w33(array, self.energy_density_uniform_mass)
        label = self.labels[row["label_lsm"]]
        if self.successor_relation_pairing:
            peer_index = int(np.random.choice(self.successor_relation_candidates[index]))
            peer = self.rows[peer_index]
            if peer.get("label_lsm") != row.get("label_lsm") or peer.get("signer_id") == row.get("signer_id"):
                raise RuntimeError("La pareja relacional sucesora no conserva clase o cambia firmante")
            peer_array = load_array(feature_path(peer, self.cache_root), self.expected)
            return torch.from_numpy(array), torch.from_numpy(peer_array), torch.tensor(label, dtype=torch.long)
        if self.covariance_pairing:
            peer_index = int(np.random.choice(self.cross_signer_candidates[index]))
            peer = self.rows[peer_index]
            if peer.get("label_lsm") != row.get("label_lsm") or peer.get("signer_id") == row.get("signer_id"):
                raise RuntimeError("W23 seleccionó una pareja inválida")
            peer_array = load_array(feature_path(peer, self.cache_root), self.expected)
            return torch.from_numpy(array), torch.from_numpy(peer_array), torch.tensor(label, dtype=torch.long)
        if self.alignment_triplets:
            positive_index = int(np.random.choice(self.alignment_positive_candidates[index]))
            negative_index = int(np.random.choice(self.alignment_negative_candidates[index]))
            positive = self.rows[positive_index]
            negative = self.rows[negative_index]
            if positive.get("label_lsm") != row.get("label_lsm") or positive.get("signer_id") == row.get("signer_id"):
                raise RuntimeError("W21 seleccionó un positivo inválido")
            if negative.get("label_lsm") == row.get("label_lsm") or negative.get("signer_id") != row.get("signer_id"):
                raise RuntimeError("W21 seleccionó un negativo inválido")
            positive_array = load_array(feature_path(positive, self.cache_root), self.expected)
            negative_array = load_array(feature_path(negative, self.cache_root), self.expected)
            return torch.from_numpy(array), torch.from_numpy(positive_array), torch.from_numpy(negative_array), torch.tensor(label, dtype=torch.long)
        if self.cross_signer_pairing:
            peer_index = int(np.random.choice(self.cross_signer_candidates[index]))
            peer = self.rows[peer_index]
            if peer.get("label_lsm") != row.get("label_lsm") or peer.get("signer_id") == row.get("signer_id"):
                raise RuntimeError("W19 seleccionó una pareja inválida")
            peer_array = load_array(feature_path(peer, self.cache_root), self.expected)
            if self.combined_cross_signer_consistency:
                low, high = self.temporal_consistency_range
                power = float(np.random.uniform(low, high))
                augmented = monotonic_time_warp(array, power, base_feature_dim=226, derivative_feature_dim=126)
                return torch.from_numpy(array), torch.from_numpy(peer_array), torch.from_numpy(augmented), torch.tensor(label, dtype=torch.long)
            return torch.from_numpy(array), torch.from_numpy(peer_array), torch.tensor(label, dtype=torch.long)
        if self.temporal_consistency_range is not None:
            low, high = self.temporal_consistency_range
            power = float(np.random.uniform(low, high))
            augmented = monotonic_time_warp(array, power, base_feature_dim=226, derivative_feature_dim=126)
            return torch.from_numpy(array), torch.from_numpy(augmented), torch.tensor(label, dtype=torch.long)
        if self.signer_labels is not None:
            signer = row.get("signer_id", "")
            if signer not in self.signer_labels:
                raise ValueError(f"Firmante fuera de la adversaria de entrenamiento: {signer!r}")
            return torch.from_numpy(array), torch.tensor(label, dtype=torch.long), torch.tensor(self.signer_labels[signer], dtype=torch.long)
        if self.soft_presence_weight:
            presence = float(np.any(np.abs(array.reshape(30, 2, 21, 3)) > 1e-8, axis=(2, 3)).mean())
            return torch.from_numpy(array), torch.tensor(label, dtype=torch.long), torch.tensor(0.25 + 0.75 * presence, dtype=torch.float32)
        return torch.from_numpy(array), torch.tensor(label, dtype=torch.long)


class CrossSignerEpisodeBatchSampler(Sampler[list[int]]):
    """Lotes de tres clips reales de firmantes distintos por clase."""

    def __init__(self, rows: list[dict], classes_per_episode: int = 21, episodes_per_epoch: int = 23, seed: int = 0):
        if classes_per_episode < 2 or episodes_per_epoch < 1:
            raise ValueError("El sampler episódico requiere al menos dos clases y un episodio")
        self.classes_per_episode = int(classes_per_episode)
        self.episodes_per_epoch = int(episodes_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        self.by_class_signer: dict[str, dict[str, list[int]]] = {}
        for index, row in enumerate(rows):
            label, signer = row.get("label_lsm", ""), row.get("signer_id", "")
            if not label or not signer:
                raise ValueError("El sampler episódico requiere etiqueta y firmante por clip train")
            self.by_class_signer.setdefault(label, {}).setdefault(signer, []).append(index)
        self.eligible_classes = sorted(label for label, signers in self.by_class_signer.items() if len(signers) >= 3)
        if len(self.eligible_classes) < self.classes_per_episode:
            raise ValueError("No hay suficientes clases con tres firmantes train para un episodio")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed * 1_000_003 + self.epoch)
        for _ in range(self.episodes_per_epoch):
            episode_classes = rng.choice(self.eligible_classes, size=self.classes_per_episode, replace=False)
            batch: list[int] = []
            for label in episode_classes:
                signer_map = self.by_class_signer[str(label)]
                signers = rng.choice(sorted(signer_map), size=3, replace=False)
                batch.extend(int(rng.choice(signer_map[str(signer)])) for signer in signers)
            yield batch


class SignerStratifiedBatchSampler(Sampler[list[int]]):
    def __init__(self, rows: list[dict], per_signer: int = 9, batches: int = 23, seed: int = 0):
        self.groups: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            self.groups.setdefault(row["signer_id"], []).append(index)
        if len(self.groups) != 7 or any(not values for values in self.groups.values()):
            raise ValueError("El sampler sucesor requiere exactamente siete firmantes train no vacíos")
        self.per_signer, self.batches, self.seed, self.epoch = per_signer, batches, seed, 0
    def set_epoch(self, epoch: int) -> None: self.epoch = int(epoch)
    def __len__(self) -> int: return self.batches
    def __iter__(self):
        rng = np.random.default_rng(self.seed * 1_000_003 + self.epoch)
        for _ in range(self.batches):
            batch=[]
            for signer in sorted(self.groups): batch.extend(rng.choice(self.groups[signer], size=self.per_signer, replace=True).tolist())
            rng.shuffle(batch); yield batch


def class_labels(rows: list[dict]) -> list[str]:
    return sorted({row["label_lsm"] for row in rows if row.get("label_lsm")})


def metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, classes: int) -> dict:
    predictions = logits.argmax(dim=1).detach().cpu().numpy()
    truth = targets.detach().cpu().numpy()
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for actual, predicted in zip(truth, predictions):
        matrix[int(actual), int(predicted)] += 1
    per_class_f1 = []
    for index in range(classes):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        per_class_f1.append(2 * precision * recall / max(1e-12, precision + recall))
    return {
        "accuracy": float((predictions == truth).mean()) if len(truth) else 0.0,
        "macro_f1": float(np.mean(per_class_f1)) if per_class_f1 else 0.0,
        "support": int(len(truth)),
        "confusion_matrix": matrix.tolist(),
    }


def episodic_real_reference_loss(embeddings: torch.Tensor, targets: torch.Tensor, temperature: float = 0.20) -> torch.Tensor:
    """CE sobre consultas y prototipos construidos de soportes reales del episodio."""
    if embeddings.ndim != 2 or targets.ndim != 1 or embeddings.shape[0] != targets.numel():
        raise ValueError("La pérdida episódica requiere embeddings (batch,dim) y targets alineados")
    if temperature <= 0.0 or targets.numel() % 3 != 0:
        raise ValueError("La pérdida episódica requiere temperatura positiva y tres clips por clase")
    classes = targets.numel() // 3
    grouped_targets = targets.reshape(classes, 3)
    if not torch.equal(grouped_targets, grouped_targets[:, :1].expand_as(grouped_targets)):
        raise ValueError("Cada terna episódica debe compartir la misma clase")
    if torch.unique(grouped_targets[:, 0]).numel() != classes:
        raise ValueError("Cada episodio requiere clases distintas")
    grouped = nn.functional.normalize(embeddings, p=2, dim=1).reshape(classes, 3, -1)
    prototypes = nn.functional.normalize(grouped[:, :2].mean(dim=1), p=2, dim=1)
    queries = grouped[:, 2]
    logits = queries @ prototypes.transpose(0, 1) / float(temperature)
    return nn.functional.cross_entropy(logits, torch.arange(classes, device=targets.device))


def temporal_relation_pair_loss(anchor_temporal: torch.Tensor, peer_temporal: torch.Tensor) -> torch.Tensor:
    """Alinea relaciones temporales internas de una pareja real, no sus valores absolutos."""
    if anchor_temporal.ndim != 3 or peer_temporal.shape != anchor_temporal.shape:
        raise ValueError("La pérdida relacional requiere temporales emparejados (batch,channels,frames)")
    anchor = nn.functional.normalize(anchor_temporal.transpose(1, 2), p=2, dim=2)
    peer = nn.functional.normalize(peer_temporal.transpose(1, 2), p=2, dim=2)
    anchor_gram = anchor @ anchor.transpose(1, 2)
    peer_gram = peer @ peer.transpose(1, 2)
    return nn.functional.smooth_l1_loss(anchor_gram, peer_gram)


SELECTIVE_CORE_CHANNELS = (125, 122, 119, 116, 107, 110, 104, 113, 67, 95, 98, 101, 92, 54, 89, 51, 86, 83)


def selective_core_relation_loss(model, anchor_features: torch.Tensor, peer_features: torch.Tensor) -> torch.Tensor:
    """Relación del encoder sobre una vista fija de evidencia central estable."""
    if anchor_features.shape != peer_features.shape or anchor_features.ndim != 3 or anchor_features.shape[1:] != (30, 126):
        raise ValueError("La vista selectiva requiere pares (batch,30,126)")
    mask = torch.zeros((1, 30, 126), device=anchor_features.device, dtype=anchor_features.dtype)
    mask[:, 11:22, list(SELECTIVE_CORE_CHANNELS)] = 1.0
    anchor_temporal = model.forward_features(anchor_features * mask)
    peer_temporal = model.forward_features(peer_features * mask)
    return temporal_relation_pair_loss(anchor_temporal[:, :, 11:22], peer_temporal[:, :, 11:22])


def symmetric_prediction_consistency(logits: torch.Tensor, augmented_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """KL simétrico con objetivos desacoplados entre vista original y auxiliar."""
    if temperature <= 0:
        raise ValueError("La temperatura de consistencia debe ser positiva")
    original_log = nn.functional.log_softmax(logits / temperature, dim=1)
    augmented_log = nn.functional.log_softmax(augmented_logits / temperature, dim=1)
    original_target = nn.functional.softmax(logits / temperature, dim=1).detach()
    augmented_target = nn.functional.softmax(augmented_logits / temperature, dim=1).detach()
    return 0.5 * (
        nn.functional.kl_div(augmented_log, original_target, reduction="batchmean")
        + nn.functional.kl_div(original_log, augmented_target, reduction="batchmean")
    )


def cross_signer_supervised_contrast_loss(anchor_temporal: torch.Tensor, peer_temporal: torch.Tensor, targets: torch.Tensor, temperature: float = 0.20) -> torch.Tensor:
    """W57: SupCon global y simétrico sobre pares same-label cross-signer de train."""
    if anchor_temporal.ndim != 3 or peer_temporal.shape != anchor_temporal.shape or anchor_temporal.shape[1] != 128:
        raise ValueError("W57 requiere temporales W33 compatibles (batch,128,frames)")
    if targets.ndim != 1 or targets.numel() != anchor_temporal.shape[0] or (targets < 0).any():
        raise ValueError("W57 requiere un target no negativo por embedding")
    if temperature <= 0.0:
        raise ValueError("W57 requiere temperatura positiva")
    anchors = nn.functional.normalize(anchor_temporal.mean(dim=-1), p=2, dim=1)
    peers = nn.functional.normalize(peer_temporal.mean(dim=-1), p=2, dim=1)
    positive_mask = targets[:, None].eq(targets[None, :])
    if not positive_mask.any(dim=1).all():
        raise ValueError("W57 requiere al menos un positivo supervisado por ancla")

    def directed_loss(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
        log_probabilities = source @ destination.transpose(0, 1) / float(temperature)
        log_probabilities = log_probabilities - torch.logsumexp(log_probabilities, dim=1, keepdim=True)
        positive_counts = positive_mask.sum(dim=1)
        return -((log_probabilities * positive_mask.to(log_probabilities)).sum(dim=1) / positive_counts).mean()

    return 0.5 * (directed_loss(anchors, peers) + directed_loss(peers, anchors))


def freeze_for_decoupled_classifier_retraining(model: nn.Module) -> list[nn.Parameter]:
    """W59: congela encoder W33 y deja entrenable solo la cabeza ya aprendida."""
    head_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        is_head = name.startswith("head.")
        parameter.requires_grad_(is_head)
        if is_head:
            head_parameters.append(parameter)
    if not head_parameters or any(parameter.requires_grad for name, parameter in model.named_parameters() if not name.startswith("head.")):
        raise ValueError("W59 no pudo congelar exactamente el encoder W33")
    return head_parameters


def stochastic_dropout_ldam_loss(model, features: torch.Tensor, targets: torch.Tensor, margins: torch.Tensor, class_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """W42: dos pasadas dropout del mismo W33, LDAM en ambas y KL simétrica."""
    if not model.training:
        raise ValueError("W42 requiere model.train(True) para muestrear dos máscaras dropout")
    first_logits = model(features)
    second_logits = model(features)
    first_loss = ldam_deferred_reweighting_loss(first_logits, targets, margins, class_weights)
    second_loss = ldam_deferred_reweighting_loss(second_logits, targets, margins, class_weights)
    consistency = symmetric_prediction_consistency(first_logits, second_logits, temperature=1.0)
    return 0.5 * (first_loss + second_loss) + consistency, 0.5 * (first_logits + second_logits), consistency


def canonical_motion_vat_scale(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Escala W43 fijada por RMS del split train; forma manual queda en cero."""
    scale = torch.zeros((1, 1, 352), device=device, dtype=dtype)
    scale[:, :, :126] = 0.615639
    scale[:, :, 226:352] = 0.176314
    return scale


def _w43_rms_normalize(direction: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Normaliza por muestra sobre canales W43 permitidos, preservando la máscara cero."""
    active = scale.ne(0.0).to(dtype=direction.dtype)
    active_count = active.sum() * direction.shape[1]
    if active_count <= 0:
        raise ValueError("W43 requiere al menos un canal perturbable")
    masked = direction * active
    rms = torch.sqrt(masked.square().sum(dim=(1, 2), keepdim=True) / active_count + 1e-12)
    return masked / rms


def canonical_motion_vat_loss(model, features: torch.Tensor, epsilon: float = 0.10, xi: float = 0.001, weight: float = 0.10) -> tuple[torch.Tensor, torch.Tensor]:
    """W43: KL adversarial virtual local sobre posición/velocidad canónica W33."""
    if features.ndim != 3 or features.shape[1:] != (30, 352):
        raise ValueError("W43 requiere features canónicos (batch,30,352)")
    if not model.training:
        raise ValueError("W43 solo se calcula durante model.train(True)")
    if epsilon < 0.0 or xi <= 0.0 or weight < 0.0:
        raise ValueError("W43 requiere epsilon y peso no negativos, xi positivo")
    if epsilon == 0.0 or weight == 0.0:
        return features.new_zeros(()), torch.zeros_like(features)
    scale = canonical_motion_vat_scale(features.device, features.dtype)
    was_training = model.training
    try:
        model.train(False)
        with torch.no_grad():
            clean_target = nn.functional.softmax(model(features), dim=1)
        direction = torch.randn_like(features)
        direction = _w43_rms_normalize(direction, scale).detach().requires_grad_(True)
        probe_logits = model(features + xi * scale * direction)
        probe_loss = nn.functional.kl_div(nn.functional.log_softmax(probe_logits, dim=1), clean_target, reduction="batchmean")
        gradient = torch.autograd.grad(probe_loss, direction, only_inputs=True)[0]
        adversarial_direction = _w43_rms_normalize(gradient.detach(), scale)
        perturbation = epsilon * scale * adversarial_direction
        adversarial_logits = model(features + perturbation)
        vat = nn.functional.kl_div(nn.functional.log_softmax(adversarial_logits, dim=1), clean_target, reduction="batchmean").clamp_min(0.0)
        if not torch.isfinite(vat) or not torch.isfinite(perturbation).all():
            raise FloatingPointError("W43 produjo VAT o perturbación no finita")
        return weight * vat, perturbation
    finally:
        model.train(was_training)


W93_TRAIN_MEDIAN_TRANSITION_ENERGY = 0.04617374696623902
W93_COHERENCE_WEIGHT = 0.010


def motion_adaptive_temporal_coherence_loss(features: torch.Tensor, temporal: torch.Tensor, scale: float = W93_TRAIN_MEDIAN_TRANSITION_ENERGY) -> torch.Tensor:
    """W93: curvatura temporal interna, reducida durante movimiento manual alto."""
    if features.ndim != 3 or features.shape[1:] != (30, 352):
        raise ValueError("W93 requiere features W33 (batch,30,352)")
    if temporal.ndim != 3 or temporal.shape != (features.shape[0], 128, 30):
        raise ValueError("W93 requiere temporales W33 (batch,128,30)")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("W93 requiere escala train positiva y finita")
    transition = torch.sqrt((features[:, 1:, :126] - features[:, :-1, :126]).square().mean(dim=2) + 1e-12)
    centered_energy = 0.5 * (transition[:, :-1] + transition[:, 1:])
    adaptive_weight = torch.exp(-centered_energy / (float(scale) + 1e-8))
    normalized = temporal / torch.sqrt(temporal.square().mean(dim=1, keepdim=True) + 1e-6)
    curvature = normalized[:, :, 2:] - 2.0 * normalized[:, :, 1:-1] + normalized[:, :, :-2]
    loss = (curvature.square() * adaptive_weight[:, None, :]).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("W93 produjo pérdida de coherencia no finita")
    return loss


def mean_teacher_temporal_consistency(student_augmented_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """W30: KL asimétrico desde maestro EMA canónico hacia estudiante deformado."""
    if temperature <= 0:
        raise ValueError("La temperatura del maestro EMA debe ser positiva")
    if student_augmented_logits.shape != teacher_logits.shape or student_augmented_logits.ndim != 2:
        raise ValueError("W30 requiere logits de estudiante y maestro de igual forma (batch,clases)")
    student_log = nn.functional.log_softmax(student_augmented_logits / temperature, dim=1)
    teacher_target = nn.functional.softmax(teacher_logits.detach() / temperature, dim=1)
    return (temperature ** 2) * nn.functional.kl_div(student_log, teacher_target, reduction="batchmean")


def position_velocity_representation_consistency(model, features: torch.Tensor) -> torch.Tensor:
    """W31: acuerdo coseno entre posición 0:226 y velocidad manual 226:352, solo en train."""
    if features.ndim != 3 or features.shape[1:] != (30, 352):
        raise ValueError("W31 requiere features canónicos (batch,30,352)")
    position_view = features.clone()
    position_view[:, :, 226:352] = 0.0
    velocity_view = features.clone()
    velocity_view[:, :, :226] = 0.0
    position_embedding = nn.functional.normalize(model.temporal_features(position_view).mean(dim=-1), p=2, dim=1)
    velocity_embedding = nn.functional.normalize(model.temporal_features(velocity_view).mean(dim=-1), p=2, dim=1)
    return 1.0 - (position_embedding * velocity_embedding).sum(dim=1).mean()


def ldam_margins(class_counts: torch.Tensor, max_margin: float = 0.5) -> torch.Tensor:
    """W33: margen fijo por clase de train, proporcional a n^(-1/4)."""
    if class_counts.ndim != 1 or class_counts.numel() == 0 or (class_counts <= 0).any():
        raise ValueError("W33 requiere conteos de clase train positivos en un vector")
    if max_margin <= 0:
        raise ValueError("W33 requiere max_margin positivo")
    return max_margin / class_counts.to(dtype=torch.float32).pow(0.25)


def effective_number_class_weights(class_counts: torch.Tensor, beta: float = 0.9999) -> torch.Tensor:
    """W33: pesos de número efectivo, normalizados a media uno, solo para época tardía."""
    if class_counts.ndim != 1 or class_counts.numel() == 0 or (class_counts <= 0).any():
        raise ValueError("W33 requiere conteos de clase train positivos en un vector")
    if not 0.0 <= beta < 1.0:
        raise ValueError("W33 requiere beta en [0,1)")
    counts = class_counts.to(dtype=torch.float32)
    beta_tensor = torch.as_tensor(beta, dtype=counts.dtype, device=counts.device)
    weights = (1.0 - beta) / (1.0 - torch.pow(beta_tensor, counts))
    return weights / weights.mean()


def ldam_deferred_reweighting_loss(logits: torch.Tensor, targets: torch.Tensor, margins: torch.Tensor, class_weights: torch.Tensor | None = None, scale: float = 30.0) -> torch.Tensor:
    """W33: LDAM, sin pesos antes de época 31 y con pesos efectivos después."""
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("W33 requiere logits (batch,clases) y targets (batch,) compatibles")
    if margins.shape != (logits.shape[1],) or (margins <= 0).any():
        raise ValueError("W33 requiere un margen positivo por clase")
    if targets.numel() == 0 or targets.min() < 0 or targets.max() >= logits.shape[1]:
        raise ValueError("W33 recibió targets fuera de rango")
    if class_weights is not None and class_weights.shape != margins.shape:
        raise ValueError("W33 requiere pesos tardíos por clase compatibles")
    if scale <= 0:
        raise ValueError("W33 requiere escala positiva")
    adjusted = logits.clone()
    adjusted[torch.arange(targets.shape[0], device=targets.device), targets] -= margins.to(logits)[targets]
    return nn.functional.cross_entropy(scale * adjusted, targets, weight=None if class_weights is None else class_weights.to(logits))


def group_dro_ldam_deferred_reweighting_loss(logits: torch.Tensor, targets: torch.Tensor, signer_targets: torch.Tensor, log_group_weights: torch.Tensor, margins: torch.Tensor, class_weights: torch.Tensor | None = None, eta: float = 0.01, logit_clip: float = 20.0, scale: float = 30.0) -> torch.Tensor:
    """W85: LDAM-DRW con peor pérdida media de firmante exclusivamente de train."""
    if eta <= 0.0 or logit_clip <= 0.0:
        raise ValueError("W85 requiere eta y clipping positivos")
    if logits.ndim != 2 or targets.ndim != 1 or signer_targets.ndim != 1 or logits.shape[0] != targets.numel() or targets.shape != signer_targets.shape:
        raise ValueError("W85 requiere logits, targets y firmantes train compatibles")
    if margins.shape != (logits.shape[1],) or (margins <= 0).any():
        raise ValueError("W85 requiere márgenes LDAM positivos por clase")
    if signer_targets.numel() == 0 or signer_targets.min() < 0 or signer_targets.max() >= log_group_weights.numel():
        raise ValueError("W85 recibió un firmante fuera de train")
    if not torch.isfinite(log_group_weights).all():
        raise ValueError("W85 requiere log-pesos de grupo finitos")
    if class_weights is not None and (class_weights.shape != margins.shape or (class_weights <= 0).any()):
        raise ValueError("W85 requiere pesos tardíos de clase positivos compatibles")
    adjusted = logits.clone()
    adjusted[torch.arange(targets.shape[0], device=targets.device), targets] -= margins.to(logits)[targets]
    per_sample = nn.functional.cross_entropy(float(scale) * adjusted, targets, reduction="none")
    sample_weights = None if class_weights is None else class_weights.to(logits)[targets]
    groups = torch.unique(signer_targets, sorted=True)
    group_losses = []
    for group in groups:
        selected = signer_targets == group
        if sample_weights is None:
            group_losses.append(per_sample[selected].mean())
        else:
            current_weights = sample_weights[selected]
            group_losses.append((current_weights * per_sample[selected]).sum() / current_weights.sum().clamp_min(1e-12))
    losses = torch.stack(group_losses)
    with torch.no_grad():
        log_group_weights[groups] = (log_group_weights[groups] + float(eta) * losses.detach()).clamp(-float(logit_clip), float(logit_clip))
        log_group_weights.sub_(torch.logsumexp(log_group_weights, dim=0))
    return (torch.softmax(log_group_weights[groups], dim=0) * losses).sum()


def signer_vrex_loss(logits: torch.Tensor, targets: torch.Tensor, signer_targets: torch.Tensor, variance_weight: float = 1.0) -> torch.Tensor:
    """V-REx: media de riesgos por firmante más varianza poblacional train-only."""
    if variance_weight < 0.0:
        raise ValueError("V-REx requiere peso de varianza no negativo")
    if logits.ndim != 2 or targets.ndim != 1 or signer_targets.ndim != 1 or logits.shape[0] != targets.numel() or targets.shape != signer_targets.shape:
        raise ValueError("V-REx requiere logits, targets y firmantes compatibles")
    if targets.numel() == 0 or targets.min() < 0 or targets.max() >= logits.shape[1] or signer_targets.min() < 0:
        raise ValueError("V-REx recibió objetivos o firmantes inválidos")
    per_sample = nn.functional.cross_entropy(logits, targets, reduction="none")
    group_losses = torch.stack([per_sample[signer_targets == signer].mean() for signer in torch.unique(signer_targets, sorted=True)])
    return group_losses.mean() + float(variance_weight) * group_losses.var(unbiased=False)


def confusion_spectral_regularizer(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """W72: norma espectral de la confusión suave exclusivamente fuera de diagonal."""
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("W72 requiere logits no vacíos (batch,clases)")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0] or (targets < 0).any() or (targets >= logits.shape[1]).any():
        raise ValueError("W72 recibió targets inválidos")
    if not torch.isfinite(logits).all():
        raise ValueError("W72 requiere logits finitos")
    probabilities = nn.functional.softmax(logits, dim=1)
    one_hot = nn.functional.one_hot(targets, num_classes=logits.shape[1]).to(dtype=logits.dtype)
    counts = one_hot.sum(dim=0).clamp_min(1.0)
    confusion = (one_hot.transpose(0, 1) @ probabilities) / counts[:, None]
    off_diagonal = confusion - torch.diag_embed(torch.diagonal(confusion))
    regularizer = torch.linalg.matrix_norm(off_diagonal, ord=2)
    if not torch.isfinite(regularizer):
        raise FloatingPointError("W72 produjo regularizador no finito")
    return regularizer


def signer_covariance_alignment_loss(temporal: torch.Tensor, signer_targets: torch.Tensor) -> torch.Tensor:
    """W74: alinea covarianzas globales de embeddings de firmantes solo en train."""
    if temporal.ndim != 3 or temporal.shape[1] != 128 or temporal.shape[2] < 1:
        raise ValueError("W74 requiere temporales (batch,128,frames)")
    if signer_targets.ndim != 1 or signer_targets.shape[0] != temporal.shape[0] or (signer_targets < 0).any():
        raise ValueError("W74 requiere un firmante train válido por embedding")
    pooled = temporal.mean(dim=-1)
    covariances = []
    for signer in torch.unique(signer_targets, sorted=True):
        group = pooled[signer_targets == signer]
        if group.shape[0] < 2:
            continue
        centered = group - group.mean(dim=0, keepdim=True)
        covariances.append((centered.transpose(0, 1) @ centered) / float(group.shape[0] - 1))
    if len(covariances) < 2:
        return temporal.sum() * 0.0
    stacked = torch.stack(covariances)
    average = stacked.mean(dim=0, keepdim=True)
    loss = (stacked - average).square().sum(dim=(1, 2)).mean() / float(4 * temporal.shape[1] ** 2)
    if not torch.isfinite(loss):
        raise FloatingPointError("W74 produjo una alineación de covarianza no finita")
    return loss


def focal_ldam_deferred_reweighting_loss(logits: torch.Tensor, targets: torch.Tensor, margins: torch.Tensor, class_weights: torch.Tensor | None = None, gamma: float = 1.0, scale: float = 30.0) -> torch.Tensor:
    """W51: LDAM-DRW modulada por dificultad focal del ejemplo objetivo."""
    if gamma < 0.0:
        raise ValueError("W51 requiere gamma no negativo")
    if logits.ndim != 2 or margins.ndim != 1 or logits.shape[1] != margins.numel():
        raise ValueError("W51 requiere logits y márgenes compatibles")
    if targets.ndim != 1 or targets.numel() != logits.shape[0] or (targets < 0).any() or (targets >= logits.shape[1]).any():
        raise ValueError("W51 recibió targets inválidos")
    adjusted = logits.clone()
    adjusted[torch.arange(targets.shape[0], device=targets.device), targets] -= margins.to(logits)[targets]
    scaled = float(scale) * adjusted
    log_probabilities = nn.functional.log_softmax(scaled, dim=1)
    target_log_probabilities = log_probabilities.gather(1, targets[:, None]).squeeze(1)
    target_probabilities = target_log_probabilities.exp()
    per_example = -target_log_probabilities * (1.0 - target_probabilities).clamp_min(0.0).pow(float(gamma))
    if class_weights is None:
        return per_example.mean()
    if class_weights.ndim != 1 or class_weights.numel() != logits.shape[1] or (class_weights <= 0).any():
        raise ValueError("W51 requiere pesos de clase positivos compatibles")
    sample_weights = class_weights.to(logits)[targets]
    return (sample_weights * per_example).sum() / sample_weights.sum()


def class_balanced_train_sample_weights(train_rows: list[dict]) -> torch.Tensor:
    """W52: pesos inversos a frecuencia, exclusivamente para filas train."""
    if not train_rows or any(row.get("split_active") != "train" for row in train_rows):
        raise ValueError("W52 requiere exclusivamente filas train no vacías")
    counts = Counter(row["label_lsm"] for row in train_rows)
    if any(count <= 0 for count in counts.values()):
        raise ValueError("W52 requiere conteos train positivos")
    return torch.tensor([1.0 / counts[row["label_lsm"]] for row in train_rows], dtype=torch.double)


def build_class_balanced_train_sampler(train_rows: list[dict], seed: int) -> WeightedRandomSampler:
    weights = class_balanced_train_sample_weights(train_rows)
    generator = torch.Generator().manual_seed(int(seed) + 1701)
    return WeightedRandomSampler(weights, num_samples=len(train_rows), replacement=True, generator=generator)


def w89_observation_quality(masks: np.ndarray) -> float:
    """W89: calidad factual de dos máscaras W84 `(30,2)` sin usar etiquetas ni firmante."""
    masks = np.asarray(masks, dtype=np.float32)
    if masks.shape != (30, 2) or not np.all(np.isfinite(masks)) or not np.all(np.isin(masks, (0.0, 1.0))):
        raise ValueError("W89 requiere máscaras binarias finitas de forma (30,2)")
    mean_hands = float(masks.mean())
    any_hand = float((masks.sum(axis=1) > 0.0).mean())
    return 0.70 * mean_hands + 0.30 * any_hand


def w89_train_quality_by_sample(train_rows: list[dict], presence_cache_root: Path) -> dict[str, float]:
    """W89: calcula una vez la calidad exclusivamente desde filas train y cache W84."""
    if not train_rows or any(row.get("split_active") != "train" for row in train_rows):
        raise ValueError("W89 requiere exclusivamente filas train no vacías para construir calidad")
    scores: dict[str, float] = {}
    for row in train_rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in scores:
            raise ValueError("W89 requiere sample_id train único y no vacío")
        feature = load_array(feature_path(row, presence_cache_root), (30, 354))
        scores[sample_id] = w89_observation_quality(feature[:, 352:354])
    return scores


def w89_curriculum_rows(train_rows: list[dict], quality_by_sample: dict[str, float], epoch: int) -> tuple[list[dict], str]:
    """W89: top calidad por clase en 70/85/100%, con cobertura de cada clase preservada."""
    if not train_rows or any(row.get("split_active") != "train" for row in train_rows):
        raise ValueError("W89 solo selecciona filas train no vacías")
    if epoch < 1:
        raise ValueError("W89 requiere época positiva")
    if epoch <= 15:
        fraction, phase = 0.70, "quality_top_70"
    elif epoch <= 30:
        fraction, phase = 0.85, "quality_top_85"
    else:
        fraction, phase = 1.00, "quality_top_100"
    by_label: dict[str, list[dict]] = {}
    for row in train_rows:
        sample_id = str(row.get("sample_id", ""))
        quality = quality_by_sample.get(sample_id)
        if quality is None or not np.isfinite(quality) or not 0.0 <= float(quality) <= 1.0:
            raise ValueError(f"W89 no tiene calidad train válida para {sample_id!r}")
        by_label.setdefault(str(row["label_lsm"]), []).append(row)
    selected: list[dict] = []
    for label in sorted(by_label):
        ranked = sorted(by_label[label], key=lambda row: (-float(quality_by_sample[str(row["sample_id"])]), str(row["sample_id"])))
        take = int(np.ceil(fraction * len(ranked)))
        if take < 1:
            raise ValueError("W89 debe preservar al menos un ejemplo por clase")
        selected.extend(ranked[:take])
    if set(row["label_lsm"] for row in selected) != set(by_label):
        raise ValueError("W89 perdió cobertura de una clase train")
    return selected, phase


def train_prior_log_priors(class_counts: torch.Tensor) -> torch.Tensor:
    """W48: log-priors de clases calculados exclusivamente de conteos train positivos."""
    if class_counts.ndim != 1 or class_counts.numel() == 0 or (class_counts <= 0).any():
        raise ValueError("W48 requiere conteos train positivos en un vector")
    priors = class_counts.to(dtype=torch.float32) / class_counts.sum()
    return priors.log()


def train_prior_logit_adjusted_ldam_loss(logits: torch.Tensor, targets: torch.Tensor, margins: torch.Tensor, log_priors: torch.Tensor, tau: float = 1.0, scale: float = 30.0) -> torch.Tensor:
    """W48: LDAM con ajuste de prior aplicado solo en la pérdida de entrenamiento."""
    if tau < 0.0:
        raise ValueError("W48 requiere tau no negativo")
    if log_priors.shape != margins.shape or not torch.isfinite(log_priors).all():
        raise ValueError("W48 requiere log-priors finitos compatibles con las clases")
    if logits.ndim != 2 or logits.shape[1] != log_priors.numel():
        raise ValueError("W48 requiere logits compatibles con log-priors")
    adjusted = logits + float(tau) * log_priors.to(logits)
    return ldam_deferred_reweighting_loss(adjusted, targets, margins, class_weights=None, scale=scale)


def classifier_weight_coherence_loss(weights: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """W50: penaliza coherencia angular fuera de diagonal de una cabeza lineal."""
    if weights.ndim != 2 or min(weights.shape) < 1:
        raise ValueError("W50 requiere una matriz no vacía de pesos de clasificador")
    if epsilon <= 0.0 or not torch.isfinite(weights).all():
        raise ValueError("W50 requiere pesos finitos y epsilon positivo")
    normalized = nn.functional.normalize(weights, p=2, dim=1, eps=epsilon)
    gram = normalized @ normalized.transpose(0, 1)
    off_diagonal = gram * (1.0 - torch.eye(weights.shape[0], device=weights.device, dtype=weights.dtype))
    return weights.shape[0] * off_diagonal.square().mean()


def manual_factorial_decorrelation_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """W34: penaliza solo el colapso exacto de las cuatro ramas auxiliares."""
    if embeddings.ndim != 3 or embeddings.shape[1:] != (4, 32):
        raise ValueError("W34 requiere embeddings de factores (batch,4,32)")
    if not torch.isfinite(embeddings).all():
        raise ValueError("W34 recibió embeddings no finitos")
    normalized = nn.functional.normalize(embeddings, p=2, dim=-1)
    similarity = normalized @ normalized.transpose(1, 2)
    indices = torch.triu_indices(4, 4, offset=1, device=embeddings.device)
    return similarity[:, indices[0], indices[1]].square().mean()


def soft_dtw_cost(first: torch.Tensor, second: torch.Tensor, gamma: float) -> torch.Tensor:
    """Costo soft-DTW por batch para secuencias (batch,frames,canales)."""
    if gamma <= 0:
        raise ValueError("gamma de soft-DTW debe ser positivo")
    if first.ndim != 3 or second.shape != first.shape:
        raise ValueError("soft-DTW espera secuencias compatibles (batch,frames,canales)")
    distances = (first[:, :, None, :] - second[:, None, :, :]).square().sum(dim=-1)
    batch, first_frames, second_frames = distances.shape
    infinity = distances.new_full((batch,), float("inf"))
    previous = [distances.new_zeros(batch)] + [infinity] * second_frames
    for first_index in range(first_frames):
        current = [infinity]
        for second_index in range(second_frames):
            previous_costs = torch.stack([previous[second_index], previous[second_index + 1], current[second_index]], dim=0)
            soft_minimum = -gamma * torch.logsumexp(-previous_costs / gamma, dim=0)
            current.append(distances[:, first_index, second_index] + soft_minimum)
        previous = current
    return previous[-1]


def soft_dtw_divergence(first: torch.Tensor, second: torch.Tensor, gamma: float) -> torch.Tensor:
    """Divergencia soft-DTW corregida por el sesgo de autoalineación."""
    return soft_dtw_cost(first, second, gamma) - 0.5 * (soft_dtw_cost(first, first, gamma) + soft_dtw_cost(second, second, gamma))


def temporal_contrastive_soft_dtw_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, gamma: float = 0.10, temperature: float = 0.25) -> torch.Tensor:
    """W21: alinea la misma seña entre firmantes y separa otra seña del mismo firmante."""
    if temperature <= 0:
        raise ValueError("La temperatura contrastiva debe ser positiva")
    if anchor.ndim != 3 or positive.shape != anchor.shape or negative.shape != anchor.shape:
        raise ValueError("W21 espera tripletes (batch,canales,frames) de la misma forma")
    anchor_sequence = nn.functional.normalize(anchor.transpose(1, 2), p=2, dim=-1)
    positive_sequence = nn.functional.normalize(positive.transpose(1, 2), p=2, dim=-1)
    negative_sequence = nn.functional.normalize(negative.transpose(1, 2), p=2, dim=-1)
    positive_distance = soft_dtw_divergence(anchor_sequence, positive_sequence, gamma)
    negative_distance = soft_dtw_divergence(anchor_sequence, negative_sequence, gamma)
    return -nn.functional.logsigmoid((negative_distance - positive_distance) / temperature).mean()


def log_euclidean_temporal_covariance(features: torch.Tensor, diagonal: float = 1e-3) -> torch.Tensor:
    """Covarianzas SPD Log-Euclidianas por ocho bloques de 16 canales."""
    if diagonal <= 0:
        raise ValueError("La diagonal SPD debe ser positiva")
    if features.ndim != 3 or features.shape[1] != 128 or features.shape[2] < 2:
        raise ValueError("W23 espera rasgos temporales (batch,128,frames>=2)")
    batch, _, frames = features.shape
    blocks = features.reshape(batch, 8, 16, frames)
    centered = blocks - blocks.mean(dim=-1, keepdim=True)
    covariance = centered @ centered.transpose(-1, -2) / float(frames - 1)
    eye = torch.eye(16, device=features.device, dtype=features.dtype).reshape(1, 1, 16, 16)
    covariance = covariance + diagonal * eye
    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    covariance64 = covariance.to(torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance64)
    logarithms = eigenvalues.clamp_min(diagonal).log()
    return ((eigenvectors * logarithms.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)).to(features.dtype)


def log_euclidean_covariance_consistency(anchor: torch.Tensor, peer: torch.Tensor, diagonal: float = 1e-3) -> torch.Tensor:
    """W23: distancia cuadrática entre covarianzas temporales de la misma palabra."""
    if peer.shape != anchor.shape:
        raise ValueError("W23 requiere pareja temporal de la misma forma")
    return (log_euclidean_temporal_covariance(anchor, diagonal) - log_euclidean_temporal_covariance(peer, diagonal)).square().mean()


def uncertainty_weighted_two_loss(classification_loss: torch.Tensor, covariance_loss: torch.Tensor, log_variances: torch.Tensor) -> torch.Tensor:
    """W24: equilibrio homoscedástico entrenable, sin hiperparámetro ajustado en test."""
    if log_variances.shape != (2,):
        raise ValueError("W24 requiere exactamente dos log-incertidumbres escalares")
    if not torch.isfinite(log_variances).all():
        raise ValueError("W24 recibió log-incertidumbres no finitas")
    return torch.exp(-log_variances[0]) * classification_loss + log_variances[0] + torch.exp(-log_variances[1]) * covariance_loss + log_variances[1]


def sam_perturb_parameters(parameters: list[torch.nn.Parameter], rho: float = 0.05) -> list[torch.Tensor]:
    """Aplica la perturbación SAM global y devuelve los deltas para restauración exacta."""
    if rho <= 0:
        raise ValueError("SAM requiere rho positivo")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        raise ValueError("SAM requiere gradientes antes de perturbar")
    gradient_norm = torch.linalg.vector_norm(torch.stack([gradient.norm(p=2) for gradient in gradients]), ord=2)
    scale = rho / (gradient_norm + 1e-12)
    perturbations = []
    with torch.no_grad():
        for parameter in parameters:
            perturbation = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad * scale.to(parameter)
            parameter.add_(perturbation)
            perturbations.append(perturbation)
    return perturbations


def sam_restore_parameters(parameters: list[torch.nn.Parameter], perturbations: list[torch.Tensor]) -> None:
    if len(parameters) != len(perturbations):
        raise ValueError("SAM requiere una perturbación por parámetro")
    with torch.no_grad():
        for parameter, perturbation in zip(parameters, perturbations):
            parameter.sub_(perturbation)


def group_dro_weighted_loss(per_sample_losses: torch.Tensor, group_targets: torch.Tensor, log_group_weights: torch.Tensor, eta: float = 0.01) -> torch.Tensor:
    """W26: maximiza de forma exponenciada la pérdida de firmantes train observados."""
    if eta <= 0:
        raise ValueError("W26 requiere eta positivo")
    if per_sample_losses.ndim != 1 or group_targets.ndim != 1 or per_sample_losses.shape != group_targets.shape:
        raise ValueError("W26 requiere pérdidas y grupos vectoriales de igual longitud")
    if log_group_weights.ndim != 1 or not torch.isfinite(log_group_weights).all():
        raise ValueError("W26 requiere log-pesos de grupo finitos")
    if group_targets.numel() == 0 or group_targets.min() < 0 or group_targets.max() >= log_group_weights.numel():
        raise ValueError("W26 recibió un firmante fuera de train")
    groups = torch.unique(group_targets, sorted=True)
    group_losses = torch.stack([per_sample_losses[group_targets == group].mean() for group in groups])
    with torch.no_grad():
        log_group_weights[groups] += eta * group_losses.detach()
        log_group_weights.sub_(torch.logsumexp(log_group_weights, dim=0))
    present_weights = torch.softmax(log_group_weights[groups], dim=0)
    return (present_weights * group_losses).sum()


def masked_temporal_embedding_loss(student, teacher, features: torch.Tensor, block_size: int = 4, starts: torch.Tensor | None = None) -> torch.Tensor:
    """W27: predice embeddings de un maestro EMA para un bloque temporal oculto."""
    if features.ndim != 3 or features.shape[1] < block_size:
        raise ValueError("W27 requiere secuencias (batch,frames,features) con bloque temporal válido")
    batch, frames, _ = features.shape
    if starts is None:
        starts = torch.randint(0, frames - block_size + 1, (batch,), device=features.device)
    if starts.shape != (batch,) or starts.min() < 0 or starts.max() > frames - block_size:
        raise ValueError("W27 recibió inicios de máscara fuera de rango")
    masked = features.clone()
    temporal_mask = torch.zeros((batch, frames), dtype=torch.bool, device=features.device)
    for index, start in enumerate(starts.tolist()):
        masked[index, start : start + block_size] = 0.0
        temporal_mask[index, start : start + block_size] = True
    with torch.no_grad():
        teacher_temporal = teacher.temporal_features(features)
    student_temporal = student.temporal_features(masked)
    mask = temporal_mask[:, None, :].to(dtype=student_temporal.dtype)
    return ((student_temporal - teacher_temporal).square() * mask).sum() / (mask.sum() * student_temporal.shape[1]).clamp_min(1.0)


@torch.no_grad()
def update_ema_teacher(student, teacher, momentum: float = 0.99) -> None:
    """W27: actualización EMA del maestro, fuera de inferencia y de los checkpoints finales."""
    if not 0.0 < momentum < 1.0:
        raise ValueError("W27 requiere un momentum EMA estrictamente entre 0 y 1")
    for student_parameter, teacher_parameter in zip(student.parameters(), teacher.parameters()):
        teacher_parameter.mul_(momentum).add_(student_parameter, alpha=1.0 - momentum)
    for student_buffer, teacher_buffer in zip(student.buffers(), teacher.buffers()):
        teacher_buffer.copy_(student_buffer)


def ema_inference_model(student, teacher, use_ema_weight_average_inference: bool):
    """W32: valida que selección, checkpoint y evaluación usen el único TCN EMA."""
    if use_ema_weight_average_inference:
        if teacher is None:
            raise ValueError("W32 requiere un modelo EMA para selección e inferencia")
        return teacher
    return student


def run_masked_temporal_pretraining(student, teacher, loader, optimizer, device, block_size: int = 4, ema_momentum: float = 0.99) -> dict:
    """Un epoch W27 sobre train sin leer etiquetas, validation ni test."""
    student.train(True)
    teacher.train(False)
    total_loss = 0.0
    support = 0
    for batch in loader:
        if len(batch) != 2:
            raise ValueError("W27 preentrena únicamente con secuencias train sin auxiliares")
        features, _unused_labels = batch
        features = features.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = masked_temporal_embedding_loss(student, teacher, features, block_size=block_size)
        if not torch.isfinite(loss):
            raise FloatingPointError("W27 produjo una pérdida de preentrenamiento no finita")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        update_ema_teacher(student, teacher, momentum=ema_momentum)
        total_loss += loss.detach().item() * features.size(0)
        support += features.size(0)
    return {"masked_embedding_loss": total_loss / max(1, support), "support": support}


def temporal_order_verification_loss(student, order_head, features: torch.Tensor, segment_size: int = 5, starts: torch.Tensor | None = None) -> torch.Tensor:
    """W29: BCE para el orden de dos segmentos consecutivos de embeddings, sin permutar la."""
    if features.ndim != 3 or features.shape[1] < 2 * segment_size:
        raise ValueError("W29 requiere secuencias con dos segmentos temporales consecutivos válidos")
    batch, frames, _ = features.shape
    if starts is None:
        starts = torch.randint(0, frames - 2 * segment_size + 1, (batch,), device=features.device)
    if starts.shape != (batch,) or starts.min() < 0 or starts.max() > frames - 2 * segment_size:
        raise ValueError("W29 recibió inicios de segmentos fuera de rango")
    temporal = student.temporal_features(features).transpose(1, 2)
    channels = temporal.shape[-1]
    offsets = torch.arange(segment_size, device=features.device)
    first_indices = starts[:, None] + offsets
    second_indices = starts[:, None] + segment_size + offsets
    first = temporal.gather(1, first_indices[:, :, None].expand(-1, -1, channels)).mean(dim=1)
    second = temporal.gather(1, second_indices[:, :, None].expand(-1, -1, channels)).mean(dim=1)
    forward_logits = order_head(torch.cat([first, second], dim=1)).squeeze(1)
    reversed_logits = order_head(torch.cat([second, first], dim=1)).squeeze(1)
    return 0.5 * (
        nn.functional.binary_cross_entropy_with_logits(forward_logits, torch.ones_like(forward_logits))
        + nn.functional.binary_cross_entropy_with_logits(reversed_logits, torch.zeros_like(reversed_logits))
    )


def run_temporal_order_pretraining(student, order_head, loader, optimizer, device, segment_size: int = 5) -> dict:
    """Un epoch W29 sobre clips train sin etiquetas, firmantes ni entradas permutadas."""
    student.train(True)
    order_head.train(True)
    total_loss = 0.0
    support = 0
    for batch in loader:
        if len(batch) != 2:
            raise ValueError("W29 preentrena únicamente con secuencias train sin auxiliares")
        features, _unused_labels = batch
        features = features.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = temporal_order_verification_loss(student, order_head, features, segment_size=segment_size)
        if not torch.isfinite(loss):
            raise FloatingPointError("W29 produjo una pérdida de verificación de orden no finita")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*student.parameters(), *order_head.parameters()], max_norm=1.0)
        optimizer.step()
        total_loss += loss.detach().item() * features.size(0)
        support += features.size(0)
    return {"temporal_order_loss": total_loss / max(1, support), "support": support}


def masked_hand_reconstruction_view(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Oculta una punta distal por mano y clip sin crear una muestra nueva."""
    if features.ndim != 3 or tuple(features.shape[1:]) != (30, 126):
        raise ValueError("La reconstrucción enmascarada requiere batch de forma (n,30,126)")
    distal = torch.tensor((4, 8, 12, 16, 20), device=features.device, dtype=torch.long)
    sample = torch.arange(features.shape[0], device=features.device)
    left = distal[sample.remainder(len(distal))]
    right = distal[(sample // len(distal)).remainder(len(distal))]
    coordinate = torch.arange(3, device=features.device)
    mask = torch.zeros_like(features, dtype=torch.bool)
    mask[sample[:, None], :, (left[:, None] * 3 + coordinate[None, :])] = True
    mask[sample[:, None], :, (63 + right[:, None] * 3 + coordinate[None, :])] = True
    return features.masked_fill(mask, 0.0), features, mask


def masked_hand_reconstruction_loss(predicted: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if predicted.shape != targets.shape or mask.shape != targets.shape or predicted.ndim != 3:
        raise ValueError("La reconstrucción requiere predicción, objetivo y máscara con la misma forma")
    if not bool(mask.any()):
        raise ValueError("La reconstrucción requiere al menos una coordenada oculta")
    loss = (predicted - targets).square().masked_select(mask).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("La pérdida de reconstrucción produjo NaN o Inf")
    return loss


# TODO: partir run_epoch 280→3
def run_epoch(model, loader, criterion, device, optimizer=None, classes=1, signer_loss_weight: float = 0.0, consistency_loss_weight: float = 0.0, consistency_temperature: float = 2.0, cross_signer_feature_mixup: bool = False, cross_signer_supervised_contrast: bool = False, contrastive_soft_dtw_alignment: bool = False, hopfield_prototype_memory: bool = False, use_log_euclidean_covariance_consistency: bool = False, uncertainty_log_variances: torch.nn.Parameter | None = None, use_sharpness_aware_minimization: bool = False, sam_rho: float = 0.05, group_dro_log_weights: torch.Tensor | None = None, group_dro_ldam: bool = False, mean_teacher=None, use_position_velocity_representation_consistency: bool = False, manual_factorial_ldam: bool = False, stochastic_dropout_consistency: bool = False, canonical_motion_vat: bool = False, train_prior_log_priors_tensor: torch.Tensor | None = None, classifier_coherence_weight: float = 0.0, focal_ldam_gamma: float | None = None, ldam_class_margins: torch.Tensor | None = None, ldam_late_weights: torch.Tensor | None = None, confusion_spectral_weight: float = 0.0, signer_covariance_weight: float = 0.0, signer_covariance_task: bool = False, motion_adaptive_temporal_coherence: bool = False, episodic_real_reference: bool = False, successor_temporal_relation_pairs: bool = False, successor_selective_core_relation: bool = False, soft_presence_weight: bool = False, masked_hand_reconstruction: bool = False, uniform_label_smoothing: bool = False, ecoc_auxiliary_head: bool = False, signer_vrex: bool = False, epoch: int | None = None, frozen_encoder_eval: bool = False):
    training = optimizer is not None
    model.train(training)
    if training and frozen_encoder_eval:
        for name, module in model.named_children():
            if name != "head":
                module.train(False)
    running_loss = 0.0
    signer_loss_total = 0.0
    reconstruction_loss_total = 0.0
    all_logits = []
    all_targets = []
    for batch in loader:
        peer_features = None
        consistency_features = None
        if len(batch) == 2:
            features, targets = batch
            signer_targets = None
        elif len(batch) == 3:
            if batch[1].ndim == 3:
                features, auxiliary_features, targets = batch
                if training and (cross_signer_feature_mixup or cross_signer_supervised_contrast or successor_temporal_relation_pairs or successor_selective_core_relation):
                    peer_features = auxiliary_features
                else:
                    consistency_features = auxiliary_features
                signer_targets = None
            else:
                features, targets, signer_targets = batch
        elif len(batch) == 4:
            features, peer_features, consistency_features, targets = batch
            signer_targets = None
        else:
            raise ValueError("El DataLoader debe devolver dos, tres o cuatro tensores")
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if peer_features is not None:
            peer_features = peer_features.to(device, non_blocking=True)
        if consistency_features is not None:
            consistency_features = consistency_features.to(device, non_blocking=True)
        if signer_targets is not None:
            signer_targets = signer_targets.to(device, non_blocking=True)
        if training and use_sharpness_aware_minimization:
            if peer_features is not None or consistency_features is not None or signer_targets is not None or uncertainty_log_variances is not None:
                raise ValueError("W25 usa exclusivamente CE W3 sin auxiliares ni pares")
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            rng_state = torch.get_rng_state()
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            first_loss = criterion(logits, targets)
            first_loss.backward()
            perturbations = sam_perturb_parameters(parameters, rho=sam_rho)
            optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(rng_state)
            robust_logits = model(features)
            loss = criterion(robust_logits, targets)
            loss.backward()
            sam_restore_parameters(parameters, perturbations)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.detach().item() * features.size(0)
            all_logits.append(logits.detach())
            all_targets.append(targets.detach())
            continue
        if training:
            optimizer.zero_grad(set_to_none=True)
        if training and ldam_class_margins is not None:
            if epoch is None or epoch < 1:
                raise ValueError("W33 requiere época positiva durante entrenamiento")
            if confusion_spectral_weight < 0.0:
                raise ValueError("W72 requiere peso espectral no negativo")
            if signer_covariance_weight < 0.0:
                raise ValueError("W74 requiere peso de covarianza no negativo")
            if (peer_features is not None and not cross_signer_supervised_contrast) or consistency_features is not None or (signer_targets is not None and not signer_covariance_task and signer_covariance_weight <= 0.0 and not group_dro_ldam) or mean_teacher is not None:
                raise ValueError("W33 usa exclusivamente logits W3 y conteos de clase train")
            if group_dro_ldam:
                if signer_targets is None or group_dro_log_weights is None or signer_covariance_weight > 0.0 or confusion_spectral_weight > 0.0 or cross_signer_supervised_contrast or focal_ldam_gamma is not None or classifier_coherence_weight > 0.0 or train_prior_log_priors_tensor is not None or stochastic_dropout_consistency or canonical_motion_vat or manual_factorial_ldam:
                    raise ValueError("W85 solo usa LDAM-DRW y firmantes exclusivos de train")
                logits = model(features)
                loss = group_dro_ldam_deferred_reweighting_loss(logits, targets, signer_targets, group_dro_log_weights, ldam_class_margins, ldam_late_weights if epoch >= 31 else None, eta=0.01, logit_clip=20.0)
            elif signer_covariance_weight > 0.0:
                if epoch < 31 or signer_targets is None or cross_signer_supervised_contrast or focal_ldam_gamma is not None or classifier_coherence_weight > 0.0 or train_prior_log_priors_tensor is not None or stochastic_dropout_consistency or canonical_motion_vat or manual_factorial_ldam:
                    raise ValueError("W74 solo se activa desde época 31 con firmantes train y sin otra pérdida auxiliar")
                temporal = model.temporal_features(features)
                logits = model.logits_from_temporal_features(temporal)
                loss = ldam_deferred_reweighting_loss(logits, targets, ldam_class_margins, ldam_late_weights if epoch >= 31 else None)
                loss = loss + float(signer_covariance_weight) * signer_covariance_alignment_loss(temporal, signer_targets)
            elif confusion_spectral_weight > 0.0:
                if epoch < 31 or cross_signer_supervised_contrast or focal_ldam_gamma is not None or classifier_coherence_weight > 0.0 or train_prior_log_priors_tensor is not None or stochastic_dropout_consistency or canonical_motion_vat or manual_factorial_ldam:
                    raise ValueError("W72 solo se activa desde época 31 y no se combina con otras pérdidas auxiliares")
                logits = model(features)
                loss = ldam_deferred_reweighting_loss(logits, targets, ldam_class_margins, ldam_late_weights if epoch >= 31 else None)
                loss = loss + float(confusion_spectral_weight) * confusion_spectral_regularizer(logits, targets)
            elif cross_signer_supervised_contrast:
                if focal_ldam_gamma is not None or classifier_coherence_weight > 0.0 or train_prior_log_priors_tensor is not None or stochastic_dropout_consistency or canonical_motion_vat or manual_factorial_ldam or peer_features is None:
                    raise ValueError("W57 no se combina con otras pérdidas auxiliares o ajustes LDAM")
                temporal = model.temporal_features(features)
                peer_temporal = model.temporal_features(peer_features)
                logits = model.logits_from_temporal_features(temporal)
                loss = ldam_deferred_reweighting_loss(logits, targets, ldam_class_margins, ldam_late_weights if epoch >= 31 else None)
                loss = loss + 0.05 * cross_signer_supervised_contrast_loss(temporal, peer_temporal, targets, temperature=0.20)
            elif focal_ldam_gamma is not None:
                if focal_ldam_gamma < 0.0 or train_prior_log_priors_tensor is not None or classifier_coherence_weight > 0.0 or stochastic_dropout_consistency or canonical_motion_vat or manual_factorial_ldam:
                    raise ValueError("W51 no se combina con W34, W42, W43, W48 ni W50")
                logits = model(features)
                loss = focal_ldam_deferred_reweighting_loss(logits, targets, ldam_class_margins, ldam_late_weights if epoch >= 31 else None, gamma=focal_ldam_gamma)
            elif classifier_coherence_weight < 0.0:
                raise ValueError("W50 requiere peso de coherencia no negativo")
            if classifier_coherence_weight > 0.0:
                if train_prior_log_priors_tensor is not None or stochastic_dropout_consistency or canonical_motion_vat or manual_factorial_ldam:
                    raise ValueError("W50 no se combina con W34, W42, W43 ni W48")
                if not hasattr(model, "head") or not isinstance(model.head, nn.Sequential) or not isinstance(model.head[-1], nn.Linear):
                    raise ValueError("W50 requiere la cabeza lineal W33")
                logits = model(features)
                loss = ldam_deferred_reweighting_loss(logits, targets, ldam_class_margins, ldam_late_weights if epoch >= 31 else None)
                loss = loss + classifier_coherence_weight * classifier_weight_coherence_loss(model.head[-1].weight)
            elif motion_adaptive_temporal_coherence:
                if peer_features is not None or consistency_features is not None or signer_targets is not None or cross_signer_supervised_contrast or focal_ldam_gamma is not None or train_prior_log_priors_tensor is not None or stochastic_dropout_consistency or canonical_motion_vat or manual_factorial_ldam or classifier_coherence_weight != 0.0:
                    raise ValueError("W93 usa solo LDAM-DRW y coherencia temporal intraclase train-only")
                temporal = model.temporal_features(features)
                logits = model.logits_from_temporal_features(temporal)
                loss = ldam_deferred_reweighting_loss(logits, targets, ldam_class_margins, ldam_late_weights if epoch >= 31 else None)
                loss = loss + W93_COHERENCE_WEIGHT * motion_adaptive_temporal_coherence_loss(features, temporal)
            elif train_prior_log_priors_tensor is not None:
                if stochastic_dropout_consistency or canonical_motion_vat or manual_factorial_ldam or ldam_late_weights is not None:
                    raise ValueError("W48 no se combina con W34, W42, W43 ni DRW tardío")
                logits = model(features)
                loss = train_prior_logit_adjusted_ldam_loss(logits, targets, ldam_class_margins, train_prior_log_priors_tensor, tau=1.0)
            elif stochastic_dropout_consistency:
                if manual_factorial_ldam or canonical_motion_vat:
                    raise ValueError("W42 no se combina con W34 ni W43")
                loss, logits, _consistency = stochastic_dropout_ldam_loss(model, features, targets, ldam_class_margins, ldam_late_weights if epoch >= 31 else None)
            elif canonical_motion_vat:
                if manual_factorial_ldam:
                    raise ValueError("W43 no se combina con factorización manual W34")
                logits = model(features)
                loss = ldam_deferred_reweighting_loss(logits, targets, ldam_class_margins, ldam_late_weights if epoch >= 31 else None)
                vat_loss, _perturbation = canonical_motion_vat_loss(model, features, epsilon=0.10, xi=0.001, weight=0.10)
                loss = loss + vat_loss
            elif manual_factorial_ldam:
                logits, factor_embeddings = model.forward_with_factor_embeddings(features)
                loss = ldam_deferred_reweighting_loss(logits, targets, ldam_class_margins, ldam_late_weights if epoch >= 31 else None)
                loss = loss + 0.05 * manual_factorial_decorrelation_loss(factor_embeddings)
            else:
                logits = model(features)
                loss = ldam_deferred_reweighting_loss(logits, targets, ldam_class_margins, ldam_late_weights if epoch >= 31 else None)
        elif training and group_dro_log_weights is not None:
            if signer_targets is None:
                raise ValueError("W26 requiere firmantes exclusivamente durante entrenamiento")
            logits = model(features)
            per_sample_losses = nn.functional.cross_entropy(logits, targets, weight=criterion.weight, reduction="none")
            loss = group_dro_weighted_loss(per_sample_losses, signer_targets, group_dro_log_weights)
        elif training and uncertainty_log_variances is not None:
            if consistency_features is None:
                raise ValueError("W24 requiere una pareja cross-signer durante entrenamiento")
            temporal = model.temporal_features(features)
            peer_temporal = model.temporal_features(consistency_features)
            logits = model.logits_from_temporal_features(temporal)
            covariance_loss = log_euclidean_covariance_consistency(temporal, peer_temporal)
            loss = uncertainty_weighted_two_loss(criterion(logits, targets), covariance_loss, uncertainty_log_variances)
        elif training and use_log_euclidean_covariance_consistency:
            if consistency_features is None:
                raise ValueError("W23 requiere una pareja cross-signer durante entrenamiento")
            temporal = model.temporal_features(features)
            peer_temporal = model.temporal_features(consistency_features)
            logits = model.logits_from_temporal_features(temporal)
            covariance_loss = log_euclidean_covariance_consistency(temporal, peer_temporal)
            loss = criterion(logits, targets) + 0.05 * covariance_loss
        elif training and hopfield_prototype_memory:
            logits, memory_logits = model.forward_with_memory(features)
            loss = criterion(logits, targets) + 0.10 * criterion(memory_logits, targets)
        elif training and contrastive_soft_dtw_alignment:
            if peer_features is None or consistency_features is None:
                raise ValueError("W21 requiere positivo cross-signer y negativo same-signer durante entrenamiento")
            anchor_temporal = model.temporal_features(features)
            positive_temporal = model.temporal_features(peer_features)
            negative_temporal = model.temporal_features(consistency_features)
            logits = model.logits_from_temporal_features(anchor_temporal)
            alignment_loss = temporal_contrastive_soft_dtw_loss(anchor_temporal, positive_temporal, negative_temporal)
            loss = criterion(logits, targets) + 0.10 * alignment_loss
        elif training and cross_signer_feature_mixup:
            if peer_features is None:
                raise ValueError("W19 requiere una pareja cross-signer durante entrenamiento")
            original_temporal = model.temporal_features(features)
            peer_temporal = model.temporal_features(peer_features)
            logits = model.logits_from_temporal_features(original_temporal)
            mixed_logits = model.logits_from_temporal_features(0.5 * (original_temporal + peer_temporal))
            loss = 0.5 * criterion(logits, targets) + 0.5 * criterion(mixed_logits, targets)
        elif training and soft_presence_weight:
            if signer_targets is None:
                raise ValueError("La ponderación sucesora requiere pesos de presencia train-only")
            logits = model(features)
            per_sample = nn.functional.cross_entropy(logits, targets, reduction="none")
            normalized = signer_targets / signer_targets.mean().clamp_min(1e-6)
            loss = (per_sample * normalized).mean()
        elif training and masked_hand_reconstruction:
            if not hasattr(model, "forward_with_reconstruction"):
                raise ValueError("La candidata requiere un modelo con decodificador de reconstrucción")
            masked_features, reconstruction_targets, reconstruction_mask = masked_hand_reconstruction_view(features)
            logits, reconstructed = model.forward_with_reconstruction(masked_features)
            reconstruction_loss = masked_hand_reconstruction_loss(reconstructed, reconstruction_targets, reconstruction_mask)
            loss = criterion(logits, targets) + 0.10 * reconstruction_loss
            reconstruction_loss_total += reconstruction_loss.detach().item() * features.size(0)
        elif training and uniform_label_smoothing:
            logits = model(features)
            loss = nn.functional.cross_entropy(logits, targets, label_smoothing=0.05)
        elif training and ecoc_auxiliary_head:
            if not hasattr(model, "forward_with_ecoc") or not hasattr(model, "ecoc_targets"):
                raise ValueError("La candidata ECOC requiere cabeza y códigos auxiliares")
            logits, code_logits = model.forward_with_ecoc(features)
            code_loss = nn.functional.binary_cross_entropy_with_logits(code_logits, model.ecoc_targets(targets))
            loss = criterion(logits, targets) + 0.05 * code_loss
        elif training and signer_vrex:
            if signer_targets is None:
                raise ValueError("V-REx requiere firmantes exclusivamente durante entrenamiento")
            logits = model(features)
            loss = criterion(logits, targets) if epoch is not None and epoch <= 5 else signer_vrex_loss(logits, targets, signer_targets, variance_weight=1.0)
        elif training and signer_targets is not None and signer_loss_weight > 0:
            logits, signer_logits = model.forward_with_signer(features)
            signer_loss = nn.functional.cross_entropy(signer_logits, signer_targets)
            loss = criterion(logits, targets) + signer_loss_weight * signer_loss
            signer_loss_total += signer_loss.detach().item() * features.size(0)
        elif training and episodic_real_reference:
            temporal = model.forward_features(features)
            logits = model.head(temporal)
            loss = criterion(logits, targets) + 0.25 * episodic_real_reference_loss(temporal.mean(dim=2), targets, temperature=0.20)
        elif training and successor_temporal_relation_pairs:
            if peer_features is None:
                raise ValueError("Los pares relacionales sucesores requieren una pareja real cross-signer")
            temporal = model.forward_features(features)
            peer_temporal = model.forward_features(peer_features)
            logits = model.head(temporal)
            loss = criterion(logits, targets) + 0.05 * temporal_relation_pair_loss(temporal, peer_temporal)
        elif training and successor_selective_core_relation:
            if peer_features is None:
                raise ValueError("La relación selectiva requiere una pareja real cross-signer")
            logits = model(features)
            loss = criterion(logits, targets) + 0.03 * selective_core_relation_loss(model, features, peer_features)
        else:
            logits = model(features)
            loss = criterion(logits, targets)
        if training and consistency_features is not None and consistency_loss_weight > 0:
            if mean_teacher is None:
                augmented_logits = model(consistency_features)
                loss = loss + consistency_loss_weight * symmetric_prediction_consistency(logits, augmented_logits, consistency_temperature)
            else:
                mean_teacher.train(False)
                with torch.no_grad():
                    teacher_logits = mean_teacher(features)
                augmented_logits = model(consistency_features)
                loss = loss + consistency_loss_weight * mean_teacher_temporal_consistency(augmented_logits, teacher_logits, consistency_temperature)
        if training and use_position_velocity_representation_consistency:
            if peer_features is not None or consistency_features is not None or signer_targets is not None or mean_teacher is not None:
                raise ValueError("W31 usa exclusivamente CE W3 y dos máscaras de canal internas")
            loss = loss + 0.05 * position_velocity_representation_consistency(model, features)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if uncertainty_log_variances is not None:
                torch.nn.utils.clip_grad_norm_([uncertainty_log_variances], max_norm=1.0)
            optimizer.step()
            if mean_teacher is not None:
                update_ema_teacher(model, mean_teacher, momentum=0.99)
        running_loss += loss.detach().item() * features.size(0)
        all_logits.append(logits.detach())
        all_targets.append(targets.detach())
    if not all_logits:
        return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0, "support": 0, "confusion_matrix": []}
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    result = metrics_from_logits(logits, targets, classes)
    result["loss"] = float(running_loss / max(1, result["support"]))
    if signer_loss_weight > 0:
        result["signer_loss"] = float(signer_loss_total / max(1, result["support"]))
    if masked_hand_reconstruction:
        result["masked_hand_reconstruction_loss"] = float(reconstruction_loss_total / max(1, result["support"]))
    if uncertainty_log_variances is not None:
        result["uncertainty_log_variances"] = [float(value) for value in uncertainty_log_variances.detach().cpu()]
    if group_dro_log_weights is not None:
        result["group_dro_weights"] = [float(value) for value in torch.softmax(group_dro_log_weights.detach().cpu(), dim=0)]
    return result


def compact_epoch_metrics(metrics: dict) -> dict:
    """Conserva métricas escalares por época; la matriz final vive en test."""
    return {key: value for key, value in metrics.items() if key != "confusion_matrix"}


def save_checkpoint(path: Path, model, args, labels, history, feature_shape, signer_labels: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "task": args.task,
        "labels": labels,
        "feature_shape": list(feature_shape),
        "model_class": model.__class__.__name__,
        "seed": args.seed,
        "history": history,
        "parameters": parameter_count(model),
        "signer_labels": list(signer_labels or []),
        "args": vars(args).copy(),
    }, path)


def json_safe_cli_args(args) -> dict:
    """Convierte rutas CLI a texto al escribir reportes JSON reproducibles."""
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


# TODO: partir main 350→3 helpers: setup_training, run_epoch_split, finalize_training
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASK_CONFIG), required=True)
    parser.add_argument("--frames", type=int, default=None, help="Sobrescribe los frames del contrato, solo si el cache ya está validado")
    parser.add_argument("--feature-dim", type=int, default=None, help="Sobrescribe la dimensión del baseline o PTA-LSM")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--temporal-warp-min", type=float, default=1.0, help="Exponente mínimo W8; 1 desactiva la deformación")
    parser.add_argument("--temporal-warp-max", type=float, default=1.0, help="Exponente máximo W8; 1 desactiva la deformación")
    parser.add_argument("--signer-loss-weight", type=float, default=0.10, help="Peso W9 y escala de reversión de gradiente")
    parser.add_argument("--temporal-consistency-weight", type=float, default=0.10, help="Peso W18 de consistencia predictiva")
    parser.add_argument("--temporal-consistency-temperature", type=float, default=2.0, help="Temperatura W18 de consistencia predictiva")
    parser.add_argument("--pretrain-epochs", type=int, default=20, help="Épocas autocontenidas W27 sobre solo train")
    parser.add_argument("--pretrain-lr", type=float, default=1e-3, help="Learning rate AdamW de preentrenamiento W27")
    parser.add_argument("--energy-density-uniform-mass", type=float, default=None, help="Cuota uniforme η de W71; el valor se fija antes de test")
    parser.add_argument("--confusion-spectral-weight", type=float, default=None, help="Peso λ W72, fijado mediante validation antes de test")
    parser.add_argument("--signer-covariance-weight", type=float, default=None, help="Peso λ W74, fijado mediante validation antes de test")
    parser.add_argument("--w89-presence-cache-root", type=Path, default=None, help="Caché W84 `(30,354)` solo para construir el ranking train W89")
    parser.add_argument("--skip-test-evaluation", action="store_true", help="W71/W72: mantiene test cerrado mientras se selecciona por validation")
    args = parser.parse_args()

    seed_everything(args.seed)
    config = dict(TASK_CONFIG[args.task])
    if args.frames is not None:
        config["frames"] = args.frames
    if args.feature_dim is not None:
        config["features"] = args.feature_dim
    expected = (config["frames"], config["features"])
    if args.temporal_warp_min <= 0 or args.temporal_warp_max <= 0 or args.temporal_warp_min > args.temporal_warp_max:
        raise SystemExit("Los límites de temporal warp deben ser positivos y min <= max")
    temporal_warp_range = None
    if not (np.isclose(args.temporal_warp_min, 1.0) and np.isclose(args.temporal_warp_max, 1.0)):
        if args.task != "isolated_word" or expected[1] != 352:
            raise SystemExit("W8 temporal warp solo admite isolated_word con feature_dim=352")
        temporal_warp_range = (args.temporal_warp_min, args.temporal_warp_max)
    if args.signer_loss_weight <= 0:
        raise SystemExit("signer-loss-weight debe ser positivo")
    if args.temporal_consistency_weight <= 0 or args.temporal_consistency_temperature <= 0:
        raise SystemExit("Los parámetros de consistencia temporal deben ser positivos")
    if args.pretrain_epochs <= 0 or args.pretrain_lr <= 0:
        raise SystemExit("Los parámetros de preentrenamiento W27 deben ser positivos")
    energy_density_uniform_mass = args.energy_density_uniform_mass
    if energy_density_uniform_mass is not None:
        if args.task != "isolated_word_energy_density_reparameterized_ldam" or not 0.0 <= energy_density_uniform_mass <= 1.0:
            raise SystemExit("energy-density-uniform-mass solo admite W71 y valores dentro de [0,1]")
    if args.task == "isolated_word_energy_density_reparameterized_ldam" and energy_density_uniform_mass is None:
        raise SystemExit("W71 requiere --energy-density-uniform-mass explícito")
    confusion_spectral_weight = args.confusion_spectral_weight
    if confusion_spectral_weight is not None:
        if args.task != "isolated_word_confusion_spectral_ldam" or confusion_spectral_weight not in {0.005, 0.010, 0.020}:
            raise SystemExit("confusion-spectral-weight solo admite W72 y λ∈{0.005,0.010,0.020}")
    if args.task == "isolated_word_confusion_spectral_ldam" and confusion_spectral_weight is None:
        raise SystemExit("W72 requiere --confusion-spectral-weight explícito")
    signer_covariance_weight = args.signer_covariance_weight
    if signer_covariance_weight is not None:
        if args.task != "isolated_word_signer_covariance_alignment_ldam" or signer_covariance_weight not in {0.25, 0.50, 1.00}:
            raise SystemExit("signer-covariance-weight solo admite W74 y λ∈{0.25,0.50,1.00}")
    if args.task == "isolated_word_signer_covariance_alignment_ldam" and signer_covariance_weight is None:
        raise SystemExit("W74 requiere --signer-covariance-weight explícito")
    if args.skip_test_evaluation and args.task not in {"isolated_word_energy_density_reparameterized_ldam", "isolated_word_confusion_spectral_ldam", "isolated_word_signer_covariance_alignment_ldam", "recovery_path_signature", "dynamic_alphabet_zenodo", "successor_positions126", "successor_positions126_train_augmented", "successor_episodic_real_reference", "successor_temporal_relation_pairs", "successor_selective_core_relation", "successor_intramanual_bone166", "successor_intramanual_kinematic196", "successor_signer_stratified_batch", "successor_soft_presence_weight", "successor_masked_hand_reconstruction", "successor_intraclip_style_normalization", "successor_temporal_pyramid_pooling", "successor_logavgexp_temporal_pooling", "successor_uniform_label_smoothing", "successor_train_only_swa", "successor_shared_bilateral_tcn", "successor_ecoc_auxiliary_head", "successor_signer_vrex", "successor_fixed_hand_graph_tcn", "successor_arc_length_frame_reindexing", "successor_bidirectional_gru", "successor_cosine_classifier", "successor_spectral_tcn", "successor_global_wrist132", "successor_wrist_velocity132"}:
        raise SystemExit("skip-test-evaluation se reserva para tareas seleccionadas por validation, incluido el control sucesor positions126")
    if args.w89_presence_cache_root is not None and args.task != "isolated_word_w89_class_conditional_hand_quality_curriculum_ldam":
        raise SystemExit("w89-presence-cache-root solo admite W89")
    if args.task == "isolated_word_w89_class_conditional_hand_quality_curriculum_ldam" and args.w89_presence_cache_root is None:
        raise SystemExit("W89 requiere --w89-presence-cache-root para construir su ranking train")
    rows = read_manifest(args.manifest, args.task)
    if args.limit_per_class is not None:
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["label_lsm"], []).append(row)
        rows = [row for label in sorted(grouped) for row in grouped[label][:args.limit_per_class]]
    labels_list = class_labels(rows)
    labels = {label: index for index, label in enumerate(labels_list)}
    if not labels_list:
        raise SystemExit("El manifiesto no contiene filas de la tarea solicitada")
    missing = [str(feature_path(row, args.cache_root)) for row in rows if not feature_path(row, args.cache_root).is_file()]
    if missing:
        sample = "\n".join(missing[:10])
        raise SystemExit(f"Faltan {len(missing)} caches .npy; ejemplos:\n{sample}")

    train_rows = [row for row in rows if row.get("split_active") == "train"]
    val_rows = [row for row in rows if row.get("split_active") == "validation"]
    test_rows = [row for row in rows if row.get("split_active") == "test"]
    if not train_rows or not val_rows:
        raise SystemExit("Se requieren filas train y validation asignadas por firmante")
    train_counts = Counter(row["label_lsm"] for row in train_rows)
    weights = torch.tensor([len(train_rows) / max(1, len(labels_list) * train_counts.get(label, 0)) for label in labels_list], dtype=torch.float32)
    ldam_counts = torch.tensor([train_counts.get(label, 0) for label in labels_list], dtype=torch.float32)
    signer_task = args.task in {"isolated_word_signer_invariant", "isolated_word_group_dro_signer", "isolated_word_signer_covariance_alignment_ldam", "isolated_word_w85_train_signer_group_dro_ldam", "successor_signer_vrex"}
    signer_labels = sorted({row.get("signer_id", "") for row in train_rows}) if signer_task else []
    if signer_task and (len(signer_labels) < 2 or "" in signer_labels):
        raise SystemExit("La tarea por firmante requiere al menos dos signer_id válidos exclusivamente en train")
    signer_index = {signer: index for index, signer in enumerate(signer_labels)}
    w89_quality_by_sample = w89_train_quality_by_sample(train_rows, args.w89_presence_cache_root) if args.task == "isolated_word_w89_class_conditional_hand_quality_curriculum_ldam" else None

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Se solicitó CUDA pero no está disponible")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model_kwargs = {"frames": config["frames"], "feature_dim": config["features"], "classes": len(labels_list)}
    if args.task == "isolated_word_signer_invariant":
        model_kwargs |= {"signer_classes": len(signer_labels), "adversarial_scale": args.signer_loss_weight}
    model = build_model(args.task, **model_kwargs)
    model.to(device)
    ldam_deferred_reweighting = args.task in {"isolated_word_ldam_deferred_reweighting", "isolated_word_manual_factorial_ldam", "isolated_word_body_anchor_residual_ldam", "isolated_word_quality_gated_body_anchor_ldam", "isolated_word_palm_axis_residual_ldam", "isolated_word_latent_style_mix_ldam", "isolated_word_stochastic_dropout_consistency_ldam", "isolated_word_canonical_motion_vat_ldam", "isolated_word_motion_biased_attentive_pooling_ldam", "isolated_word_energy_phase_residual_ldam", "isolated_word_channel_recalibration_ldam", "isolated_word_hand_branch_structural_dropout_ldam", "isolated_word_dominant_hand_canonicalization_ldam", "isolated_word_multiscale_temporal_difference_residual_ldam", "isolated_word_dct_spectral_residual_ldam", "isolated_word_cosine_classifier_ldam", "isolated_word_train_prior_logit_adjusted_ldam", "isolated_word_fixed_hand_graph_residual_ldam", "isolated_word_classifier_coherence_ldam", "isolated_word_focal_ldam_deferred_reweighting", "isolated_word_class_balanced_sampling_ldam", "isolated_word_path_signature_early_fusion_ldam"}
    if args.task == "isolated_word_cross_signer_supervised_contrast_ldam":
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_global_inplane_rotation_ldam":
        ldam_deferred_reweighting = True
    velocity_magnitude_canonicalization = args.task == "isolated_word_velocity_magnitude_canonicalization_ldam"
    if velocity_magnitude_canonicalization:
        ldam_deferred_reweighting = True
    original_duration_aware = args.task == "isolated_word_original_duration_aware_ldam"
    if original_duration_aware:
        ldam_deferred_reweighting = True
    relative_time_coordinates = args.task == "isolated_word_relative_time_coordinates_ldam"
    if relative_time_coordinates:
        ldam_deferred_reweighting = True
    feature_standardization_stats = load_w63_train_feature_stats() if args.task == "isolated_word_train_only_feature_standardization_ldam" else None
    if feature_standardization_stats is not None:
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_parallel_receptive_field_tcn_ldam":
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_parameter_free_temporal_shift_ldam":
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_temporal_weight_standardization_ldam":
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_depthwise_separable_temporal_tcn_ldam":
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_linear_stochastic_depth_ldam":
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_rezero_temporal_residual_ldam":
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_energy_density_reparameterized_ldam":
        ldam_deferred_reweighting = True
    confusion_spectral_ldam = args.task == "isolated_word_confusion_spectral_ldam"
    if confusion_spectral_ldam:
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_adaptive_temporal_prototype_ldam":
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_compositional_temporal_prototype_ldam":
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_shape_motion_bilinear_ldam":
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_dense_hand_spectral_signature_ldam":
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_w80_interior_hand_reconstruction_ldam":
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_w81_activity_boundary_ldam":
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_w83_world_hand_geometry_residual_ldam":
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_w84_explicit_hand_presence_ldam":
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_w85_train_signer_group_dro_ldam":
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_w89_class_conditional_hand_quality_curriculum_ldam":
        ldam_deferred_reweighting = True
    if args.task == "isolated_word_w93_motion_adaptive_temporal_coherence_ldam":
        ldam_deferred_reweighting = True
    signer_covariance_alignment_ldam = args.task == "isolated_word_signer_covariance_alignment_ldam"
    if signer_covariance_alignment_ldam:
        ldam_deferred_reweighting = True
    decoupled_classifier_retraining = args.task == "isolated_word_decoupled_classifier_retraining_ldam"
    if decoupled_classifier_retraining:
        if args.epochs != 60:
            raise SystemExit("W59 exige exactamente 60 épocas: 45 de representación y 15 de reclasificación")
        ldam_deferred_reweighting = True
    manual_factorial_ldam = args.task == "isolated_word_manual_factorial_ldam"
    successor_plain_cross_entropy = args.task in {"successor_positions126", "successor_positions126_train_augmented", "successor_episodic_real_reference", "successor_temporal_relation_pairs", "successor_selective_core_relation", "successor_intramanual_bone166", "successor_intramanual_kinematic196", "successor_signer_stratified_batch", "successor_soft_presence_weight", "successor_masked_hand_reconstruction", "successor_intraclip_style_normalization", "successor_temporal_pyramid_pooling", "successor_logavgexp_temporal_pooling", "successor_uniform_label_smoothing", "successor_train_only_swa", "successor_shared_bilateral_tcn", "successor_ecoc_auxiliary_head", "successor_signer_vrex", "successor_fixed_hand_graph_tcn", "successor_arc_length_frame_reindexing", "successor_bidirectional_gru", "successor_cosine_classifier", "successor_spectral_tcn", "successor_global_wrist132", "successor_wrist_velocity132"}
    criterion = nn.CrossEntropyLoss() if (ldam_deferred_reweighting or successor_plain_cross_entropy) else nn.CrossEntropyLoss(weight=weights.to(device))
    ldam_class_margins = ldam_margins(ldam_counts).to(device) if ldam_deferred_reweighting else None
    train_prior_logit_adjusted = args.task == "isolated_word_train_prior_logit_adjusted_ldam"
    class_balanced_sampling = args.task == "isolated_word_class_balanced_sampling_ldam"
    ldam_late_weights = None if (train_prior_logit_adjusted or class_balanced_sampling) else (effective_number_class_weights(ldam_counts).to(device) if ldam_deferred_reweighting else None)
    prior_log_values = train_prior_log_priors(ldam_counts).to(device) if train_prior_logit_adjusted else None
    temporal_consistency_range = (0.92, 1.08) if args.task in {"isolated_word_temporal_prediction_consistency", "isolated_word_cross_signer_temporal_consistency", "isolated_word_pretrained_temporal_consistency", "isolated_word_mean_teacher_temporal_consistency", "isolated_word_ema_weight_average_inference"} else None
    cross_signer_feature_mixup = args.task in {"isolated_word_cross_signer_feature_mixup", "isolated_word_cross_signer_temporal_consistency"}
    cross_signer_supervised_contrast = args.task == "isolated_word_cross_signer_supervised_contrast_ldam"
    combined_cross_signer_consistency = args.task == "isolated_word_cross_signer_temporal_consistency"
    contrastive_soft_dtw_alignment = args.task == "isolated_word_contrastive_soft_dtw_alignment"
    hopfield_prototype_memory = args.task == "isolated_word_hopfield_prototype_memory"
    covariance_pairing = args.task in {"isolated_word_log_euclidean_covariance_consistency", "isolated_word_uncertainty_balanced_covariance"}
    uncertainty_balanced_covariance = args.task == "isolated_word_uncertainty_balanced_covariance"
    sharpness_aware_w3 = args.task == "isolated_word_sharpness_aware_w3"
    group_dro_signer = args.task == "isolated_word_group_dro_signer"
    group_dro_ldam = args.task == "isolated_word_w85_train_signer_group_dro_ldam"
    masked_temporal_pretraining = args.task in {"isolated_word_masked_temporal_pretraining", "isolated_word_pretrained_temporal_consistency"}
    uniform_label_smoothing = args.task == "successor_uniform_label_smoothing"
    train_only_swa = args.task == "successor_train_only_swa"
    ecoc_auxiliary_head = args.task == "successor_ecoc_auxiliary_head"
    signer_vrex = args.task == "successor_signer_vrex"
    temporal_order_pretraining = args.task == "isolated_word_temporal_order_pretraining"
    mean_teacher_temporal_consistency = args.task == "isolated_word_mean_teacher_temporal_consistency"
    position_velocity_representation_consistency = args.task == "isolated_word_position_velocity_representation_consistency"
    stochastic_dropout_consistency = args.task == "isolated_word_stochastic_dropout_consistency_ldam"
    canonical_motion_vat = args.task == "isolated_word_canonical_motion_vat_ldam"
    motion_adaptive_temporal_coherence = args.task == "isolated_word_w93_motion_adaptive_temporal_coherence_ldam"
    classifier_coherence = args.task == "isolated_word_classifier_coherence_ldam"
    focal_ldam = args.task == "isolated_word_focal_ldam_deferred_reweighting"
    hand_branch_dropout_probability = 0.15 if args.task == "isolated_word_hand_branch_structural_dropout_ldam" else None
    dominant_hand_canonicalization = args.task == "isolated_word_dominant_hand_canonicalization_ldam"
    global_inplane_rotation_range = (-10.0, 10.0) if args.task == "isolated_word_global_inplane_rotation_ldam" else None
    ema_weight_average_inference = args.task == "isolated_word_ema_weight_average_inference"
    needs_clean_pretraining = masked_temporal_pretraining or temporal_order_pretraining
    group_dro_log_weights = torch.zeros(len(signer_labels), device=device) if (group_dro_signer or group_dro_ldam) else None
    uncertainty_log_variances = nn.Parameter(torch.zeros(2, device=device)) if uncertainty_balanced_covariance else None
    optimizer_groups = [{"params": model.parameters(), "weight_decay": args.weight_decay}]
    if uncertainty_log_variances is not None:
        optimizer_groups.append({"params": [uncertainty_log_variances], "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(optimizer_groups, lr=args.lr)
    mean_teacher = copy.deepcopy(model).to(device) if (mean_teacher_temporal_consistency or ema_weight_average_inference) else None
    if mean_teacher is not None:
        mean_teacher.requires_grad_(False)
        mean_teacher.train(False)
    successor_train_augmentation = args.task == "successor_positions126_train_augmented"
    successor_relation_pairing = args.task in {"successor_temporal_relation_pairs", "successor_selective_core_relation"}
    soft_presence_weight = args.task == "successor_soft_presence_weight"
    masked_hand_reconstruction = args.task == "successor_masked_hand_reconstruction"
    train_dataset = CachedSequenceDataset(train_rows, args.cache_root, labels, expected, temporal_warp_range=temporal_warp_range, signer_labels=signer_index or None, temporal_consistency_range=temporal_consistency_range, cross_signer_pairing=cross_signer_feature_mixup or cross_signer_supervised_contrast, combined_cross_signer_consistency=combined_cross_signer_consistency, alignment_triplets=contrastive_soft_dtw_alignment, covariance_pairing=covariance_pairing, hand_branch_dropout_probability=hand_branch_dropout_probability, dominant_hand_canonicalization=dominant_hand_canonicalization, global_inplane_rotation_range=global_inplane_rotation_range, velocity_magnitude_canonicalization=velocity_magnitude_canonicalization, original_duration_aware=original_duration_aware, relative_time_coordinates=relative_time_coordinates, feature_standardization_stats=feature_standardization_stats, energy_density_uniform_mass=energy_density_uniform_mass, successor_train_augmentation=successor_train_augmentation, augmentation_seed=args.seed, successor_relation_pairing=successor_relation_pairing, soft_presence_weight=soft_presence_weight)
    episodic_real_reference = args.task == "successor_episodic_real_reference"
    signer_stratified = args.task == "successor_signer_stratified_batch"
    train_sampler = build_class_balanced_train_sampler(train_rows, args.seed) if class_balanced_sampling else None
    episode_sampler = CrossSignerEpisodeBatchSampler(train_rows, seed=args.seed) if episodic_real_reference else None
    signer_sampler = SignerStratifiedBatchSampler(train_rows, seed=args.seed) if signer_stratified else None
    train_loader = DataLoader(train_dataset, batch_sampler=episode_sampler or signer_sampler, num_workers=args.num_workers, pin_memory=device.type == "cuda") if (episode_sampler is not None or signer_sampler is not None) else DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    classifier_retraining_sampler = build_class_balanced_train_sampler(train_rows, args.seed) if decoupled_classifier_retraining else None
    classifier_retraining_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=classifier_retraining_sampler, num_workers=args.num_workers, pin_memory=device.type == "cuda") if classifier_retraining_sampler is not None else None
    pretrain_loader = DataLoader(CachedSequenceDataset(train_rows, args.cache_root, labels, expected), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda") if needs_clean_pretraining else None
    val_loader = DataLoader(CachedSequenceDataset(val_rows, args.cache_root, labels, expected, dominant_hand_canonicalization=dominant_hand_canonicalization, velocity_magnitude_canonicalization=velocity_magnitude_canonicalization, original_duration_aware=original_duration_aware, relative_time_coordinates=relative_time_coordinates, feature_standardization_stats=feature_standardization_stats, energy_density_uniform_mass=energy_density_uniform_mass), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(CachedSequenceDataset(test_rows, args.cache_root, labels, expected, dominant_hand_canonicalization=dominant_hand_canonicalization, velocity_magnitude_canonicalization=velocity_magnitude_canonicalization, original_duration_aware=original_duration_aware, relative_time_coordinates=relative_time_coordinates, feature_standardization_stats=feature_standardization_stats, energy_density_uniform_mass=energy_density_uniform_mass), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda") if test_rows and not args.skip_test_evaluation else None

    args.out.mkdir(parents=True, exist_ok=True)
    pretrain_history = []
    if masked_temporal_pretraining:
        teacher = copy.deepcopy(model).to(device)
        teacher.requires_grad_(False)
        pretrain_optimizer = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay)
        pretrain_start = time.time()
        for pretrain_epoch in range(1, args.pretrain_epochs + 1):
            pretrain_metrics = run_masked_temporal_pretraining(model, teacher, pretrain_loader, pretrain_optimizer, device)
            pretrain_record = {"epoch": pretrain_epoch, **pretrain_metrics, "elapsed_seconds": time.time() - pretrain_start}
            pretrain_history.append(pretrain_record)
            print(json.dumps({"pretrain": pretrain_record}, ensure_ascii=False), flush=True)
        del teacher
    elif temporal_order_pretraining:
        order_head = nn.Linear(256, 1).to(device)
        pretrain_optimizer = torch.optim.AdamW([*model.parameters(), *order_head.parameters()], lr=args.pretrain_lr, weight_decay=args.weight_decay)
        pretrain_start = time.time()
        for pretrain_epoch in range(1, args.pretrain_epochs + 1):
            pretrain_metrics = run_temporal_order_pretraining(model, order_head, pretrain_loader, pretrain_optimizer, device)
            pretrain_record = {"epoch": pretrain_epoch, **pretrain_metrics, "elapsed_seconds": time.time() - pretrain_start}
            pretrain_history.append(pretrain_record)
            print(json.dumps({"pretrain": pretrain_record}, ensure_ascii=False), flush=True)
        del order_head
    history = []
    best_score = -1.0
    stale = 0
    t_start = time.time()
    swa_model = AveragedModel(model, use_buffers=True).to(device) if train_only_swa else None
    swa_start_epoch = 21
    for epoch in range(1, args.epochs + 1):
        phase = "representation"
        epoch_loader = train_loader
        train_dataset.set_augmentation_epoch(epoch)
        if episode_sampler is not None:
            episode_sampler.set_epoch(epoch)
        if signer_sampler is not None:
            signer_sampler.set_epoch(epoch)
        epoch_criterion = criterion
        epoch_margins = ldam_class_margins
        epoch_late_weights = ldam_late_weights
        if decoupled_classifier_retraining and epoch == 46:
            head_parameters = freeze_for_decoupled_classifier_retraining(model)
            optimizer = torch.optim.AdamW(head_parameters, lr=0.0005, weight_decay=args.weight_decay)
        if decoupled_classifier_retraining and epoch >= 46:
            phase = "classifier_retraining"
            epoch_loader = classifier_retraining_loader
            epoch_criterion = nn.CrossEntropyLoss()
            epoch_margins = None
            epoch_late_weights = None
        if w89_quality_by_sample is not None:
            epoch_rows, phase = w89_curriculum_rows(train_rows, w89_quality_by_sample, epoch)
            epoch_dataset = CachedSequenceDataset(epoch_rows, args.cache_root, labels, expected)
            epoch_generator = torch.Generator().manual_seed(int(args.seed) + 8900 + epoch)
            epoch_loader = DataLoader(epoch_dataset, batch_size=args.batch_size, shuffle=True, generator=epoch_generator, num_workers=args.num_workers, pin_memory=device.type == "cuda")
        train_metrics = run_epoch(model, epoch_loader, epoch_criterion, device, optimizer, len(labels_list), signer_loss_weight=args.signer_loss_weight if args.task == "isolated_word_signer_invariant" else 0.0, consistency_loss_weight=args.temporal_consistency_weight if temporal_consistency_range else 0.0, consistency_temperature=args.temporal_consistency_temperature, cross_signer_feature_mixup=cross_signer_feature_mixup, cross_signer_supervised_contrast=cross_signer_supervised_contrast, contrastive_soft_dtw_alignment=contrastive_soft_dtw_alignment, hopfield_prototype_memory=hopfield_prototype_memory, use_log_euclidean_covariance_consistency=covariance_pairing, uncertainty_log_variances=uncertainty_log_variances, use_sharpness_aware_minimization=sharpness_aware_w3, group_dro_log_weights=group_dro_log_weights, group_dro_ldam=group_dro_ldam, mean_teacher=mean_teacher, use_position_velocity_representation_consistency=position_velocity_representation_consistency, manual_factorial_ldam=manual_factorial_ldam, stochastic_dropout_consistency=stochastic_dropout_consistency, canonical_motion_vat=canonical_motion_vat, motion_adaptive_temporal_coherence=motion_adaptive_temporal_coherence, episodic_real_reference=episodic_real_reference, successor_temporal_relation_pairs=args.task == "successor_temporal_relation_pairs", successor_selective_core_relation=args.task == "successor_selective_core_relation", soft_presence_weight=soft_presence_weight, masked_hand_reconstruction=masked_hand_reconstruction, uniform_label_smoothing=uniform_label_smoothing, ecoc_auxiliary_head=ecoc_auxiliary_head, signer_vrex=signer_vrex, train_prior_log_priors_tensor=prior_log_values, classifier_coherence_weight=0.10 if classifier_coherence else 0.0, focal_ldam_gamma=1.0 if focal_ldam else None, ldam_class_margins=epoch_margins, ldam_late_weights=epoch_late_weights, confusion_spectral_weight=confusion_spectral_weight if confusion_spectral_ldam and epoch >= 31 else 0.0, signer_covariance_weight=signer_covariance_weight if signer_covariance_alignment_ldam and epoch >= 31 else 0.0, signer_covariance_task=signer_covariance_alignment_ldam, epoch=epoch, frozen_encoder_eval=decoupled_classifier_retraining and epoch >= 46)
        if swa_model is not None and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            update_bn(train_loader, swa_model, device=device)
            evaluation_model = swa_model
        else:
            evaluation_model = ema_inference_model(model, mean_teacher, ema_weight_average_inference)
        with torch.inference_mode():
            val_metrics = run_epoch(evaluation_model, val_loader, criterion, device, None, len(labels_list))
        record = {
            "epoch": epoch,
            "phase": phase,
            "train": compact_epoch_metrics(train_metrics),
            "validation": compact_epoch_metrics(val_metrics),
            "elapsed_seconds": time.time() - t_start,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        score = val_metrics["macro_f1"]
        if score > best_score:
            best_score = score
            stale = 0
            save_checkpoint(args.out / "best.pt", evaluation_model, args, labels_list, history, expected, signer_labels=signer_labels)
        else:
            stale += 1
        if stale >= args.patience and not decoupled_classifier_retraining and (swa_model is None or epoch >= swa_start_epoch):
            print(json.dumps({"early_stopping": True, "epoch": epoch}, ensure_ascii=False), flush=True)
            break

    del mean_teacher
    if test_loader is not None:
        checkpoint = torch.load(args.out / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        with torch.inference_mode():
            test_metrics = run_epoch(model, test_loader, criterion, device, None, len(labels_list))
    else:
        test_metrics = None
    report = {
        "task": args.task,
        "device": str(device),
        "labels": labels_list,
        "feature_shape": list(expected),
        "counts": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
        "parameters": parameter_count(model),
        "signer_labels": signer_labels,
        "best_validation_macro_f1": best_score,
        "test": test_metrics,
        "history": history,
        "pretrain_history": pretrain_history,
        "args": json_safe_cli_args(args) | {"manifest": str(args.manifest), "cache_root": str(args.cache_root), "out": str(args.out)},
    }
    (args.out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    test_summary = compact_epoch_metrics(test_metrics) if test_metrics is not None else None
    print(json.dumps({"done": True, "best_validation_macro_f1": best_score, "test": test_summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())