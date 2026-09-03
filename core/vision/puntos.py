from __future__ import annotations

from typing import Any

import numpy as np


POINTS_PER_HAND = 21
VALUES_PER_POINT = 3
VALUES_PER_HAND = POINTS_PER_HAND * VALUES_PER_POINT
TWO_HAND_KEYPOINTS = VALUES_PER_HAND * 2
SELECTED_POSE_LANDMARKS = (0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26)
SELECTED_FACE_LANDMARKS = (1, 4, 10, 13, 14, 17, 33, 133, 263, 362, 61, 291, 78, 308, 152, 199)
POSE_VALUES = len(SELECTED_POSE_LANDMARKS) * 4
FACE_VALUES = len(SELECTED_FACE_LANDMARKS) * VALUES_PER_POINT
RICH_KEYPOINT_SIZE = TWO_HAND_KEYPOINTS + POSE_VALUES + FACE_VALUES
MIN_NORMALIZATION_SCALE = 1e-6
POSE_OFFSET = TWO_HAND_KEYPOINTS
FACE_OFFSET = TWO_HAND_KEYPOINTS + POSE_VALUES


def _flatten_hand(hand_landmarks: Any | None) -> np.ndarray:
    """ flatten hand."""
    if hand_landmarks is None:
        return np.zeros(VALUES_PER_HAND, dtype=np.float32)

    values: list[float] = []
    for point in hand_landmarks.landmark:
        values.extend([float(point.x), float(point.y), float(point.z)])

    if len(values) != VALUES_PER_HAND:
        raise ValueError(f"Expected {VALUES_PER_HAND} values per hand, got {len(values)}")

    return np.array(values, dtype=np.float32)


def _landmark_xyz_array(landmarks: Any | None) -> np.ndarray:
    if landmarks is None:
        return np.zeros((0, VALUES_PER_POINT), dtype=np.float64)
    values = [[float(point.x), float(point.y), float(point.z)] for point in landmarks.landmark]
    return np.asarray(values, dtype=np.float64)


def _anchor_from_indices(points: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    valid = [index for index in indices if index < len(points)]
    if not valid:
        return np.zeros(VALUES_PER_POINT, dtype=np.float32)
    return np.mean(points[valid], axis=0).astype(np.float32)


def _scale_from_indices(
    points: np.ndarray,
    anchor: np.ndarray,
    indices: tuple[int, ...],
) -> float:
    valid = [index for index in indices if index < len(points)]
    if len(valid) >= 2:
        scale = float(np.linalg.norm(points[valid[0], :2] - points[valid[1], :2]))
        if scale > MIN_NORMALIZATION_SCALE:
            return scale
    if len(points) == 0:
        return 1.0
    centered = points - anchor
    scale = float(np.max(np.linalg.norm(centered[:, :2], axis=1)))
    if scale <= MIN_NORMALIZATION_SCALE:
        return 1.0
    return scale


def _normalize_coordinate_points(
    points: np.ndarray,
    *,
    anchor_indices: tuple[int, ...],
    scale_indices: tuple[int, ...],
) -> np.ndarray:
    if len(points) == 0 or not np.any(np.abs(points) > MIN_NORMALIZATION_SCALE):
        return np.zeros_like(points, dtype=np.float32)
    anchor = _anchor_from_indices(points, anchor_indices)
    scale = _scale_from_indices(points, anchor, scale_indices)
    return ((points - anchor) / scale).astype(np.float32)


def _normalize_flat_points(
    flat_values: np.ndarray,
    *,
    values_per_point: int,
    anchor_indices: tuple[int, ...],
    scale_indices: tuple[int, ...],
) -> np.ndarray:
    values = np.asarray(flat_values, dtype=np.float64)
    if values_per_point == VALUES_PER_POINT:
        points = values.reshape((-1, VALUES_PER_POINT))
        return _normalize_coordinate_points(
            points,
            anchor_indices=anchor_indices,
            scale_indices=scale_indices,
        ).reshape(-1)

    points_with_extra = values.reshape((-1, values_per_point))
    coords = points_with_extra[:, :VALUES_PER_POINT]
    extras = points_with_extra[:, VALUES_PER_POINT:]
    normalized = _normalize_coordinate_points(
        coords,
        anchor_indices=anchor_indices,
        scale_indices=scale_indices,
    )
    return np.concatenate([normalized, extras.astype(np.float32)], axis=1).reshape(-1)


def _flatten_normalized_hand(hand_landmarks: Any | None) -> np.ndarray:
    points = _landmark_xyz_array(hand_landmarks)
    if len(points) == 0:
        return np.zeros(VALUES_PER_HAND, dtype=np.float32)
    if len(points) != POINTS_PER_HAND:
        raise ValueError(f"Expected {POINTS_PER_HAND} points per hand, got {len(points)}")

    anchor = points[0]
    scale = _scale_from_indices(points, anchor, tuple(range(POINTS_PER_HAND)))
    normalized = (points - anchor) / scale
    return normalized.astype(np.float32).reshape(-1)


def _flatten_selected_points(
    landmarks: Any | None,
    indices: tuple[int, ...],
    *,
    include_visibility: bool = False,
) -> np.ndarray:
    values_per_point = 4 if include_visibility else VALUES_PER_POINT
    if landmarks is None:
        return np.zeros(len(indices) * values_per_point, dtype=np.float32)

    values: list[float] = []
    points = landmarks.landmark
    for index in indices:
        if index >= len(points):
            values.extend([0.0] * values_per_point)
            continue
        point = points[index]
        values.extend([float(point.x), float(point.y), float(point.z)])
        if include_visibility:
            values.append(float(getattr(point, "visibility", 0.0)))

    expected = len(indices) * values_per_point
    if len(values) != expected:
        raise ValueError(f"Expected {expected} selected landmark values, got {len(values)}")

    return np.array(values, dtype=np.float32)


def _flatten_selected_points_normalized(
    landmarks: Any | None,
    indices: tuple[int, ...],
    *,
    anchor_indices: tuple[int, ...],
    scale_indices: tuple[int, ...],
    include_visibility: bool = False,
) -> np.ndarray:
    values_per_point = 4 if include_visibility else VALUES_PER_POINT
    if landmarks is None:
        return np.zeros(len(indices) * values_per_point, dtype=np.float32)

    points = _landmark_xyz_array(landmarks)
    anchor = _anchor_from_indices(points, anchor_indices)
    scale = _scale_from_indices(points, anchor, scale_indices)

    values: list[float] = []
    source_points = landmarks.landmark
    for index in indices:
        if index >= len(points):
            values.extend([0.0] * values_per_point)
            continue
        normalized = (points[index] - anchor) / scale
        values.extend([float(normalized[0]), float(normalized[1]), float(normalized[2])])
        if include_visibility:
            values.append(float(getattr(source_points[index], "visibility", 0.0)))

    expected = len(indices) * values_per_point
    if len(values) != expected:
        raise ValueError(f"Expected {expected} selected landmark values, got {len(values)}")

    return np.array(values, dtype=np.float32)


def extract_hand_keypoints(results: Any) -> np.ndarray:
    left = _flatten_hand(getattr(results, "left_hand_landmarks", None))
    right = _flatten_hand(getattr(results, "right_hand_landmarks", None))
    return np.concatenate([left, right]).astype(np.float32)


def extract_rich_keypoints(results: Any) -> np.ndarray:
    hands = extract_hand_keypoints(results)
    pose = _flatten_selected_points(
        getattr(results, "pose_landmarks", None),
        SELECTED_POSE_LANDMARKS,
        include_visibility=True,
    )
    face = _flatten_selected_points(
        getattr(results, "face_landmarks", None),
        SELECTED_FACE_LANDMARKS,
    )
    return np.concatenate([hands, pose, face]).astype(np.float32)


def normalize_rich_keypoints(keypoints: np.ndarray) -> np.ndarray:
    values = np.asarray(keypoints, dtype=np.float64)
    if values.shape != (RICH_KEYPOINT_SIZE,):
        raise ValueError(f"Expected rich keypoints shape {(RICH_KEYPOINT_SIZE,)}, got {values.shape}")

    left = _normalize_flat_points(
        values[:VALUES_PER_HAND],
        values_per_point=VALUES_PER_POINT,
        anchor_indices=(0,),
        scale_indices=tuple(range(POINTS_PER_HAND)),
    )
    right = _normalize_flat_points(
        values[VALUES_PER_HAND:TWO_HAND_KEYPOINTS],
        values_per_point=VALUES_PER_POINT,
        anchor_indices=(0,),
        scale_indices=tuple(range(POINTS_PER_HAND)),
    )

    pose = values[POSE_OFFSET:FACE_OFFSET]
    pose_lookup = {landmark_index: index for index, landmark_index in enumerate(SELECTED_POSE_LANDMARKS)}
    pose_normalized = _normalize_flat_points(
        pose,
        values_per_point=4,
        anchor_indices=(pose_lookup[11], pose_lookup[12]),
        scale_indices=(pose_lookup[11], pose_lookup[12]),
    )

    face = values[FACE_OFFSET:]
    face_lookup = {landmark_index: index for index, landmark_index in enumerate(SELECTED_FACE_LANDMARKS)}
    face_normalized = _normalize_flat_points(
        face,
        values_per_point=VALUES_PER_POINT,
        anchor_indices=(face_lookup[1],),
        scale_indices=(face_lookup[33], face_lookup[263]),
    )

    return np.concatenate([left, right, pose_normalized, face_normalized]).astype(np.float32)


def normalize_rich_sequence(sequence: np.ndarray) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != RICH_KEYPOINT_SIZE:
        raise ValueError(f"Expected rich sequence with {RICH_KEYPOINT_SIZE} values per frame, got {values.shape}")
    return np.stack([normalize_rich_keypoints(frame) for frame in values]).astype(np.float32)


def extract_normalized_rich_keypoints(results: Any) -> np.ndarray:
    return normalize_rich_keypoints(extract_rich_keypoints(results))


def extract_keypoints_for_size(results: Any, keypoint_size: int) -> np.ndarray:
    if keypoint_size == TWO_HAND_KEYPOINTS:
        return extract_hand_keypoints(results)
    if keypoint_size == RICH_KEYPOINT_SIZE:
        return extract_normalized_rich_keypoints(results)
    raise ValueError(f"No hay extractor para modelos de {keypoint_size} valores por frame.")