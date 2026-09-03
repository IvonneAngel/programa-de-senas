from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np


MIN_SEQUENCE_NONZERO_RATIO = 0.005
MIN_SEQUENCE_ACTIVE_FRAMES = 3
SIGNAL_EPSILON = 1e-8


def validate_sequence(sequence: np.ndarray, frame_count: int, keypoint_size: int) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    expected = (frame_count, keypoint_size)
    if sequence.shape != expected:
        raise ValueError(f"Expected sequence shape {expected}, got {sequence.shape}")
    return sequence


def sequence_signal_report(
    sequence: np.ndarray,
    *,
    min_nonzero_ratio: float = MIN_SEQUENCE_NONZERO_RATIO,
    min_active_frames: int = MIN_SEQUENCE_ACTIVE_FRAMES,
) -> dict[str, float | int | bool | str]:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.size == 0:
        return {
            "has_signal": False,
            "nonzero_ratio": 0.0,
            "active_frame_count": 0,
            "max_abs": 0.0,
            "issue": "secuencia vacia",
        }

    finite = bool(np.isfinite(sequence).all())
    nonzero_mask = np.abs(sequence) > SIGNAL_EPSILON
    nonzero_ratio = float(np.count_nonzero(nonzero_mask) / sequence.size)
    if sequence.ndim >= 2:
        active_frame_count = int(np.count_nonzero(np.any(nonzero_mask.reshape(sequence.shape[0], -1), axis=1)))
    else:
        active_frame_count = int(np.count_nonzero(nonzero_mask))
    max_abs = float(np.nanmax(np.abs(sequence))) if sequence.size else 0.0

    issue = ""
    has_signal = finite and nonzero_ratio >= min_nonzero_ratio and active_frame_count >= min_active_frames
    if not finite:
        issue = "contiene valores NaN o infinitos"
    elif nonzero_ratio < min_nonzero_ratio:
        issue = "sin senal de landmarks suficiente"
    elif active_frame_count < min_active_frames:
        issue = "muy pocos frames con landmarks activos"

    return {
        "has_signal": has_signal,
        "nonzero_ratio": round(nonzero_ratio, 6),
        "active_frame_count": active_frame_count,
        "max_abs": round(max_abs, 6),
        "issue": issue,
    }


def validate_real_sequence(
    sequence: np.ndarray,
    frame_count: int,
    keypoint_size: int,
    *,
    min_nonzero_ratio: float = MIN_SEQUENCE_NONZERO_RATIO,
    min_active_frames: int = MIN_SEQUENCE_ACTIVE_FRAMES,
) -> np.ndarray:
    sequence = validate_sequence(sequence, frame_count, keypoint_size)
    report = sequence_signal_report(
        sequence,
        min_nonzero_ratio=min_nonzero_ratio,
        min_active_frames=min_active_frames,
    )
    if not report["has_signal"]:
        raise ValueError(str(report["issue"] or "secuencia sin senal real"))
    return sequence


def save_sequence(data_dir: str | Path, label: str, sequence: np.ndarray) -> Path:
    label_dir = Path(data_dir) / label
    label_dir.mkdir(parents=True, exist_ok=True)
    path = label_dir / f"{uuid4().hex}.npy"
    np.save(path, np.asarray(sequence, dtype=np.float32))
    return path


def load_dataset(
    data_dir: str | Path,
    frame_count: int,
    keypoint_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    root = Path(data_dir)
    label_files: dict[str, list[Path]] = {}
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        direct_files = sorted(path.glob("*.npy"))
        if direct_files:
            label_files.setdefault(path.name, []).extend(direct_files)
            continue
        for label_dir in sorted(child for child in path.iterdir() if child.is_dir()):
            nested_files = sorted(label_dir.glob("*.npy"))
            if nested_files:
                label_files.setdefault(label_dir.name, []).extend(nested_files)

    labels = sorted(label_files)
    if not label_files:
        raise ValueError(f"No label folders found in {root}")

    sequences: list[np.ndarray] = []
    targets: list[int] = []

    for label_index, label in enumerate(labels):
        files = sorted(label_files[label])
        if not files:
            continue
        for file_path in files:
            sequence = validate_real_sequence(np.load(file_path), frame_count, keypoint_size)
            sequences.append(sequence)
            targets.append(label_index)

    if not sequences:
        raise ValueError(f"No .npy sequences found in {root}")

    return np.stack(sequences), np.array(targets, dtype=np.int64), labels