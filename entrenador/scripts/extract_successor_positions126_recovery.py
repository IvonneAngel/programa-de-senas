"""Extrae la caché sucesora `(30,126)` desde los frames originales Mendeley."""
from __future__ import annotations

import argparse
import csv
import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


_DETECTOR: Any = None
_MODEL_PATH: str = ''


def initialize_worker(model_path: str, detection_confidence: float, presence_confidence: float, tracking_confidence: float) -> None:
    global _DETECTOR, _MODEL_PATH
    _MODEL_PATH = model_path
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=detection_confidence,
        min_hand_presence_confidence=presence_confidence,
        min_tracking_confidence=tracking_confidence,
    )
    _DETECTOR = vision.HandLandmarker.create_from_options(options)


def natural_frames(source_dir: Path) -> list[Path]:
    frames = [path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() == '.jpg']
    return sorted(frames, key=lambda path: int(''.join(character for character in path.stem if character.isdigit()) or '0'))


def resampled_indices(frame_count: int) -> np.ndarray:
    if frame_count < 1:
        raise ValueError('Un clip debe contener al menos un frame')
    return np.rint(np.linspace(0, frame_count - 1, num=30)).astype(np.int64)


def landmark_vector(frame_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f'No se pudo decodificar {frame_path}')
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = _DETECTOR.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb))
    hands: dict[str, tuple[float, np.ndarray]] = {}
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        if len(landmarks) != 21 or not handedness:
            continue
        category = handedness[0]
        name = str(category.category_name).strip().lower()
        if name not in {'left', 'right'}:
            continue
        score = float(category.score)
        points = np.asarray([[landmark.x, landmark.y, landmark.z] for landmark in landmarks], dtype=np.float32)
# `positions126` codifica forma manual local. La muñeca global se dejó
# fuera de este control y fue evaluada aparte en global_wrist132.
        values = (points - points[0:1]).reshape(63)
        if not np.isfinite(values).all():
            continue
        if name not in hands or score > hands[name][0]:
            hands[name] = (score, values)
    vector = np.zeros(126, dtype=np.float32)
    if 'left' in hands:
        vector[:63] = hands['left'][1]
    if 'right' in hands:
        vector[63:] = hands['right'][1]
    return vector


def extract_one(payload: tuple[dict[str, str], str]) -> dict[str, str]:
    row, cache_root_string = payload
    source_dir = Path(row['source_dir'])
    frames = natural_frames(source_dir)
    output = dict(row)
    output['extractor'] = 'mediapipe_hand_landmarker_0.10.21_threshold_0.10'
    try:
        indices = resampled_indices(len(frames))
        sequence = np.stack([landmark_vector(frames[int(index)]) for index in indices], axis=0)
        if sequence.shape != (30, 126) or not np.isfinite(sequence).all():
            raise ValueError(f'Secuencia inválida {sequence.shape}')
        cache_path = Path(cache_root_string) / output['feature_path']
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix('.npy.tmp')
        with temporary_path.open('wb') as handle:
            np.save(handle, sequence, allow_pickle=False)
        os.replace(temporary_path, cache_path)
        presence = (np.abs(sequence).sum(axis=1) > 0).astype(np.int32)
        output['feature_status'] = 'ok'
        output['feature_error'] = ''
        output['sampled_frame_indices'] = ';'.join(str(int(index)) for index in indices)
        output['detected_frame_count'] = str(int(presence.sum()))
        output['sequence_nonzero'] = str(int(bool(sequence.any())))
    except Exception as error:
        output['feature_status'] = 'excluded'
        output['feature_error'] = f'{type(error).__name__}: {error}'
        output['sampled_frame_indices'] = ''
        output['detected_frame_count'] = '0'
        output['sequence_nonzero'] = '0'
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output-manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--detection-confidence', type=float, default=0.10)
    parser.add_argument('--presence-confidence', type=float, default=0.10)
    parser.add_argument('--tracking-confidence', type=float, default=0.10)
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not all(0.0 <= value <= 1.0 for value in (args.detection_confidence, args.presence_confidence, args.tracking_confidence)):
        raise ValueError('Los umbrales de MediaPipe deben pertenecer a [0,1]')
    with args.manifest.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('--limit debe ser positivo')
        rows = rows[:args.limit]
    args.cache_root.mkdir(parents=True, exist_ok=True)
    payloads = [(row, str(args.cache_root)) for row in rows]
    with Pool(processes=args.workers, initializer=initialize_worker, initargs=(str(args.model), args.detection_confidence, args.presence_confidence, args.tracking_confidence)) as pool:
        extracted = list(pool.imap_unordered(extract_one, payloads, chunksize=1))
    by_sample = {row['sample_id']: row for row in extracted}
    ordered = [by_sample[row['sample_id']] for row in rows]
    fieldnames = [*rows[0].keys(), 'extractor', 'feature_error', 'sampled_frame_indices', 'detected_frame_count', 'sequence_nonzero']
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)
    counts = {status: sum(row['feature_status'] == status for row in ordered) for status in ('ok', 'excluded')}
    print(json.dumps({'rows': len(ordered), 'counts': counts, 'output_manifest': str(args.output_manifest)}, ensure_ascii=False))


if __name__ == '__main__':
    main()