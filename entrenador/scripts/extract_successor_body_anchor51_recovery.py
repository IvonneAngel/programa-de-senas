"""Extrae `(30,51)` body_anchor desde el corpus sucesor recuperado."""
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

HAND_POINTS = (0, 4, 8, 12, 16, 20)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
_HAND: Any = None
_POSE: Any = None


def initialize_worker(hand_model: str, pose_model: str) -> None:
    global _HAND, _POSE
    _HAND = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model), running_mode=vision.RunningMode.IMAGE,
        num_hands=2, min_hand_detection_confidence=0.30, min_hand_presence_confidence=0.30, min_tracking_confidence=0.50,
    ))
    _POSE = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=pose_model), running_mode=vision.RunningMode.IMAGE,
        num_poses=1, min_pose_detection_confidence=0.30, min_pose_presence_confidence=0.30, min_tracking_confidence=0.50,
    ))


def natural_frames(source_dir: Path) -> list[Path]:
    return sorted((path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() == '.jpg'), key=lambda path: int(''.join(character for character in path.stem if character.isdigit()) or '0'))


def sampled_indices(count: int) -> np.ndarray:
    if count < 1:
        raise ValueError('El clip no contiene frames')
    return np.rint(np.linspace(0, count - 1, num=30)).astype(np.int64)


def frame_feature(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f'No se pudo decodificar {path}')
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    hand_result = _HAND.detect(mp_image)
    pose_result = _POSE.detect(mp_image)
    output = np.zeros(51, dtype=np.float32)
    if not pose_result.pose_landmarks:
        return output
    pose_landmarks = pose_result.pose_landmarks[0]
    if len(pose_landmarks) != 33:
        return output
    left = pose_landmarks[LEFT_SHOULDER]
    right = pose_landmarks[RIGHT_SHOULDER]
    if min(float(left.visibility), float(right.visibility)) < 0.50:
        return output
    center_x, center_y = (float(left.x) + float(right.x)) * 0.5, (float(left.y) + float(right.y)) * 0.5
    scale = float(np.hypot(float(left.x) - float(right.x), float(left.y) - float(right.y)))
    if not np.isfinite(scale) or scale <= 1e-6:
        return output
    output[50] = 1.0
    hands: dict[str, tuple[float, np.ndarray]] = {}
    for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        if len(landmarks) != 21 or not handedness:
            continue
        category = handedness[0]
        side = str(category.category_name).lower().strip()
        if side not in {'left', 'right'}:
            continue
        values = np.asarray([[landmark.x, landmark.y, landmark.z] for landmark in landmarks], dtype=np.float32)
        if not np.isfinite(values).all():
            continue
        if side not in hands or float(category.score) > hands[side][0]:
            hands[side] = (float(category.score), values)
    for slot, side in enumerate(('right', 'left')):
        if side not in hands:
            continue
        points = hands[side][1]
        start = slot * 12
        for offset, point_index in enumerate(HAND_POINTS):
            output[start + offset * 2] = (points[point_index, 0] - center_x) / scale
            output[start + offset * 2 + 1] = (points[point_index, 1] - center_y) / scale
        output[48 + slot] = 1.0
    return output


def body_motion(positioned: np.ndarray) -> np.ndarray:
    if positioned.shape != (30, 51) or not np.isfinite(positioned).all():
        raise ValueError(f'Posiciones body_anchor inválidas: {positioned.shape}')
    output = np.zeros_like(positioned)
    output[:, :24] = positioned[:, :24]
    output[:, 48:] = positioned[:, 48:]
    output[0, 24:48] = positioned[1, :24] - positioned[0, :24]
    output[-1, 24:48] = positioned[-1, :24] - positioned[-2, :24]
    output[1:-1, 24:48] = (positioned[2:, :24] - positioned[:-2, :24]) * 0.5
    output[output[:, 48] < 0.5, 24:36] = 0.0
    output[output[:, 49] < 0.5, 36:48] = 0.0
    return output


def extract_one(payload: tuple[dict[str, str], str]) -> dict[str, str]:
    row, cache_root_text = payload
    output = dict(row)
    output['extractor'] = 'mediapipe_hand_pose_body_anchor51_threshold_0.30_0.50_0.50'
    output['feature_path'] = f"body_anchor51/{row['sample_id']}.npy"
    try:
        frames = natural_frames(Path(row['source_dir']))
        indices = sampled_indices(len(frames))
        positioned = np.stack([frame_feature(frames[int(index)]) for index in indices])
        sequence = body_motion(positioned)
        body_eligible = int((sequence[:, 50] > 0.5).sum())
        if body_eligible == 0:
            raise ValueError('No hubo frames cuerpo+mano elegibles')
        cache_path = Path(cache_root_text) / output['feature_path']
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix('.npy.tmp')
        with temporary.open('wb') as handle:
            np.save(handle, sequence.astype(np.float32), allow_pickle=False)
        os.replace(temporary, cache_path)
        output.update(feature_status='ok', feature_error='', sampled_frame_indices=';'.join(map(str, indices.tolist())), detected_frame_count=str(body_eligible), sequence_nonzero=str(int(bool(sequence.any()))))
    except Exception as error:
        output.update(feature_status='excluded', feature_error=f'{type(error).__name__}: {error}', sampled_frame_indices='', detected_frame_count='0', sequence_nonzero='0')
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output-manifest', type=Path, required=True)
    parser.add_argument('--hand-model', type=Path, required=True)
    parser.add_argument('--pose-model', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if not args.hand_model.is_file() or not args.pose_model.is_file():
        raise FileNotFoundError('No se localizó Hand/Pose Landmarker')
    with args.manifest.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('Manifiesto vacío')
    args.cache_root.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers, initializer=initialize_worker, initargs=(str(args.hand_model), str(args.pose_model))) as pool:
        extracted = list(pool.imap_unordered(extract_one, [(row, str(args.cache_root)) for row in rows], chunksize=1))
    indexed = {row['sample_id']: row for row in extracted}
    ordered = [indexed[row['sample_id']] for row in rows]
    fieldnames = [*rows[0].keys()]
    for name in ('extractor', 'feature_error', 'sampled_frame_indices', 'detected_frame_count', 'sequence_nonzero'):
        if name not in fieldnames:
            fieldnames.append(name)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)
    print(json.dumps({'rows': len(ordered), 'ok': sum(row['feature_status'] == 'ok' for row in ordered), 'excluded': sum(row['feature_status'] != 'ok' for row in ordered)}, ensure_ascii=False))


if __name__ == '__main__':
    main()