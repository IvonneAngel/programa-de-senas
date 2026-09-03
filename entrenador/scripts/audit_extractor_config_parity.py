"""Compara configuraciones MediaPipe solo sobre filas train del benchmark recuperado."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


CONFIGS = {
    'recovery_010': {'detection': 0.10, 'presence': 0.10, 'tracking': 0.10},
    'legacy_030': {'detection': 0.30, 'presence': 0.50, 'tracking': 0.50},
}


def frames_for(source_dir: Path) -> list[Path]:
    frames = [path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() == '.jpg']
    return sorted(frames, key=lambda path: int(''.join(char for char in path.stem if char.isdigit()) or '0'))


def sampled_indices(frame_count: int) -> np.ndarray:
    return np.rint(np.linspace(0, frame_count - 1, 30)).astype(np.int64)


def detected(detector: vision.HandLandmarker, frame_path: Path) -> bool:
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        return False
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    return bool(result.hand_landmarks)


def audit(rows: list[dict[str, str]], model: Path, name: str, settings: dict[str, float]) -> dict[str, object]:
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=settings['detection'],
        min_hand_presence_confidence=settings['presence'],
        min_tracking_confidence=settings['tracking'],
    )
    distribution: Counter[int] = Counter()
    per_signer: dict[str, list[int]] = {}
    with vision.HandLandmarker.create_from_options(options) as detector:
        for index, row in enumerate(rows, start=1):
            frames = frames_for(Path(row['source_dir']))
            count = sum(detected(detector, frames[int(frame_index)]) for frame_index in sampled_indices(len(frames)))
            distribution[count] += 1
            per_signer.setdefault(row['signer_id'], []).append(count)
            if index % 100 == 0:
                print(json.dumps({'configuration': name, 'clips': index}), flush=True)
    return {
        'settings': settings,
        'sample_count': len(rows),
        'fully_null_count': distribution[0],
        'detected_frame_count_distribution': {str(key): distribution[key] for key in sorted(distribution)},
        'per_signer_mean_presence': {signer: float(np.mean(values) / 30.0) for signer, values in sorted(per_signer.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding='utf-8', newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if row['split_model'] == 'train']
    if len(rows) != 1470 or any(row['signer_id'] in {'S08', 'S09'} for row in rows):
        raise ValueError('La auditoría de paridad solo admite las 1,470 filas train S01–S07')
    result = {'scope': 'train_only', 'historical_reference_fully_null': 16, 'configurations': {}}
    for name, settings in CONFIGS.items():
        result['configurations'][name] = audit(rows, args.model, name, settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()