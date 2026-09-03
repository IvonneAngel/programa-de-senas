"""Audita presencia y movimiento de una caché sucesora sin entrenar modelos."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        'count': int(array.size),
        'min': float(array.min()),
        'median': float(np.median(array)),
        'mean': float(array.mean()),
        'max': float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', default='train')
    args = parser.parse_args()
    with args.manifest.open(encoding='utf-8', newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if row['split_model'] == args.split and row['feature_status'] == 'ok']
    presence_values: list[float] = []
    movement_values: list[float] = []
    null_samples: list[str] = []
    detected_frames = Counter()
    signer_presence: dict[str, list[float]] = defaultdict(list)
    signer_motion: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        array = np.load(args.cache_root / row['feature_path'], allow_pickle=False)
        if array.shape != (30, 126) or not np.isfinite(array).all():
            raise ValueError(f"{row['sample_id']}: caché inválida {array.shape}")
        present = np.abs(array).sum(axis=1) > 0
        presence = float(present.mean())
        displacement = np.diff(array, axis=0)
        movement = float(np.linalg.norm(displacement, axis=1).mean())
        presence_values.append(presence)
        movement_values.append(movement)
        signer_presence[row['signer_id']].append(presence)
        signer_motion[row['signer_id']].append(movement)
        detected_frames[str(int(present.sum()))] += 1
        if not present.any():
            null_samples.append(row['sample_id'])
    payload = {
        'split': args.split,
        'sample_count': len(rows),
        'presence': summary(presence_values),
        'movement': summary(movement_values),
        'fully_null_sample_count': len(null_samples),
        'fully_null_samples': null_samples,
        'detected_frame_count_distribution': dict(sorted(detected_frames.items(), key=lambda item: int(item[0]))),
        'per_signer': {
            signer: {'presence': summary(signer_presence[signer]), 'movement': summary(signer_motion[signer])}
            for signer in sorted(signer_presence)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: payload[key] for key in ('split', 'sample_count', 'presence', 'movement', 'fully_null_sample_count')}, ensure_ascii=False))


if __name__ == '__main__':
    main()