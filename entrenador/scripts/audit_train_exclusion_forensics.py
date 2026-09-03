"""Audita hipótesis de exclusión del benchmark sucesor solo con S01–S07."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def raw_frame_count(source_dir: Path) -> int:
    return sum(path.is_file() and path.suffix.lower() == '.jpg' for path in source_dir.iterdir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(encoding='utf-8', newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if row['split_model'] == 'train' and row['feature_status'] == 'ok']
    if len(rows) != 1470 or any(row['signer_id'] in {'S08', 'S09'} for row in rows):
        raise ValueError('La auditoría requiere exactamente S01–S07: 1,470 filas train')

    records: list[dict[str, object]] = []
    by_signer: dict[str, list[int]] = defaultdict(list)
    raw_counts: Counter[int] = Counter()
    for row in rows:
        array = np.load(args.cache_root / row['feature_path'], allow_pickle=False)
        if array.shape != (30, 126):
            raise ValueError(f"{row['sample_id']}: forma inválida {array.shape}")
        detected = int((np.abs(array).sum(axis=1) > 0).sum())
        frames = raw_frame_count(Path(row['source_dir']))
        records.append({
            'sample_id': row['sample_id'],
            'signer_id': row['signer_id'],
            'label_lsm': row['label_lsm'],
            'detected_frames': detected,
            'raw_frame_count': frames,
        })
        by_signer[row['signer_id']].append(detected)
        raw_counts[frames] += 1

    thresholds = {str(threshold): sum(record['detected_frames'] <= threshold for record in records) for threshold in range(31)}
    payload = {
        'scope': 'train_only_s01_s07',
        'historical_excluded_count': 33,
        'historical_fully_null_count': 16,
        'sample_count': len(records),
        'exclusion_counts_if_max_detected_frames': thresholds,
        'raw_frame_count_distribution': {str(key): raw_counts[key] for key in sorted(raw_counts)},
        'per_signer_detected_frame_summary': {
            signer: {
                'count': len(values),
                'min': int(min(values)),
                'median': float(np.median(values)),
                'mean': float(np.mean(values)),
                'max': int(max(values)),
            }
            for signer, values in sorted(by_signer.items())
        },
        'lowest_presence_records': sorted(records, key=lambda record: (record['detected_frames'], record['raw_frame_count'], record['sample_id']))[:80],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'scope': payload['scope'],
        'thresholds': {key: value for key, value in thresholds.items() if value in {16, 33}},
        'raw_frame_count_distribution': payload['raw_frame_count_distribution'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()