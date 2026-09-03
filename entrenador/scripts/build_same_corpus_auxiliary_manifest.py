"""Construye el manifiesto auxiliar S01–S07, disjunto de las 210 clases objetivo."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    'sample_id', 'task', 'label_lsm', 'class_number', 'signer_id', 'split_model',
    'split_project', 'source_dir', 'frame_count', 'frames_original', 'feature_status', 'feature_path',
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark-manifest', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.benchmark_manifest.open(encoding='utf-8', newline='') as handle:
        target_classes = {row['class_number'] for row in csv.DictReader(handle)}
    rows: list[dict[str, str]] = []
    for class_dir in sorted(path for path in args.raw_root.iterdir() if path.is_dir() and path.name.isdigit() and len(path.name) == 3):
        if class_dir.name in target_classes:
            continue
        for signer in range(1, 8):
            source_dir = class_dir / f'{signer:02d}{class_dir.name}'
            if not source_dir.is_dir():
                continue
            frame_count = sum(1 for frame in source_dir.iterdir() if frame.suffix.lower() == '.jpg')
            if not frame_count:
                continue
            rows.append({
                'sample_id': f'aux_mendeley_c{class_dir.name}_s{signer:02d}',
                'task': 'successor_same_corpus_auxiliary_lexicon',
                'label_lsm': f'auxiliary_{class_dir.name}',
                'class_number': class_dir.name,
                'signer_id': f'S{signer:02d}',
                'split_model': 'auxiliary_train',
                'split_project': 'auxiliary_train',
                'source_dir': str(source_dir),
                'frame_count': str(frame_count),
                'frames_original': str(frame_count),
                'feature_status': 'pending',
                'feature_path': f'aux_mendeley_c{class_dir.name}_s{signer:02d}.npy',
            })
    classes = {row['class_number'] for row in rows}
    if len(classes) != 39 or len(rows) != 236:
        raise RuntimeError(f'Conteo auxiliar inesperado: {len(classes)} clases, {len(rows)} clips')
    if any(row['signer_id'] in {'S08', 'S09'} for row in rows):
        raise RuntimeError('El manifiesto auxiliar no puede contener S08/S09')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print({'rows': len(rows), 'classes': len(classes), 'allowed_signers': sorted({row['signer_id'] for row in rows})})


if __name__ == '__main__':
    main()