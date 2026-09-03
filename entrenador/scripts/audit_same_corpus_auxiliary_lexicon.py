"""Audita clases Mendeley fuera del benchmark 210 usando solo firmantes S01–S07."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark-manifest', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.benchmark_manifest.open(encoding='utf-8', newline='') as handle:
        benchmark_classes = {row['class_number'] for row in csv.DictReader(handle)}
    auxiliary: list[dict[str, object]] = []
    for class_dir in sorted(path for path in args.raw_root.iterdir() if path.is_dir() and path.name.isdigit() and len(path.name) == 3):
        if class_dir.name in benchmark_classes:
            continue
        train_dirs = []
        for signer in range(1, 8):
            canonical = class_dir / f'{signer:02d}{class_dir.name}'
            if canonical.is_dir():
                frame_count = sum(1 for frame in canonical.iterdir() if frame.suffix.lower() in {'.jpg', '.jpeg', '.png'})
                if frame_count:
                    train_dirs.append({'signer_id': f'S{signer:02d}', 'source_dir': str(canonical), 'frame_count': frame_count})
        auxiliary.append({'class_number': class_dir.name, 'train_clip_count': len(train_dirs), 'train_clips': train_dirs})
    payload = {
        'scope': 'same_corpus_auxiliary_lexicon_train_only',
        'allowed_signers': [f'S{signer:02d}' for signer in range(1, 8)],
        'forbidden_signers': ['S08', 'S09'],
        'benchmark_class_count': len(benchmark_classes),
        'auxiliary_class_count': len(auxiliary),
        'auxiliary_train_clip_count': sum(int(item['train_clip_count']) for item in auxiliary),
        'auxiliary_classes_with_at_least_four_train_clips': sum(int(item['train_clip_count']) >= 4 for item in auxiliary),
        'auxiliary': auxiliary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in payload.items() if key != 'auxiliary'}, ensure_ascii=False))


if __name__ == '__main__':
    main()