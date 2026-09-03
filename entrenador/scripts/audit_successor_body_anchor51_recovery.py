"""Audita integridad técnica y cobertura del caché body_anchor51 recuperado."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('El manifiesto está vacío')
    total = Counter()
    valid = Counter()
    failures: list[dict[str, str]] = []
    class_total = Counter()
    class_valid = Counter()
    for row in rows:
        split = row.get('split_project') or row.get('split_model') or row.get('split') or row.get('signer_id') or 'unknown'
        label = row.get('label_lsm') or row.get('label') or row.get('class_name') or row.get('gloss') or 'unknown'
        total[split] += 1
        class_total[(split, label)] += 1
        if row.get('feature_status') != 'ok':
            continue
        path = args.cache_root / row['feature_path']
        try:
            tensor = np.load(path, allow_pickle=False)
            if tensor.shape != (30, 51):
                raise ValueError(f'forma={tensor.shape}')
            if tensor.dtype != np.float32:
                raise ValueError(f'dtype={tensor.dtype}')
            if not np.isfinite(tensor).all():
                raise ValueError('valores no finitos')
            if int((tensor[:, 50] > 0.5).sum()) < 1:
                raise ValueError('sin frame corporal elegible')
        except Exception as error:
            failures.append({'sample_id': row.get('sample_id', ''), 'reason': f'{type(error).__name__}: {error}'})
            continue
        valid[split] += 1
        class_valid[(split, label)] += 1
    report = {
        'contract': {'shape': [30, 51], 'dtype': 'float32', 'body_mask_channel': 50},
        'rows': len(rows),
        'manifest_ok': sum(row.get('feature_status') == 'ok' for row in rows),
        'tensor_verified_ok': sum(valid.values()),
        'integrity_failures': failures,
        'by_split': {split: {'total': total[split], 'verified_ok': valid[split], 'excluded_or_invalid': total[split] - valid[split], 'coverage': valid[split] / total[split]} for split in sorted(total)},
        'class_coverage': {split: {'classes_total': len({label for candidate, label in class_total if candidate == split}), 'classes_with_tensor': len({label for candidate, label in class_valid if candidate == split}), 'minimum_valid_examples_per_class': min((count for (candidate, _), count in class_valid.items() if candidate == split), default=0)} for split in sorted(total)},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    if failures:
        raise SystemExit(f'Falló la integridad de {len(failures)} tensores')


if __name__ == '__main__':
    main()