"""Valida una restauración del benchmark sucesor sin entrenar ni evaluar modelos."""
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
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-limit', type=int, default=32)
    args = parser.parse_args()
    with args.manifest.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    required = {'sample_id', 'label_lsm', 'signer_id', 'split_model', 'feature_status', 'feature_path'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Manifiesto sin columnas requeridas: {required}')
    accepted = [row for row in rows if row.get('feature_status', 'ok') == 'ok']
    rejected = [row for row in rows if row not in accepted]
    split_counts = Counter(row['split_model'] for row in accepted)
    signer_counts = Counter(row['signer_id'] for row in accepted)
    labels = {row['label_lsm'] for row in accepted}
    admissible_layout = (
        len(accepted) == 1857
        and split_counts == Counter({'train': 1437, 'validation': 210, 'test': 210})
        and len(labels) == 210
        and (len(rows) == 1857 or (len(rows) == 1890 and len(rejected) == 33 and all(row['split_model'] == 'train' for row in rejected)))
    )
    checked = []
    for row in accepted[:args.sample_limit]:
        path = args.cache_root / row['feature_path']
        array = np.load(path, allow_pickle=False)
        checked.append({'sample_id': row['sample_id'], 'shape': list(array.shape), 'finite': bool(np.isfinite(array).all())})
    tensor_contract_ok = all(item['shape'] == [30, 126] and item['finite'] for item in checked)
    payload = {
        'manifest': str(args.manifest),
        'cache_root': str(args.cache_root),
        'rows_total': len(rows),
        'accepted_rows': len(accepted),
        'rejected_rows': len(rejected),
        'accepted_split_counts': dict(sorted(split_counts.items())),
        'accepted_signer_counts': dict(sorted(signer_counts.items())),
        'label_count': len(labels),
        'layout_admissible': admissible_layout,
        'sampled_tensors': checked,
        'tensor_contract_ok': tensor_contract_ok,
        'admitted_for_control_revalidation': bool(admissible_layout and tensor_contract_ok),
        's09_evaluated': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: payload[key] for key in ('accepted_rows', 'rejected_rows', 'accepted_split_counts', 'label_count', 'layout_admissible', 'tensor_contract_ok', 'admitted_for_control_revalidation')}, ensure_ascii=False))


if __name__ == '__main__':
    main()