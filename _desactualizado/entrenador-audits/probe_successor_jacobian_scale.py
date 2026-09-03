"""Mide la escala train-only de una proyección Jacobiana en el control recuperado."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lsm.models.tcn import TemporalTCN, parameter_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with args.manifest.open(encoding='utf-8', newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if row['split_model'] == 'train' and row['feature_status'] == 'ok']
    labels = {label: index for index, label in enumerate(sorted({row['label_lsm'] for row in rows}))}
    batch_rows = rows[:args.batch_size]
    values = np.stack([np.load(args.cache_root / row['feature_path'], allow_pickle=False) for row in batch_rows]).astype(np.float32)
    targets = torch.tensor([labels[row['label_lsm']] for row in batch_rows], dtype=torch.long)
    features = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    model = TemporalTCN(feature_dim=126, classes=210, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20)
    if parameter_count(model) != 158_994:
        raise AssertionError(parameter_count(model))
    logits = model(features)
    supervised = nn.CrossEntropyLoss()(logits, targets)
    generator = torch.Generator().manual_seed(args.seed + 1)
    projection = torch.empty_like(logits).bernoulli_(0.5, generator=generator).mul_(2.0).sub_(1.0) / np.sqrt(logits.shape[1])
    scalar = (logits * projection).sum()
    input_gradient = torch.autograd.grad(scalar, features, create_graph=False)[0]
    penalty = input_gradient.square().mean()
    payload = {
        'split': 'train',
        'seed': args.seed,
        'batch_size': args.batch_size,
        'classes': logits.shape[1],
        'cross_entropy': float(supervised.detach()),
        'jacobian_projection_penalty': float(penalty.detach()),
        'lambda_for_one_percent_ce_scale': float(0.01 * supervised.detach() / penalty.detach()),
        'parameter_count': parameter_count(model),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == '__main__':
    main()