"""Audita conflictos de CrossEntropy por firmante sin actualización de parámetros."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lsm.models.tcn import TemporalTCN


def load_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [row for row in rows if row["task"] == "successor_positions126" and row["feature_status"] == "ok" and row["split_model"] == "train"]
    expected = {f"S{index:02d}" for index in range(1, 8)}
    if len(train) != 1470 or {row["signer_id"] for row in train} != expected:
        raise ValueError("La auditoría exige exactamente S01–S07 y 1,470 filas train")
    if any(row["signer_id"] in {"S08", "S09"} for row in train):
        raise AssertionError("Fuga de firmante")
    return train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.manifest)
    labels = {label: index for index, label in enumerate(sorted({row["label_lsm"] for row in rows}))}
    model = TemporalTCN(feature_dim=126, classes=210, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint sin model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    criterion = nn.CrossEntropyLoss()
    gradients: dict[str, torch.Tensor] = {}
    losses: dict[str, float] = {}
    for signer in sorted({row["signer_id"] for row in rows}):
        group = [row for row in rows if row["signer_id"] == signer]
        if len(group) != 210 or len({row["label_lsm"] for row in group}) != 210:
            raise ValueError(f"{signer}: soporte de clases inválido")
        features = torch.stack([torch.from_numpy(np.load(args.cache_root / row["feature_path"], allow_pickle=False).astype(np.float32, copy=False)) for row in group])
        targets = torch.tensor([labels[row["label_lsm"]] for row in group], dtype=torch.long)
        if features.shape != (210, 30, 126) or not torch.isfinite(features).all():
            raise ValueError(f"{signer}: features inválidos")
        loss = criterion(model(features), targets)
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        grad = torch.autograd.grad(loss, params, retain_graph=False)
        flat = torch.cat([part.detach().reshape(-1) for part in grad])
        if not torch.isfinite(flat).all():
            raise AssertionError(f"{signer}: gradiente no finito")
        gradients[signer] = flat
        losses[signer] = float(loss.detach())
    pairs = []
    for left, right in combinations(sorted(gradients), 2):
        cosine = torch.nn.functional.cosine_similarity(gradients[left], gradients[right], dim=0, eps=1e-12)
        pairs.append({"left": left, "right": right, "cosine": float(cosine)})
    cosines = np.asarray([pair["cosine"] for pair in pairs], dtype=np.float64)
    report = {
        "kind": "successor_signer_gradient_conflict_train_only_audit",
        "split_read": "train_only_S01_to_S07",
        "s08_read": False,
        "s09_read": False,
        "checkpoint": str(args.checkpoint),
        "labels": len(labels),
        "signers": sorted(gradients),
        "loss_by_signer": losses,
        "pairs": pairs,
        "cosine_summary": {"min": float(cosines.min()), "median": float(np.median(cosines)), "max": float(cosines.max()), "negative_pairs": int((cosines < 0.0).sum()), "negative_fraction": float((cosines < 0.0).mean())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()