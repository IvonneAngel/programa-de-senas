"""Auditoría OOF de pesos convexos para bone/cov/code, sin S08/S09."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def macro_f1(targets: np.ndarray, predictions: np.ndarray, classes: int = 210) -> float:
    values = []
    for label in range(classes):
        tp = int(np.sum((targets == label) & (predictions == label)))
        fp = int(np.sum((targets != label) & (predictions == label)))
        fn = int(np.sum((targets == label) & (predictions != label)))
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(values))


def fit_weights(logits: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, float]:
    torch.use_deterministic_algorithms(True, warn_only=True)
    data = torch.from_numpy(logits.astype(np.float64, copy=False))
    truth = torch.from_numpy(targets.astype(np.int64, copy=False))
    raw = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], lr=0.5, max_iter=100, tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.CrossEntropyLoss()
    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(torch.einsum("v,nvc->nc", torch.softmax(raw, dim=0), data), truth)
        loss.backward()
        return loss
    loss = float(optimizer.step(closure).detach())
    return torch.softmax(raw.detach(), dim=0).numpy(), loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.oof, allow_pickle=False)
    signers = data["signers"].astype(str)
    targets = data["targets"].astype(np.int64)
    logits = np.stack([data["bone_logits"], data["cov_logits"], data["code_logits"]], axis=1).astype(np.float64)
    if logits.shape != (1470, 3, 210) or targets.shape != (1470,) or set(signers) != {f"S{i:02d}" for i in range(1, 8)}:
        raise ValueError("OOF incompatible")
    uniform = logits.mean(axis=1)
    weights, loss = fit_weights(logits, targets)
    stacked = np.einsum("v,nvc->nc", weights, logits)
    per_signer = []
    for signer in sorted(set(signers)):
        train_mask, test_mask = signers != signer, signers == signer
        local_weights, local_loss = fit_weights(logits[train_mask], targets[train_mask])
        local_stacked = np.einsum("v,nvc->nc", local_weights, logits[test_mask])
        per_signer.append({"holdout": signer, "weights": local_weights.tolist(), "train_cross_entropy": local_loss, "uniform_macro_f1": macro_f1(targets[test_mask], uniform[test_mask].argmax(axis=1)), "stacked_macro_f1": macro_f1(targets[test_mask], local_stacked.argmax(axis=1))})
    holdout_weights = np.asarray([item["weights"] for item in per_signer], dtype=np.float64)
    report = {"protocol": "OOF S01-S07; convex weights softmax; CrossEntropy; no S08/S09", "s08_read": False, "s09_read": False, "weights_full_oof": weights.tolist(), "cross_entropy_full_oof": loss, "uniform_macro_f1_oof": macro_f1(targets, uniform.argmax(axis=1)), "stacked_macro_f1_oof": macro_f1(targets, stacked.argmax(axis=1)), "leave_one_signer_out": per_signer, "weight_range_by_view": {name: [float(holdout_weights[:, index].min()), float(holdout_weights[:, index].max())] for index, name in enumerate(("bone", "cov", "code"))}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()