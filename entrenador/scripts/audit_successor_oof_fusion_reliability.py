"""Auditoría train-only de fiabilidad para vistas de fusión de palabras aisladas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def macro_f1(targets: np.ndarray, predictions: np.ndarray, classes: int = 210) -> float:
    scores = []
    for label in range(classes):
        tp = int(np.sum((targets == label) & (predictions == label)))
        fp = int(np.sum((targets != label) & (predictions == label)))
        fn = int(np.sum((targets == label) & (predictions != label)))
        scores.append(0.0 if (denominator := 2 * tp + fp + fn) == 0 else 2.0 * tp / denominator)
    return float(np.mean(scores))


def metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    probs = softmax(logits)
    prediction = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = prediction == targets
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(axis=1)
    return {"macro_f1": macro_f1(targets, prediction), "accuracy": float(correct.mean()), "mean_confidence": float(confidence.mean()), "ece10": float(ece), "mean_entropy": float(entropy.mean())}


def entropy_fusion(views: np.ndarray, temperature: float) -> np.ndarray:
    probabilities = np.asarray([softmax(view) for view in views], dtype=np.float64)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1))).sum(axis=2)
    weights = softmax((-entropy / temperature).T).T
    return (weights[:, :, None] * views).sum(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.oof, allow_pickle=False) as archive:
        targets = np.asarray(archive["targets"], dtype=np.int64)
        signers = np.asarray(archive["signers"]).astype(str)
        views = np.stack([np.asarray(archive[key], dtype=np.float64) for key in ("bone_logits", "cov_logits", "code_logits")])
    if views.shape != (3, 1470, 210) or targets.shape != (1470,) or set(signers) != {f"S{index:02d}" for index in range(1, 8)}:
        raise ValueError("OOF incompatible: se requieren 1,470 clips, 210 clases y S01-S07")
    report: dict[str, object] = {"kind": "train_only_oof_reliability_audit", "s08_read": False, "s09_read": False, "views": {name: metrics(view, targets) for name, view in zip(("bone", "cov", "code"), views)}, "uniform": metrics(views.mean(axis=0), targets)}
    confidence = np.stack([softmax(view).max(axis=1) for view in views])
    selector = views[confidence.argmax(axis=0), np.arange(targets.size)]
    report["confidence_selector_diagnostic"] = metrics(selector, targets)
    temperatures = (0.25, 0.5, 1.0, 2.0, 4.0)
    report["entropy_fusion_grid_diagnostic"] = {str(temperature): metrics(entropy_fusion(views, temperature), targets) for temperature in temperatures}
    per_signer = {}
    for signer in sorted(set(signers)):
        mask = signers == signer
        per_signer[signer] = {"uniform": metrics(views.mean(axis=0)[mask], targets[mask]), "selector": metrics(selector[mask], targets[mask])}
    report["per_signer"] = per_signer
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()