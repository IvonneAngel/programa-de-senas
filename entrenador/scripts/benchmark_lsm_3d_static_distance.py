"""Baseline exploratorio de forma 3D LSM; no es un modelo móvil ni una evaluación."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FPS_POINTS = 128
RADIAL_QUANTILES = 32
PAIRWISE_QUANTILES = 64


def discrete_accuracy(true: np.ndarray | list[int], predicted: np.ndarray | list[int]) -> float:
    true_array = np.asarray(true, dtype=np.int64)
    predicted_array = np.asarray(predicted, dtype=np.int64)
    if true_array.shape != predicted_array.shape or true_array.size == 0:
        raise ValueError("Etiquetas inválidas para exactitud")
    return float(np.mean(true_array == predicted_array))


def discrete_macro_f1(true: np.ndarray | list[int], predicted: np.ndarray | list[int], labels: np.ndarray) -> float:
    true_array = np.asarray(true, dtype=np.int64)
    predicted_array = np.asarray(predicted, dtype=np.int64)
    if true_array.shape != predicted_array.shape or true_array.size == 0:
        raise ValueError("Etiquetas inválidas para macro-F1")
    scores: list[float] = []
    for label in np.asarray(labels, dtype=np.int64):
        true_positive = int(np.sum((true_array == label) & (predicted_array == label)))
        false_positive = int(np.sum((true_array != label) & (predicted_array == label)))
        false_negative = int(np.sum((true_array == label) & (predicted_array != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else (2.0 * true_positive) / denominator)
    return float(np.mean(scores))


def farthest_point_indices(points: np.ndarray, count: int = FPS_POINTS) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < count:
        raise ValueError(f"Se requieren al menos {count} puntos XYZ")
    first = int(np.lexsort((points[:, 2], points[:, 1], points[:, 0]))[0])
    selected = np.empty(count, dtype=np.int64)
    selected[0] = first
    minimum_squared_distance = np.sum((points - points[first]) ** 2, axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(minimum_squared_distance))
        candidate_distance = np.sum((points - points[selected[index]]) ** 2, axis=1)
        minimum_squared_distance = np.minimum(minimum_squared_distance, candidate_distance)
    return selected


def shape_descriptor(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Nube 3D inválida")
    radii = np.linalg.norm(values, axis=1)
    sampled = values[farthest_point_indices(values)]
    pairwise = np.linalg.norm(sampled[:, None, :] - sampled[None, :, :], axis=2)
    pairwise = pairwise[np.triu_indices(sampled.shape[0], k=1)]
    covariance_eigenvalues = np.linalg.eigvalsh(np.cov(values, rowvar=False))
    covariance_eigenvalues = covariance_eigenvalues / covariance_eigenvalues.sum().clip(min=1e-12)
    descriptor = np.concatenate((
        np.quantile(radii, np.linspace(0.0, 1.0, RADIAL_QUANTILES)),
        np.quantile(pairwise, np.linspace(0.0, 1.0, PAIRWISE_QUANTILES)),
        covariance_eigenvalues,
    )).astype(np.float64)
    if descriptor.shape != (RADIAL_QUANTILES + PAIRWISE_QUANTILES + 3,) or not np.isfinite(descriptor).all():
        raise AssertionError("Descriptor 3D inválido")
    return descriptor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = [row for row in raw_rows if row["is_quarantined_duplicate_copy"] == "False"]
    if len(raw_rows) != 315 or len(rows) != 311:
        raise ValueError(f"Cobertura inesperada: bruto={len(raw_rows)}, elegible={len(rows)}")
    class_names = sorted({row["label_lsm_static"] for row in rows})
    if len(class_names) != 21:
        raise ValueError("Se requieren 21 letras estáticas")
    label_to_index = {label: index for index, label in enumerate(class_names)}
    features = np.stack([shape_descriptor(np.load(args.cache_root / row["canonical_relative_path"], allow_pickle=False)) for row in rows])
    target = np.asarray([label_to_index[row["label_lsm_static"]] for row in rows], dtype=np.int64)
    indices = np.asarray([int(row["performer_index_unverified"]) for row in rows])
    labels = np.arange(len(class_names), dtype=np.int64)
    oof_true: list[int] = []
    oof_predicted: list[int] = []
    prediction_rows: list[dict[str, str | int]] = []
    fold_reports: list[dict[str, float | int]] = []
    for held_out_index in range(1, 16):
        test_mask = indices == held_out_index
        train_mask = ~test_mask
        train_features, test_features = features[train_mask], features[test_mask]
        mean = train_features.mean(axis=0, keepdims=True)
        scale = train_features.std(axis=0, keepdims=True).clip(min=1e-8)
        standardized_train = (train_features - mean) / scale
        standardized_test = (test_features - mean) / scale
        centroids = {int(label): standardized_train[target[train_mask] == label].mean(axis=0) for label in labels}
        predicted = np.asarray([min(labels, key=lambda label: float(np.sum((vector - centroids[int(label)]) ** 2))) for vector in standardized_test], dtype=np.int64)
        present_labels = np.unique(target[test_mask])
        fold_reports.append({
            "held_out_index_unverified": held_out_index,
            "samples": int(test_mask.sum()),
            "classes_present": len(present_labels),
            "accuracy": discrete_accuracy(target[test_mask], predicted),
            "macro_f1_present_classes": discrete_macro_f1(target[test_mask], predicted, present_labels),
        })
        for source_row, truth, prediction in zip(np.asarray(rows, dtype=object)[test_mask], target[test_mask], predicted):
            prediction_rows.append({"sample_id": source_row["sample_id"], "held_out_index_unverified": held_out_index, "label_lsm_static": class_names[int(truth)], "prediction": class_names[int(prediction)]})
        oof_true.extend(target[test_mask].tolist())
        oof_predicted.extend(predicted.tolist())
    report = {
        "descriptor_dim": RADIAL_QUANTILES + PAIRWISE_QUANTILES + 3,
        "descriptor": {"fps_points": FPS_POINTS, "radial_quantiles": RADIAL_QUANTILES, "pairwise_quantiles": PAIRWISE_QUANTILES, "covariance_eigenvalues": 3},
        "eligible_samples": len(rows),
        "quarantined_duplicate_copies": len(raw_rows) - len(rows),
        "protocol": "leave-one-index-out; index identity is unverified and is not a signer split",
        "accuracy": discrete_accuracy(oof_true, oof_predicted),
        "macro_f1": discrete_macro_f1(oof_true, oof_predicted, labels),
        "folds": fold_reports,
        "trained_neural_model": False,
        "mobile_or_word_recognition_claim": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(json.dumps(report))


if __name__ == "__main__":
    main()