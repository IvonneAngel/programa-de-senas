"""Diagnóstico descriptivo train–S08 de la recuperación positions126; nunca lee S09."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summary_feature(sequence: np.ndarray) -> np.ndarray:
    if sequence.shape != (30, 126) or not np.isfinite(sequence).all():
        raise ValueError(f"Tensor inválido: {sequence.shape}")
    return np.concatenate((sequence.mean(axis=0), sequence.std(axis=0), sequence[-1] - sequence[0])).astype(np.float32)


def movement_energy(sequence: np.ndarray) -> float:
    return float(np.abs(np.diff(sequence, axis=0)).mean())


def load_rows(manifest: Path, cache_root: Path) -> list[dict[str, object]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    selected = []
    for row in raw_rows:
        split = row["split_project"]
        if split == "test":
            continue
        if split not in {"train", "validation"}:
            raise ValueError(f"Split inesperado: {split}")
        if row.get("feature_status") != "ok":
            continue
        path = cache_root / row["feature_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        sequence = np.load(path, allow_pickle=False)
        selected.append({
            "sample_id": row["sample_id"],
            "label": row["label_lsm"],
            "class_number": int(row["class_number"]),
            "signer": row["signer_id"],
            "split": split,
            "feature": summary_feature(sequence),
            "movement": movement_energy(sequence),
            "nonzero_fraction": float(np.count_nonzero(sequence) / sequence.size),
        })
    if not selected:
        raise ValueError("No hay filas train/S08 válidas")
    return selected


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.manifest, args.cache_root)
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if len(train) != 1470 or len(validation) != 210:
        raise ValueError(f"Cobertura inesperada: train={len(train)}, S08={len(validation)}")
    labels = sorted({str(row["label"]) for row in train})
    if len(labels) != 210 or {str(row["label"]) for row in validation} != set(labels):
        raise ValueError("Las 210 clases deben estar completas en train y S08")
    train_x = np.stack([row["feature"] for row in train])
    location = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    by_label: dict[str, list[dict[str, object]]] = {label: [] for label in labels}
    for row in train:
        by_label[str(row["label"])].append(row)
    centroids = np.stack([np.stack([row["feature"] for row in by_label[label]]).mean(axis=0) for label in labels])
    centroids = (centroids - location) / scale
    class_rows: list[dict[str, object]] = []
    for row in validation:
        feature = (np.asarray(row["feature"]) - location) / scale
        distances = np.linalg.norm(centroids - feature, axis=1)
        correct_index = labels.index(str(row["label"]))
        nearest = int(np.argmin(distances))
        second = float(np.partition(distances, 1)[1])
        own = float(distances[correct_index])
        train_class = by_label[str(row["label"])]
        class_features = np.stack([(np.asarray(item["feature"]) - location) / scale for item in train_class])
        compactness = float(np.linalg.norm(class_features - centroids[correct_index], axis=1).mean())
        class_rows.append({
            "sample_id": row["sample_id"],
            "class_number": row["class_number"],
            "label_lsm": row["label"],
            "s08_signer": row["signer"],
            "nearest_centroid_correct": nearest == correct_index,
            "nearest_label": labels[nearest],
            "own_centroid_distance": own,
            "nearest_other_distance": second if nearest == correct_index else float(distances[nearest]),
            "margin_to_impostor": (second - own) if nearest == correct_index else (float(distances[nearest]) - own),
            "train_class_compactness": compactness,
            "shift_over_compactness": own / max(compactness, 1e-6),
            "movement_energy": row["movement"],
            "nonzero_fraction": row["nonzero_fraction"],
        })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "positions126_train_s08_separability_by_class.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    own = np.asarray([float(row["own_centroid_distance"]) for row in class_rows])
    margin = np.asarray([float(row["margin_to_impostor"]) for row in class_rows])
    movement = np.asarray([float(row["movement_energy"]) for row in class_rows])
    nonzero = np.asarray([float(row["nonzero_fraction"]) for row in class_rows])
    report = {
        "data_scope": {"splits_read": ["train", "validation"], "s09_read": False, "train_samples": len(train), "s08_samples": len(validation), "classes": len(labels)},
        "representation": "mean_std_endpoint_displacement_of_positions126",
        "nearest_centroid_diagnostic_accuracy": float(np.mean([bool(row["nearest_centroid_correct"]) for row in class_rows])),
        "own_centroid_distance": {"mean": float(own.mean()), "median": float(np.median(own)), "p90": float(np.quantile(own, 0.90))},
        "margin_to_impostor": {"mean": float(margin.mean()), "median": float(np.median(margin)), "negative_count": int(np.sum(margin <= 0))},
        "correlations": {"movement_vs_own_distance": correlation(movement, own), "movement_vs_margin": correlation(movement, margin), "nonzero_vs_own_distance": correlation(nonzero, own), "nonzero_vs_margin": correlation(nonzero, margin)},
        "hardest_classes_by_shift": sorted(class_rows, key=lambda row: float(row["shift_over_compactness"]), reverse=True)[:20],
        "lowest_margin_classes": sorted(class_rows, key=lambda row: float(row["margin_to_impostor"]))[:20],
    }
    (args.out_dir / "positions126_train_s08_separability_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"nearest_centroid_diagnostic_accuracy": report["nearest_centroid_diagnostic_accuracy"], "negative_margin_count": report["margin_to_impostor"]["negative_count"], "s09_read": False}))


if __name__ == "__main__":
    main()