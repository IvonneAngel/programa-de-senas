"""Construye un manifiesto auditable del corpus Mejía de 30 señas LSM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


FILENAME = re.compile(r"^(?P<label>.+)_(?P<index>\d{1,3})_Datos\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    return parser.parse_args()


def csv_dimensions(path: Path) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if len(rows) != 21:
        raise ValueError(f"Se esperaban cabecera y 20 frames: {path.name}")
    columns = len(rows[0])
    if columns != 202 or any(len(row) != columns for row in rows):
        raise ValueError(f"Se esperaban 202 columnas uniformes: {path.name}")
    return (20, 201)


def main() -> None:
    args = parse_args()
    candidates = sorted(args.source_root.rglob("*_Datos.csv"))
    if len(candidates) != 3000:
        raise ValueError(f"Se esperaban 3000 CSV, se hallaron {len(candidates)}")
    rows: list[dict[str, str]] = []
    invalid: list[str] = []
    for path in candidates:
        match = FILENAME.match(path.name)
        if not match:
            invalid.append(path.name)
            continue
        label, sample_index = match.group("label"), int(match.group("index"))
        if not 1 <= sample_index <= 100:
            invalid.append(path.name)
            continue
        frames, channels = csv_dimensions(path)
        source_partition = "training_validation_published" if sample_index <= 85 else "testing_published"
        split_external = (
            "train" if sample_index <= 68 else "validation" if sample_index <= 85 else "test"
        )
        rows.append(
            {
                "sample_id": f"{label}_{sample_index:03d}",
                "label": label,
                "sample_index": str(sample_index),
                "source_partition": source_partition,
                "split_external": split_external,
                "feature_path": str(path.relative_to(args.source_root)),
                "frames": str(frames),
                "channels": str(channels),
                "participant_id": "unavailable_in_release",
            }
        )
    labels = Counter(row["label"] for row in rows)
    split_counts = Counter(row["split_external"] for row in rows)
    if invalid or len(rows) != 3000 or len(labels) != 30:
        raise ValueError(f"Inventario inválido: filas={len(rows)} clases={len(labels)} inválidos={invalid[:5]}")
    if split_counts != Counter({"train": 2040, "validation": 510, "test": 450}):
        raise ValueError(f"Split inesperado: {dict(split_counts)}")
    if set(labels.values()) != {100}:
        raise ValueError("Cada seña debe conservar 100 secuencias")
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.out_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(args.out_manifest.read_bytes()).hexdigest()
    report = {
        "kind": "mejia_keypoints30_manifest",
        "rows": len(rows),
        "classes": len(labels),
        "per_class": dict(sorted(labels.items())),
        "split_counts": dict(split_counts),
        "shape": [20, 201],
        "source_partition_preserved": True,
        "participant_ids_recoverable": False,
        "validation_rule": "published TrainingValidation, deterministic per-class indices 69-85",
        "test_rule": "published Testing, per-class indices 86-100",
        "manifest_sha256": digest,
        "benchmark_210_words_touched": False,
        "s08_read": False,
        "s09_read": False,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()