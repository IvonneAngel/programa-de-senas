"""Audita el recurso CC BY de nubes 3D del alfabeto LSM sin entrenar modelos."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


LETTERS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "Y")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, str | int | float]] = []
    bad_normalization = 0
    for letter in LETTERS:
        directory = args.input_root / letter
        if not directory.is_dir():
            raise ValueError(f"Falta directorio de letra: {directory}")
        files = sorted(directory.glob(f"{letter.lower()}*.txt"), key=lambda path: int(path.stem[1:]))
        if len(files) != 15:
            raise ValueError(f"{letter} requiere 15 nubes, no {len(files)}")
        for file in files:
            index = int(file.stem[1:])
            if not 1 <= index <= 15:
                raise ValueError(f"Índice fuera de rango: {file.name}")
            points = np.loadtxt(file, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3 or not np.isfinite(points).all():
                raise ValueError(f"Nube inválida: {file}")
            centroid_norm = float(np.linalg.norm(points.mean(axis=0)))
            max_radius = float(np.linalg.norm(points, axis=1).max())
            within_strict_normalization = centroid_norm <= 1e-5 and max_radius <= 1.00001
            if not within_strict_normalization:
                bad_normalization += 1
            rows.append({
                "sample_id": f"{letter.lower()}{index}",
                "label_lsm_static": letter,
                "performer_index_unverified": index,
                "relative_path": str(file.relative_to(args.input_root)),
                "points": int(points.shape[0]),
                "centroid_norm": centroid_norm,
                "max_radius": max_radius,
                "within_strict_normalization": within_strict_normalization,
                "sha256": file_hash(file),
                "dataset_doi": "10.17632/sjt79hnb2f.2",
                "license": "CC BY 4.0",
                "split": "unassigned",
            })
    if len(rows) != 315 or Counter(row["label_lsm_static"] for row in rows) != Counter({letter: 15 for letter in LETTERS}):
        raise AssertionError("Cobertura de 21×15 inválida")
    duplicate_hash_groups = {
        digest: sorted(row["sample_id"] for row in rows if row["sha256"] == digest)
        for digest, count in Counter(row["sha256"] for row in rows).items()
        if count > 1
    }
    normalization_outliers = [
        {"sample_id": row["sample_id"], "centroid_norm": row["centroid_norm"], "max_radius": row["max_radius"]}
        for row in rows if not row["within_strict_normalization"]
    ]
    centroid_norms = [row["centroid_norm"] for row in rows]
    radii = [row["max_radius"] for row in rows]
    report = {
        "dataset_doi": "10.17632/sjt79hnb2f.2",
        "license": "CC BY 4.0",
        "classes": len(LETTERS),
        "samples": len(rows),
        "samples_per_class": 15,
        "point_count": {"min": min(row["points"] for row in rows), "max": max(row["points"] for row in rows), "mean": float(np.mean([row["points"] for row in rows]))},
        "max_centroid_norm": max(row["centroid_norm"] for row in rows),
        "max_radius": max(row["max_radius"] for row in rows),
        "normalization_percentiles": {"centroid_norm_p95": float(np.percentile(centroid_norms, 95)), "radius_p95": float(np.percentile(radii, 95))},
        "normalization_violations": bad_normalization,
        "normalization_outliers": normalization_outliers,
        "unique_content_hashes": len({row["sha256"] for row in rows}),
        "duplicate_hash_groups": duplicate_hash_groups,
        "performer_identity_mapping_verified": False,
        "trained_or_evaluated": False,
        "compatible_with_positions126_or_mobile_feature93": False,
    }
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.out_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()