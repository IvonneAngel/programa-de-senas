"""Deriva una vista canónica de las nubes LSM 3D CC BY desde los archivos."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_sort_key(sample_id: str) -> tuple[str, int]:
    return sample_id[0], int(sample_id[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.original_root.glob("*/*.txt"))
    if len(files) != 315:
        raise ValueError(f"Se esperaban 315 nubes originales, no {len(files)}")
    content_groups: dict[str, list[Path]] = defaultdict(list)
    for file in files:
        content_groups[sha256(file)].append(file)
    representatives = {digest: min(paths, key=lambda path: sample_sort_key(path.stem)) for digest, paths in content_groups.items()}
    rows: list[dict[str, str | int | float | bool]] = []
    for original_path in files:
        points = np.loadtxt(original_path, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3 or not np.isfinite(points).all():
            raise ValueError(f"Nube original inválida: {original_path}")
        centroid = points.mean(axis=0, keepdims=True)
        centered = points - centroid
        radius = float(np.linalg.norm(centered, axis=1).max())
        if radius <= 0.0 or not np.isfinite(radius):
            raise ValueError(f"Radio inválido: {original_path}")
        canonical = (centered / radius).astype(np.float32)
        if not np.isfinite(canonical).all() or float(np.linalg.norm(canonical.mean(axis=0))) > 1e-5 or float(np.linalg.norm(canonical, axis=1).max()) > 1.00001:
            raise AssertionError(f"Canónico inválido: {original_path}")
        relative = original_path.relative_to(args.original_root)
        destination = args.output_root / relative.with_suffix(".npy")
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, canonical, allow_pickle=False)
        digest = sha256(original_path)
        representative = representatives[digest]
        sample_id = original_path.stem
        rows.append({
            "sample_id": sample_id,
            "label_lsm_static": original_path.parent.name,
            "performer_index_unverified": int(sample_id[1:]),
            "source_v1_relative_path": str(relative),
            "source_v1_sha256": digest,
            "canonical_relative_path": str(destination.relative_to(args.output_root)),
            "canonical_sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
            "source_centroid_x": float(centroid[0, 0]),
            "source_centroid_y": float(centroid[0, 1]),
            "source_centroid_z": float(centroid[0, 2]),
            "source_radius": radius,
            "content_duplicate_group_size": len(content_groups[digest]),
            "is_duplicate_content": len(content_groups[digest]) > 1,
            "is_quarantined_duplicate_copy": original_path != representative,
            "split": "unassigned",
            "dataset_doi": "10.17632/sjt79hnb2f.2",
            "license": "CC BY 4.0",
        })
    if len(rows) != 315:
        raise AssertionError("Cobertura canónica inválida")
    report = {
        "source": "v1 original point clouds in CC BY package",
        "derivation": "subtract per-cloud centroid then divide by per-cloud maximum Euclidean radius",
        "samples_derived": len(rows),
        "unique_content_representatives": len(representatives),
        "quarantined_duplicate_copies": sum(bool(row["is_quarantined_duplicate_copy"]) for row in rows),
        "split_assigned": False,
        "trained_or_evaluated": False,
        "mobile_or_positions126_compatible": False,
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