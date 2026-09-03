"""Construye un manifiesto verificable del alfabeto LSM 3D normalizado de Mendeley."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


LETTERS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "Y")
SPLITS = {**{index: "train" for index in range(1, 12)}, **{index: "validation" for index in range(12, 14)}, **{index: "test" for index in range(14, 16)}}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_file(path: Path) -> tuple[str, int]:
    label = path.parent.name.upper()
    match = re.fullmatch(r"([a-z]+)(\d+)", path.stem.lower())
    if label not in LETTERS or match is None or match.group(1).upper() != label:
        raise ValueError(f"Nombre no canónico: {path}")
    capture_index = int(match.group(2))
    if capture_index not in SPLITS:
        raise ValueError(f"Índice de captura fuera de 1–15: {path}")
    return label, capture_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    candidates = sorted(args.source_root.rglob("*.txt"))
    if len(candidates) != 315:
        raise ValueError(f"Se esperaban 315 txt y se obtuvieron {len(candidates)}")
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for path in candidates:
        label, capture_index = parse_file(path)
        key = (label, capture_index)
        if key in seen:
            raise ValueError(f"Duplicado de letra/índice: {key}")
        seen.add(key)
        values = np.loadtxt(path, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 2 or not np.isfinite(values).all():
            raise ValueError(f"Nube 3D inválida: {path}")
        rows.append({"sample_id": f"static3d_{label.lower()}_{capture_index:02d}", "label": label, "capture_index": capture_index, "split_external": SPLITS[capture_index], "feature_path": str(path.relative_to(args.source_root)), "points": int(values.shape[0]), "channels": 3, "capture_index_is_verified_participant": False})
    labels = {row["label"] for row in rows}
    expected = {(label, capture_index) for label in LETTERS for capture_index in range(1, 16)}
    if set(seen) != expected or labels != set(LETTERS):
        raise ValueError("Cobertura de letras/índices incompleta")
    rows.sort(key=lambda row: (str(row["label"]), int(row["capture_index"])))
    counts = {split: sum(row["split_external"] == split for row in rows) for split in ("train", "validation", "test")}
    if counts != {"train": 231, "validation": 42, "test": 42}:
        raise AssertionError(f"Split externo inesperado: {counts}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {"kind": "mendeley_static3d_alphabet_normalized_manifest", "source_root": str(args.source_root), "source_files": 315, "letters": list(LETTERS), "classes": 21, "split_counts": counts, "participant_ids_recoverable": False, "capture_indices": list(range(1, 16)), "training_eligible_splits": ["train", "validation"], "test_closed_by_default": True, "manifest_sha256": sha256(args.out), "s08_read": False, "s09_read": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()