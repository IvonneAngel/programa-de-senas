"""Construye un fixture de landmarks S01-S07 para el benchmark offline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _serialize(points: np.ndarray) -> str:
    return ",".join(format(float(value), ".9g") for value in points.reshape(-1))


def build_fixture(manifest: Path, positions_root: Path, sample_id: str, out_fixture: Path, out_metadata: Path) -> dict[str, object]:
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    row = next((candidate for candidate in rows if candidate["sample_id"] == sample_id), None)
    if row is None:
        raise ValueError(f"sample_id inexistente: {sample_id}")
    if row["split_model"] != "train" or row["signer_id"] in {"S08", "S09"}:
        raise ValueError("El fixture offline exige una fila train S01-S07; S08/S09 están prohibidos")
    path = positions_root / f"{sample_id}.npy"
    values = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    if values.shape != (30, 126) or not np.isfinite(values).all():
        raise ValueError(f"Landmarks inválidos: {values.shape}")
    lines: list[str] = []
    for frame in values:
        left, right = frame[:63], frame[63:]
        left_present, right_present = str(int(bool(left.any()))), str(int(bool(right.any())))
        lines.append("|".join((right_present, left_present, "0", _serialize(right), _serialize(left))))
    out_fixture.parent.mkdir(parents=True, exist_ok=True)
    out_fixture.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report: dict[str, object] = {
        "kind": "offline_bone_fixture_train_only",
        "sample_id": sample_id,
        "signer_id": row["signer_id"],
        "split_model": row["split_model"],
        "frames": 30,
        "positions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(out_fixture.read_bytes()).hexdigest(),
        "camera_used": False,
        "images_used": False,
        "mediapipe_runtime_used": False,
        "s08_read": False,
        "s09_read": False,
    }
    out_metadata.parent.mkdir(parents=True, exist_ok=True)
    out_metadata.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--positions-root", type=Path, required=True)
    parser.add_argument("--sample-id", default="mendeley_c001_s01")
    parser.add_argument("--out-fixture", type=Path, required=True)
    parser.add_argument("--out-metadata", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_fixture(args.manifest, args.positions_root, args.sample_id, args.out_fixture, args.out_metadata), ensure_ascii=False))


if __name__ == "__main__":
    main()