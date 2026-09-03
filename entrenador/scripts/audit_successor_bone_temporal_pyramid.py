"""Audita una pirámide temporal fija sobre bone_vector126 solo en S01-S07."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def block_mean(values: np.ndarray, width: int) -> np.ndarray:
    if values.shape != (30, 126):
        raise ValueError(f"Se esperaba (30,126), se recibió {values.shape}")
    result = np.empty_like(values)
    for start in range(0, values.shape[0], width):
        stop = min(values.shape[0], start + width)
        result[start:stop] = values[start:stop].mean(axis=0, keepdims=True)
    return result


def pyramid(values: np.ndarray) -> np.ndarray:
    """Conserva la señal y añade residuales de media por bloque 2 y 6."""
    base = np.asarray(values, dtype=np.float32)
    return np.concatenate((base, base - block_mean(base, 2), base - block_mean(base, 6)), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [row for row in rows if row["split_model"] == "train"]
    if len(train) != 1470 or any(row["signer_id"] not in {f"S{i:02d}" for i in range(1, 8)} for row in train):
        raise ValueError("La auditoría exige exactamente S01-S07 como train")
    if any(row["split_model"] != "train" for row in train):
        raise AssertionError("Se detectó un split no train")
    energies_2: list[float] = []
    energies_6: list[float] = []
    nonzero_by_class: dict[str, int] = defaultdict(int)
    for row in train:
        base = np.load(args.cache_root / row["feature_path"], allow_pickle=False)
        transformed = pyramid(base)
        if transformed.shape != (30, 378) or not np.isfinite(transformed).all():
            raise ValueError(f"Pirámide inválida: {row['sample_id']}")
        if not np.array_equal(transformed[:, :126], base.astype(np.float32)):
            raise AssertionError("El prefijo bone_vector126 cambió")
        e2 = float(np.mean(np.abs(transformed[:, 126:252])))
        e6 = float(np.mean(np.abs(transformed[:, 252:])))
        energies_2.append(e2)
        energies_6.append(e6)
        if e2 > 1e-8 or e6 > 1e-8:
            nonzero_by_class[row["label_lsm"]] += 1
    report = {
        "kind": "successor_bone_temporal_pyramid_train_only_audit",
        "split_read": "train_only_S01_to_S07",
        "s08_read": False,
        "s09_read": False,
        "cache_written": False,
        "samples": len(train),
        "classes": len({row["label_lsm"] for row in train}),
        "signers": sorted({row["signer_id"] for row in train}),
        "candidate_shape": [30, 378],
        "detail_2_abs_mean": float(np.mean(energies_2)),
        "detail_2_abs_p95": float(np.percentile(energies_2, 95)),
        "detail_6_abs_mean": float(np.mean(energies_6)),
        "detail_6_abs_p95": float(np.percentile(energies_6, 95)),
        "classes_with_nonzero_detail": len(nonzero_by_class),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()