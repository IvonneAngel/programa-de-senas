"""Forma bone_vector126 desde el fixture móvil real anonimizado, sin vídeo ni etiquetas."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PARENTS = (0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19)
MCP = (5, 9, 13, 17)


def hand(present: str, raw: str) -> np.ndarray:
    if present != "1":
        return np.zeros((21, 3), dtype=np.float64)
    points = np.asarray([float(value) for value in raw.split(",")], dtype=np.float64).reshape(21, 3)
    scale = np.mean(np.linalg.norm(points[list(MCP)], axis=1))
    if not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError("Escala palmar inválida")
    bones = np.stack([points[index] - points[parent] for index, parent in enumerate(PARENTS, start=1)]) / scale
    palm = points[list(MCP)].mean(axis=0, keepdims=True) / scale
    return np.vstack([bones, palm])


def main() -> None:
    fixture, out = map(Path, sys.argv[1:3])
    rows = []
    for line in fixture.read_text(encoding="utf-8").strip().splitlines():
        right_present, left_present, _pose_present, right_raw, left_raw, *_ = line.split("|")
        rows.append(np.concatenate([hand(left_present, left_raw).reshape(-1), hand(right_present, right_raw).reshape(-1)]))
    values = np.stack(rows).astype(np.float32)
    if values.shape != (30, 126) or not np.isfinite(values).all():
        raise ValueError("Salida bone_vector126 inválida")
    out.write_text(json.dumps({"shape": [30, 126], "values": values.reshape(-1).tolist()}) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()