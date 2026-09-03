"""Compara salidas de paridad bone_vector126 sin cargar datos de evaluación."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    left, right, out = map(Path, sys.argv[1:4])
    arrays = [np.asarray(json.loads(path.read_text(encoding="utf-8"))["values"], dtype=np.float32) for path in (left, right)]
    if any(values.size != 30 * 126 for values in arrays):
        raise ValueError("Forma de sonda inválida")
    difference = np.abs(arrays[0] - arrays[1])
    report = {"shape": [30, 126], "max_abs_difference": float(difference.max()), "cells_above_1e5": int(np.sum(difference > 1e-5)), "images_used": False, "labels_used": False}
    if report["max_abs_difference"] > 1e-5:
        raise AssertionError(report)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()