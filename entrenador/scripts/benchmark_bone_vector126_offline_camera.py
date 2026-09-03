"""Benchmark offline de la ruta bone_vector126: landmarks ya disponibles → ONNX CPU."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

PARENTS = (0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19)
MCP = (5, 9, 13, 17)


def quantiles(values: list[float]) -> dict[str, float]:
    points = np.asarray(values, dtype=np.float64)
    return {"p50": float(np.percentile(points, 50)), "p95": float(np.percentile(points, 95)), "mean": float(points.mean()), "min": float(points.min()), "max": float(points.max())}


def form_hand(present: str, raw: str) -> np.ndarray:
    if present != "1":
        return np.zeros((63,), dtype=np.float32)
    points = np.asarray([float(value) for value in raw.split(",")], dtype=np.float64).reshape(21, 3)
    scale = np.linalg.norm(points[list(MCP)], axis=1).mean()
    if not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError("Escala palmar inválida")
    bones = np.stack([points[index] - points[parent] for index, parent in enumerate(PARENTS, start=1)]) / scale
    palm = points[list(MCP)].mean(axis=0, keepdims=True) / scale
    return np.concatenate([bones.reshape(-1), palm.reshape(-1)]).astype(np.float32)


def form_window(lines: list[str]) -> np.ndarray:
    rows = []
    for line in lines:
        right_present, left_present, _pose, right_raw, left_raw, *_ = line.split("|")
        rows.append(np.concatenate([form_hand(left_present, left_raw), form_hand(right_present, right_raw)]))
    output = np.stack(rows)
    if output.shape != (30, 126) or not np.isfinite(output).all():
        raise ValueError("Ventana bone_vector126 inválida")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 10:
        raise ValueError("warmup ≥1 y repetitions ≥10")
    lines = args.fixture.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) != 30:
        raise ValueError("El fixture debe contener 30 frames")
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name, output_name = session.get_inputs()[0].name, session.get_outputs()[0].name
    if session.get_inputs()[0].shape != [1, 30, 126] or session.get_outputs()[0].shape != [1, 210]:
        raise ValueError("Contrato ONNX bone_vector126 inesperado")
    for _ in range(args.warmup):
        tensor = form_window(lines)[None, ...]
        session.run([output_name], {input_name: tensor})
    formation, inference = [], []
    logits_finite = True
    for _ in range(args.repetitions):
        start = time.perf_counter_ns()
        tensor = form_window(lines)[None, ...]
        formation.append((time.perf_counter_ns() - start) / 1_000_000)
        start = time.perf_counter_ns()
        logits = session.run([output_name], {input_name: tensor})[0]
        inference.append((time.perf_counter_ns() - start) / 1_000_000)
        logits_finite = logits_finite and bool(np.isfinite(logits).all()) and logits.shape == (1, 210)
    report = {"kind": "offline_fixture_to_onnx_cpu_benchmark", "fixture_frames": 30, "repetitions": args.repetitions, "warmup": args.warmup, "formation_ms": quantiles(formation), "onnx_cpu_inference_ms": quantiles(inference), "logits_finite": logits_finite, "images_used": False, "camera_used": False, "mediapipe_runtime_measured": False, "physical_device_latency_measured": False, "interpretation": "Referencia de CPU sandbox para ruta lógica; no representa cámara, MediaPipe nativo, iPhone ni Android."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()