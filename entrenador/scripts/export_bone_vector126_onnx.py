"""Exporta el checkpoint bone_vector126 s42 a ONNX y verifica paridad fuera de S08/S09."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from lsm.models.tcn import TemporalTCN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-onnx", type=Path, required=True)
    parser.add_argument("--out-metadata", type=Path, required=True)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    labels: dict[str, int] = state["labels"]
    if len(labels) != 210 or sorted(labels.values()) != list(range(210)):
        raise ValueError("El checkpoint debe tener exactamente 210 etiquetas compactas")
    model = TemporalTCN(feature_dim=126, classes=210, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    example = torch.zeros((1, 30, 126), dtype=torch.float32)
    args.out_onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, example, args.out_onnx, input_names=["features"], output_names=["logits"], opset_version=17, do_constant_folding=True)
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split_model"] == "train" and row["feature_status"] == "ok"]
    if len(rows) != 1470:
        raise ValueError("La paridad ONNX solo puede usar las 1,470 filas train")
    session = ort.InferenceSession(str(args.out_onnx), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if input_meta.name != "features" or list(input_meta.shape) != [1, 30, 126] or output_meta.name != "logits" or list(output_meta.shape) != [1, 210]:
        raise AssertionError(f"Contrato ONNX inesperado: {input_meta.shape}→{output_meta.shape}")
    checks = []
    for row in rows[:3]:
        values = np.load(args.cache_root / row["feature_path"], allow_pickle=False).astype(np.float32, copy=False)
        if values.shape != (30, 126) or not np.isfinite(values).all():
            raise ValueError(f"Tensor train inválido: {row['sample_id']}")
        with torch.no_grad():
            pytorch = model(torch.from_numpy(values).unsqueeze(0)).numpy()
        onnx = session.run(["logits"], {"features": values[None, ...]})[0]
        difference = float(np.max(np.abs(pytorch - onnx)))
        if difference > 1e-5 or int(pytorch.argmax()) != int(onnx.argmax()):
            raise AssertionError(f"Paridad ONNX fallida para {row['sample_id']}: {difference}")
        checks.append({"sample_id": row["sample_id"], "max_abs_difference": difference, "argmax": int(onnx.argmax())})
    inverse_labels = [label for label, _ in sorted(labels.items(), key=lambda item: item[1])]
    metadata = {
        "kind": "bone_vector126_word_classifier_experimental",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": state["epoch"],
        "seed": 42,
        "input_name": "features",
        "feature_shape": [1, 30, 126],
        "output_name": "logits",
        "output_shape": [1, 210],
        "labels": inverse_labels,
        "representation": "bone_vector126: 20 directed bones and scaled mean MCP vector per hand",
        "selection": "s42 preselected by S08 primary before S09; separate from W33/W35",
        "parity_train_only": checks,
        "s08_read": False,
        "s09_read": False,
        "mobile_camera_validated": False,
    }
    args.out_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.out_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"onnx": str(args.out_onnx), "metadata": str(args.out_metadata), "checks": checks, "s08_read": False, "s09_read": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()