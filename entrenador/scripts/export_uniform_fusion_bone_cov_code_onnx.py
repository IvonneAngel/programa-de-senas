"""Exporta ONNX de bone/cov/code y verifica la fusión uniforme solo sobre S01-S07."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from lsm.models.tcn import TemporalTCN

PARITY_TOLERANCE = 2e-5


@dataclass(frozen=True)
class Component:
    name: str
    dimension: int
    checkpoint: Path
    manifest: Path
    cache_root: Path
    onnx_path: Path
    expected_sha256: str


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split_model"] == "train" and row["feature_status"] == "ok"]
    if len(rows) != 1470:
        raise ValueError("La paridad solo puede usar S01-S07 train")
    return sorted(rows, key=lambda row: row["sample_id"])


def export_component(component: Component) -> tuple[TemporalTCN, dict[str, int], ort.InferenceSession, list[dict[str, str]]]:
    if sha256(component.checkpoint) != component.expected_sha256:
        raise ValueError(f"Checkpoint alterado: {component.name}")
    state = torch.load(component.checkpoint, map_location="cpu", weights_only=False)
    labels: dict[str, int] = state["labels"]
    if len(labels) != 210 or sorted(labels.values()) != list(range(210)):
        raise ValueError(f"Etiquetas inválidas: {component.name}")
    model = TemporalTCN(feature_dim=component.dimension, classes=210, frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    component.onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros((1, 30, component.dimension), dtype=torch.float32), component.onnx_path, input_names=["features"], output_names=["logits"], opset_version=17, do_constant_folding=True)
    session = ort.InferenceSession(str(component.onnx_path), providers=["CPUExecutionProvider"])
    input_meta, output_meta = session.get_inputs()[0], session.get_outputs()[0]
    if input_meta.name != "features" or list(input_meta.shape) != [1, 30, component.dimension] or output_meta.name != "logits" or list(output_meta.shape) != [1, 210]:
        raise AssertionError(f"Contrato ONNX inválido: {component.name}")
    return model, labels, session, train_rows(component.manifest)


def feature(component: Component, row: dict[str, str]) -> np.ndarray:
    values = np.load(component.cache_root / row["feature_path"], allow_pickle=False).astype(np.float32, copy=False)
    if values.shape != (30, component.dimension) or not np.isfinite(values).all():
        raise ValueError(f"Tensor inválido: {component.name}/{row['sample_id']}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bone-checkpoint", type=Path, required=True)
    parser.add_argument("--bone-manifest", type=Path, required=True)
    parser.add_argument("--bone-cache-root", type=Path, required=True)
    parser.add_argument("--bone-onnx", type=Path, required=True)
    parser.add_argument("--bone-sha256", required=True)
    parser.add_argument("--cov-checkpoint", type=Path, required=True)
    parser.add_argument("--cov-manifest", type=Path, required=True)
    parser.add_argument("--cov-cache-root", type=Path, required=True)
    parser.add_argument("--cov-onnx", type=Path, required=True)
    parser.add_argument("--cov-sha256", required=True)
    parser.add_argument("--code-checkpoint", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--code-cache-root", type=Path, required=True)
    parser.add_argument("--code-onnx", type=Path, required=True)
    parser.add_argument("--code-sha256", required=True)
    parser.add_argument("--out-metadata", type=Path, required=True)
    args = parser.parse_args()
    components = [
        Component("bone", 126, args.bone_checkpoint, args.bone_manifest, args.bone_cache_root, args.bone_onnx, args.bone_sha256),
        Component("cov", 168, args.cov_checkpoint, args.cov_manifest, args.cov_cache_root, args.cov_onnx, args.cov_sha256),
        Component("code", 190, args.code_checkpoint, args.code_manifest, args.code_cache_root, args.code_onnx, args.code_sha256),
    ]
    loaded = [export_component(component) for component in components]
    labels = loaded[0][1]
    rows = loaded[0][3]
    if any(item[1] != labels for item in loaded[1:]) or any([(row["sample_id"], row["label_lsm"]) for row in item[3]] != [(row["sample_id"], row["label_lsm"]) for row in rows] for item in loaded[1:]):
        raise ValueError("Componentes no comparten etiquetas o población train")
    checks = []
    for row_index in range(3):
        torch_logits, onnx_logits = [], []
        for component, (model, _, session, component_rows) in zip(components, loaded):
            values = feature(component, component_rows[row_index])
            with torch.no_grad():
                torch_value = model(torch.from_numpy(values).unsqueeze(0)).numpy()
            onnx_value = session.run(["logits"], {"features": values[None, ...]})[0]
            difference = float(np.max(np.abs(torch_value - onnx_value)))
            if difference > PARITY_TOLERANCE or int(torch_value.argmax()) != int(onnx_value.argmax()):
                raise AssertionError(f"Paridad componente fallida: {component.name}/{rows[row_index]['sample_id']}; diferencia={difference}")
            torch_logits.append(torch_value)
            onnx_logits.append(onnx_value)
        torch_fusion = sum(torch_logits) / 3.0
        onnx_fusion = sum(onnx_logits) / 3.0
        fusion_difference = float(np.max(np.abs(torch_fusion - onnx_fusion)))
        if fusion_difference > PARITY_TOLERANCE or int(torch_fusion.argmax()) != int(onnx_fusion.argmax()):
            raise AssertionError(f"Paridad de fusión fallida: {row['sample_id']}")
        checks.append({"sample_id": rows[row_index]["sample_id"], "fusion_max_abs_difference": fusion_difference, "argmax": int(onnx_fusion.argmax())})
    metadata = {
        "kind": "uniform_logit_fusion_bone_cov_code_experimental",
        "components": [{"name": component.name, "onnx": str(component.onnx_path), "feature_shape": [1, 30, component.dimension], "checkpoint_sha256": component.expected_sha256} for component in components],
        "output_shape": [1, 210],
        "labels": [label for label, _ in sorted(labels.items(), key=lambda item: item[1])],
        "fusion": "(bone_logits + cov_logits + code_logits) / 3", "weights": [1 / 3, 1 / 3, 1 / 3],
        "parity_train_only": checks, "parity_max_abs_tolerance": PARITY_TOLERANCE, "s08_read": False, "s09_read": False, "mobile_camera_validated": False,
        "warning": "Exportación offline; no activa cámara, no mide hardware móvil y conserva la compuerta experimental.",
    }
    args.out_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.out_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": str(args.out_metadata), "checks": checks, "s08_read": False, "s09_read": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()