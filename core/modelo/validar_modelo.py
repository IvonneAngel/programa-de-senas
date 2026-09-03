from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.base.configuracion import FRAME_COUNT, KEYPOINT_SIZE


class ModelValidationError(RuntimeError):
    pass


def load_labels(labels_path: str | Path, *, min_labels: int = 1) -> list[str]:
    """load labels."""
    path = Path(labels_path)
    if not path.exists():
        raise ModelValidationError(f"No existe archivo de etiquetas: {path}")
    labels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not all(isinstance(label, str) and label.strip() for label in labels):
        raise ModelValidationError(f"Archivo de etiquetas invalido: {path}")
    if len(labels) < min_labels:
        raise ModelValidationError(f"Se requieren al menos {min_labels} etiquetas; encontre {len(labels)}")
    return labels


def _shape_to_list(shape: Any) -> list[Any]:
    if shape is None:
        return []
    if isinstance(shape, list):
        if shape and isinstance(shape[0], (list, tuple)):
            return list(shape[0])
        return list(shape)
    return list(shape)


def inspect_keras_model(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise ModelValidationError(f"No existe modelo Keras: {path}")

    import tensorflow as tf

    model = tf.keras.models.load_model(path, compile=False)
    input_shape = _shape_to_list(model.input_shape)
    output_shape = _shape_to_list(model.output_shape)
    return {
        "model_path": str(path),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "parameter_count": int(model.count_params()),
        "layer_count": len(model.layers),
    }


def validate_sequence_keras_model(
    model_path: str | Path,
    labels_path: str | Path,
    *,
    expected_frames: int = FRAME_COUNT,
    expected_keypoints: int = KEYPOINT_SIZE,
    min_labels: int = 2,
) -> dict[str, Any]:
    labels = load_labels(labels_path, min_labels=min_labels)
    info = inspect_keras_model(model_path)
    input_shape = info["input_shape"]
    output_shape = info["output_shape"]

    if len(input_shape) != 3:
        raise ModelValidationError(f"Entrada incompatible: esperaba rank 3, encontre {input_shape}")
    if input_shape[1] != expected_frames or input_shape[2] != expected_keypoints:
        raise ModelValidationError(
            f"Entrada incompatible: esperaba (None, {expected_frames}, {expected_keypoints}), encontre {input_shape}"
        )
    if not output_shape:
        raise ModelValidationError("Salida incompatible: output_shape vacio")
    output_classes = output_shape[-1]
    if output_classes != len(labels):
        raise ModelValidationError(
            f"Salida incompatible: el modelo tiene {output_classes} clases pero etiquetas tiene {len(labels)}"
        )

    return {
        "status": "compatible",
        "model_path": info["model_path"],
        "labels_path": str(Path(labels_path)),
        "label_count": len(labels),
        "labels": labels,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "parameter_count": info["parameter_count"],
        "layer_count": info["layer_count"],
        "expected_frames": expected_frames,
        "expected_keypoints": expected_keypoints,
    }


def validate_sequence_model_with_expected_labels(
    model_path: str | Path,
    labels_path: str | Path,
    expected_labels: list[str] | tuple[str, ...],
    *,
    expected_frames: int = FRAME_COUNT,
    expected_keypoints: int = KEYPOINT_SIZE,
) -> dict[str, Any]:
    expected = list(expected_labels)
    if not expected:
        raise ModelValidationError("La lista de etiquetas esperadas esta vacia")

    result = validate_sequence_keras_model(
        model_path,
        labels_path,
        expected_frames=expected_frames,
        expected_keypoints=expected_keypoints,
        min_labels=len(expected),
    )
    labels = result["labels"]
    if labels != expected:
        missing = [label for label in expected if label not in labels]
        unexpected = [label for label in labels if label not in expected]
        first_mismatch = None
        for index, (actual_label, expected_label) in enumerate(zip(labels, expected)):
            if actual_label != expected_label:
                first_mismatch = {
                    "index": index,
                    "actual": actual_label,
                    "expected": expected_label,
                }
                break
        raise ModelValidationError(
            "Etiquetas incompatibles con el contrato esperado: "
            f"esperaba {len(expected)}, encontre {len(labels)}, "
            f"faltantes={missing[:10]}, inesperadas={unexpected[:10]}, "
            f"primer_desajuste={first_mismatch}"
        )

    result["expected_label_count"] = len(expected)
    result["expected_labels_match"] = True
    return result


def validation_status(
    model_path: str | Path,
    labels_path: str | Path,
    *,
    expected_frames: int = FRAME_COUNT,
    expected_keypoints: int = KEYPOINT_SIZE,
    min_labels: int = 2,
) -> dict[str, Any]:
    try:
        return validate_sequence_keras_model(
            model_path,
            labels_path,
            expected_frames=expected_frames,
            expected_keypoints=expected_keypoints,
            min_labels=min_labels,
        )
    except Exception as exc:
        return {
            "status": "invalid",
            "model_path": str(Path(model_path)),
            "labels_path": str(Path(labels_path)),
            "error": str(exc),
            "expected_frames": expected_frames,
            "expected_keypoints": expected_keypoints,
        }


def expected_label_validation_status(
    model_path: str | Path,
    labels_path: str | Path,
    expected_labels: list[str] | tuple[str, ...],
    *,
    expected_frames: int = FRAME_COUNT,
    expected_keypoints: int = KEYPOINT_SIZE,
) -> dict[str, Any]:
    try:
        return validate_sequence_model_with_expected_labels(
            model_path,
            labels_path,
            expected_labels,
            expected_frames=expected_frames,
            expected_keypoints=expected_keypoints,
        )
    except Exception as exc:
        return {
            "status": "invalid",
            "model_path": str(Path(model_path)),
            "labels_path": str(Path(labels_path)),
            "error": str(exc),
            "expected_frames": expected_frames,
            "expected_keypoints": expected_keypoints,
            "expected_label_count": len(expected_labels),
            "expected_labels_match": False,
        }