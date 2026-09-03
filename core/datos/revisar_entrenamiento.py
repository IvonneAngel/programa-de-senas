from __future__ import annotations

import json
from pathlib import Path
from time import strftime
from typing import Any

import numpy as np

from core.base.configuracion import FRAME_COUNT, KEYPOINT_SIZE
from core.base.rutas import ruta_entrenamiento
from core.datos.conjunto_datos import validate_real_sequence


DEFAULT_TARGET_WORDS = [
    "hola",
    "gracias",
    "si",
    "no",
    "ayuda",
    "nada",
    "te_quiero",
]
MIN_SEQUENCES_PER_WORD = 20
STRONG_SEQUENCES_PER_WORD = 50


def normalize_label(label: str) -> str:
    """normalize label."""
    normalized = label.strip().lower().replace(" ", "_")
    replacements = {
        "sí": "si",
        "tequiero": "te_quiero",
        "te-quiero": "te_quiero",
    }
    return replacements.get(normalized, normalized)


def _is_valid_sequence(path: Path, frame_count: int, keypoint_size: int) -> bool:
    try:
        validate_real_sequence(np.load(path, allow_pickle=False), frame_count, keypoint_size)
    except Exception:
        return False
    return True


def count_local_sequences(
    data_dir: str | Path,
    *,
    frame_count: int = FRAME_COUNT,
    keypoint_size: int = KEYPOINT_SIZE,
) -> dict[str, int]:
    root = Path(data_dir)
    if not root.exists():
        return {}
    counts: dict[str, int] = {}
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        direct_files = list(path.glob("*.npy"))
        if direct_files:
            label = normalize_label(path.name)
            counts[label] = counts.get(label, 0) + sum(
                1 for file_path in direct_files if _is_valid_sequence(file_path, frame_count, keypoint_size)
            )
            continue
        for label_dir in sorted(child for child in path.iterdir() if child.is_dir()):
            label = normalize_label(label_dir.name)
            counts[label] = counts.get(label, 0) + sum(
                1 for file_path in label_dir.glob("*.npy") if _is_valid_sequence(file_path, frame_count, keypoint_size)
            )
    return counts


def default_readiness_data_dir(app_path: Path) -> Path:
    current = ruta_entrenamiento() / "palabras y oraciones"
    if current.exists():
        return current
    nueva = app_path / "palabras y oraciones"
    if nueva.exists():
        return nueva
    return app_path / "grabaciones"


def _word_status(count: int, minimum: int, strong_minimum: int) -> str:
    if count >= strong_minimum:
        return "strong"
    if count >= minimum:
        return "trainable"
    if count > 0:
        return "partial"
    return "missing"


def build_training_readiness(
    app_dir: str | Path,
    *,
    data_dir: str | Path | None = None,
    target_words: list[str] | None = None,
    min_sequences_per_word: int = MIN_SEQUENCES_PER_WORD,
    strong_sequences_per_word: int = STRONG_SEQUENCES_PER_WORD,
) -> dict[str, Any]:
    app_path = Path(app_dir)
    data_path = Path(data_dir) if data_dir is not None else default_readiness_data_dir(app_path)
    target = [normalize_label(word) for word in (target_words or DEFAULT_TARGET_WORDS)]
    counts = count_local_sequences(data_path)

    rows: list[dict[str, Any]] = []
    for word in target:
        count = counts.get(word, 0)
        rows.append(
            {
                "word": word,
                "sequence_count": count,
                "missing_to_minimum": max(0, min_sequences_per_word - count),
                "missing_to_strong": max(0, strong_sequences_per_word - count),
                "status": _word_status(count, min_sequences_per_word, strong_sequences_per_word),
            }
        )

    labels_with_any_data = [word for word, count in counts.items() if count > 0]
    target_trainable = [row["word"] for row in rows if row["sequence_count"] >= min_sequences_per_word]
    target_strong = [row["word"] for row in rows if row["sequence_count"] >= strong_sequences_per_word]
    partial_words = [row["word"] for row in rows if row["status"] == "partial"]
    missing_words = [row["word"] for row in rows if row["status"] == "missing"]

    minimal_training_possible = len(labels_with_any_data) >= 2
    recommended_training_ready = len(target_trainable) >= 2
    strong_training_ready = len(target_strong) >= 2

    if strong_training_ready:
        recommendation = "ready_for_strong_training"
    elif recommended_training_ready:
        recommendation = "ready_for_basic_training"
    elif minimal_training_possible:
        recommendation = "collect_more_before_serious_training"
    else:
        recommendation = "not_ready"

    return {
        "generated_at": strftime("%Y-%m-%d %H:%M:%S"),
        "app_dir": str(app_path),
        "data_dir": str(data_path),
        "target_words": target,
        "min_sequences_per_word": min_sequences_per_word,
        "strong_sequences_per_word": strong_sequences_per_word,
        "local_sequence_counts": counts,
        "rows": rows,
        "summary": {
            "target_word_count": len(target),
            "labels_with_any_data": labels_with_any_data,
            "target_trainable_count": len(target_trainable),
            "target_strong_count": len(target_strong),
            "partial_words": partial_words,
            "missing_words": missing_words,
            "minimal_training_possible": minimal_training_possible,
            "recommended_training_ready": recommended_training_ready,
            "strong_training_ready": strong_training_ready,
            "recommendation": recommendation,
        },
        "next_actions": _next_actions(recommendation, rows),
    }


def _next_actions(recommendation: str, rows: list[dict[str, Any]]) -> list[str]:
    missing = [row for row in rows if row["missing_to_minimum"] > 0]
    if recommendation == "ready_for_strong_training":
        return [
            "Entrenar en Colab TPU para comparar contra el modelo local.",
            "Guardar matriz de confusion y validar con videos reales.",
        ]
    if recommendation == "ready_for_basic_training":
        return [
            "Entrenar modelo base con las palabras que ya tienen minimo de muestras.",
            "Subir a Colab si quieres entrenamiento mas largo y metricas mas completas.",
        ]
    if recommendation == "collect_more_before_serious_training":
        hardest = sorted(missing, key=lambda row: row["missing_to_minimum"], reverse=True)[:4]
        words = ", ".join(f"{row['word']} (+{row['missing_to_minimum']})" for row in hardest)
        return [
            "Hay datos parciales, pero no suficientes para entrenamiento serio.",
            f"Prioridad de captura/importacion: {words}" if words else "Completar palabras faltantes.",
            "Usar Colab solo cuando haya datos suficientes; la TPU no arregla un dataset vacio.",
        ]
    return [
        "No entrenar TensorFlow todavia.",
        "Completar al menos dos palabras con muestras reales de secuencia.",
        "Meta minima recomendada: 20 secuencias por palabra; meta fuerte: 50.",
    ]


def save_training_readiness_report(readiness: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "training_readiness.json"
    md_path = output_path / "training_readiness.md"
    json_path.write_text(json.dumps(readiness, indent=2, ensure_ascii=True), encoding="utf-8")

    lines = [
        "# Preparacion de Entrenamiento",
        "",
        f"Generado: {readiness['generated_at']}",
        f"Recomendacion: {readiness['summary']['recommendation']}",
        f"Minimo por palabra: {readiness['min_sequences_per_word']}",
        f"Meta fuerte por palabra: {readiness['strong_sequences_per_word']}",
        "",
        "## Palabras objetivo",
        "",
    ]
    for row in readiness["rows"]:
        lines.append(
            f"- {row['word']}: {row['sequence_count']} secuencias, "
            f"faltan {row['missing_to_minimum']} para minimo, estado {row['status']}"
        )

    lines.extend(["", "## Resumen", ""])
    summary = readiness["summary"]
    lines.append(f"- Entrenamiento minimo posible: {'si' if summary['minimal_training_possible'] else 'no'}")
    lines.append(f"- Entrenamiento recomendado listo: {'si' if summary['recommended_training_ready'] else 'no'}")
    lines.append(f"- Entrenamiento fuerte listo: {'si' if summary['strong_training_ready'] else 'no'}")
    lines.append(f"- Palabras parciales: {', '.join(summary['partial_words']) or 'ninguna'}")
    lines.append(f"- Palabras faltantes: {', '.join(summary['missing_words']) or 'ninguna'}")

    lines.extend(["", "## Siguientes acciones", ""])
    for action in readiness["next_actions"]:
        lines.append(f"- {action}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path