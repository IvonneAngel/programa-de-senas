from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import strftime
from typing import Any

import numpy as np

from core.base.configuracion import FRAME_COUNT, KEYPOINT_SIZE
from core.datos.conjunto_datos import validate_real_sequence
from core.datos.dactilologia import DYNAMIC_LETTERS, LSM_ALPHABET
from core.datos.oraciones import SENTENCE_TARGET_COUNT, build_sentence_curriculum


ALPHABET_DATA_DIR = "alfabeto"
SENTENCE_DATA_DIR = "oraciones"
MINIMUM_SEQUENCES_PER_TARGET = 20
STRONG_SEQUENCES_PER_TARGET = 50


@dataclass(frozen=True, slots=True)
class AlphabetTarget:
    target_id: str
    letter: str
    data_label: str
    dynamic: bool
    model_status: str = "objetivo_no_entrenado"
    recognition_unit: str = "letter"
    minimum_sequences: int = MINIMUM_SEQUENCES_PER_TARGET
    strong_sequences: int = STRONG_SEQUENCES_PER_TARGET


def safe_label(value: str) -> str:
    normalized = value.strip().lower()
    replacements = {
        "ñ": "n_tilde",
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    cleaned = []
    for character in normalized:
        cleaned.append(character if character.isalnum() else "_")
    return "_".join(part for part in "".join(cleaned).split("_") if part)


def build_alphabet_targets() -> tuple[AlphabetTarget, ...]:
    return tuple(
        AlphabetTarget(
            target_id=f"letter_{safe_label(letter)}",
            letter=letter,
            data_label=f"letter_{safe_label(letter)}",
            dynamic=letter in DYNAMIC_LETTERS,
        )
        for letter in LSM_ALPHABET
    )


def inspect_sequence_folder(
    folder: Path,
    *,
    frame_count: int = FRAME_COUNT,
    keypoint_size: int = KEYPOINT_SIZE,
) -> dict[str, int]:
    if not folder.exists():
        return {"file_count": 0, "valid_sequence_count": 0, "invalid_sequence_count": 0}

    file_count = 0
    valid_count = 0
    invalid_count = 0
    for path in sorted(folder.glob("*.npy")):
        if not path.is_file():
            continue
        file_count += 1
        try:
            validate_real_sequence(np.load(path), frame_count, keypoint_size)
            valid_count += 1
        except Exception:
            invalid_count += 1

    return {
        "file_count": file_count,
        "valid_sequence_count": valid_count,
        "invalid_sequence_count": invalid_count,
    }


def build_coverage_report(project_dir: str | Path) -> dict[str, Any]:
    root = Path(project_dir)
    alphabet_root = root / ALPHABET_DATA_DIR
    sentence_root = root / SENTENCE_DATA_DIR

    alphabet_rows = []
    for target in build_alphabet_targets():
        counts = inspect_sequence_folder(alphabet_root / target.data_label)
        count = counts["valid_sequence_count"]
        alphabet_rows.append(
            {
                **asdict(target),
                "sequence_count": count,
                "file_count": counts["file_count"],
                "invalid_sequence_count": counts["invalid_sequence_count"],
                "minimum_ready": count >= target.minimum_sequences and counts["invalid_sequence_count"] == 0,
                "strong_ready": count >= target.strong_sequences and counts["invalid_sequence_count"] == 0,
            }
        )

    sentence_rows = []
    for target in build_sentence_curriculum():
        counts = inspect_sequence_folder(sentence_root / target.target_id)
        count = counts["valid_sequence_count"]
        sentence_rows.append(
            {
                **asdict(target),
                "sequence_count": count,
                "file_count": counts["file_count"],
                "invalid_sequence_count": counts["invalid_sequence_count"],
                "minimum_ready": count >= target.minimum_sequences and counts["invalid_sequence_count"] == 0,
                "strong_ready": count >= target.strong_sequences and counts["invalid_sequence_count"] == 0,
            }
        )

    ready_alphabet = sum(1 for row in alphabet_rows if row["minimum_ready"])
    ready_sentences = sum(1 for row in sentence_rows if row["minimum_ready"])
    strong_alphabet = sum(1 for row in alphabet_rows if row["strong_ready"])
    strong_sentences = sum(1 for row in sentence_rows if row["strong_ready"])
    invalid_alphabet_sequences = sum(row["invalid_sequence_count"] for row in alphabet_rows)
    invalid_sentence_sequences = sum(row["invalid_sequence_count"] for row in sentence_rows)

    return {
        "generated_at": strftime("%Y-%m-%d %H:%M:%S"),
        "alphabet": {
            "target_count": len(alphabet_rows),
            "ready_minimum_count": ready_alphabet,
            "ready_strong_count": strong_alphabet,
            "invalid_sequence_count": invalid_alphabet_sequences,
            "complete_minimum": ready_alphabet == len(alphabet_rows),
            "complete_strong": strong_alphabet == len(alphabet_rows),
            "data_dir": str(alphabet_root),
            "targets": alphabet_rows,
        },
        "sentences": {
            "target_count": len(sentence_rows),
            "required_target_count": SENTENCE_TARGET_COUNT,
            "ready_minimum_count": ready_sentences,
            "ready_strong_count": strong_sentences,
            "invalid_sequence_count": invalid_sentence_sequences,
            "complete_minimum": ready_sentences >= SENTENCE_TARGET_COUNT,
            "complete_strong": strong_sentences >= SENTENCE_TARGET_COUNT,
            "data_dir": str(sentence_root),
            "targets": sentence_rows,
        },
        "training_contract": {
            "minimum_sequences_per_target": MINIMUM_SEQUENCES_PER_TARGET,
            "strong_sequences_per_target": STRONG_SEQUENCES_PER_TARGET,
            "alphabet_goal": "27 LSM letters with static and dynamic targets separated.",
            "sentence_goal": "At least 700 sentence or intent targets with native LSM validation before real use.",
            "honesty_rule": f"Targets are not recognized until real labeled .npy sequences load, match shape ({FRAME_COUNT}, {KEYPOINT_SIZE}), contain finite values, include non-empty landmark signal, and meet the data contract.",
        },
    }


def save_coverage_report(project_dir: str | Path, output_dir: str | Path) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = build_coverage_report(project_dir)

    json_path = output_path / "lsm_coverage_report.json"
    md_path = output_path / "lsm_coverage_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    alphabet = report["alphabet"]
    sentences = report["sentences"]
    lines = [
        "# Cobertura LSM: alfabeto y oraciones",
        "",
        f"Generado: {report['generated_at']}",
        "",
        "## Resumen",
        "",
        f"- Letras objetivo: {alphabet['target_count']}",
        f"- Letras listas con minimo: {alphabet['ready_minimum_count']}/{alphabet['target_count']}",
        f"- Secuencias invalidas de letras: {alphabet['invalid_sequence_count']}",
        f"- Oraciones objetivo: {sentences['target_count']}",
        f"- Oraciones listas con minimo: {sentences['ready_minimum_count']}/{sentences['target_count']}",
        f"- Secuencias invalidas de oraciones: {sentences['invalid_sequence_count']}",
        f"- Minimo por objetivo: {MINIMUM_SEQUENCES_PER_TARGET} secuencias reales",
        f"- Meta fuerte por objetivo: {STRONG_SEQUENCES_PER_TARGET} secuencias reales",
        "",
        "## Estado",
        "",
        "- Un objetivo de alfabeto no esta reconocido hasta tener secuencias `.npy` validas en `alfabeto/<label>`.",
        "- Una oracion no esta reconocida hasta tener secuencias `.npy` validas en `oraciones/<sent_  # ###>`.",
        f"- Una secuencia vacia o todo en ceros no cuenta como dato real aunque tenga forma `({FRAME_COUNT}, {KEYPOINT_SIZE})`.",
        "- Las glosas de oraciones son borradores tecnicos y requieren validacion de una persona competente en LSM.",
        "",
        "## Primeras letras objetivo",
        "",
    ]
    for row in alphabet["targets"][:10]:
        lines.append(
            f"- {row['letter']}: {row['sequence_count']} validas, {row['invalid_sequence_count']} invalidas, dinamica={str(row['dynamic']).lower()}"
        )
    lines.extend(["", "## Primeras oraciones objetivo", ""])
    for row in sentences["targets"][:20]:
        lines.append(
            f"- {row['target_id']} [{row['domain']}]: {row['spanish']} -> {row['rough_lsm_gloss']} "
            f"({row['validation_status']})"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}