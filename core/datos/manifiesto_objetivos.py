from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

from core.datos.cobertura_lsm import (
    ALPHABET_DATA_DIR,
    SENTENCE_DATA_DIR,
    build_alphabet_targets,
    safe_label,
)
from core.datos.oraciones import build_sentence_curriculum


TargetType = Literal["alphabet", "sentence"]


@dataclass(frozen=True, slots=True)
class TargetSpec:
    target_type: TargetType
    target_id: str
    data_dir: str
    data_label: str
    display_label: str
    capture_prompt: str
    domain: str
    recognition_unit: str
    validation_status: str
    minimum_sequences: int
    strong_sequences: int


def iter_alphabet_specs() -> tuple[TargetSpec, ...]:
    specs: list[TargetSpec] = []
    for target in build_alphabet_targets():
        specs.append(
            TargetSpec(
                target_type="alphabet",
                target_id=target.target_id,
                data_dir=ALPHABET_DATA_DIR,
                data_label=target.data_label,
                display_label=f"Letra {target.letter}",
                capture_prompt=(
                    f"Dactilologia LSM: letra {target.letter}. "
                    "Mantén la seña clara dentro del cuadro de cámara."
                ),
                domain="dactylology",
                recognition_unit=target.recognition_unit,
                validation_status="lsm_alphabet_target",
                minimum_sequences=target.minimum_sequences,
                strong_sequences=target.strong_sequences,
            )
        )
    return tuple(specs)


def iter_sentence_specs() -> tuple[TargetSpec, ...]:
    specs: list[TargetSpec] = []
    for target in build_sentence_curriculum():
        specs.append(
            TargetSpec(
                target_type="sentence",
                target_id=target.target_id,
                data_dir=SENTENCE_DATA_DIR,
                data_label=target.target_id,
                display_label=f"{target.target_id}: {target.spanish}",
                capture_prompt=(
                    f"{target.spanish} | glosa borrador: {target.rough_lsm_gloss}. "
                    "Validar con LSM nativa antes de usar como traducción real."
                ),
                domain=target.domain,
                recognition_unit=target.recognition_unit,
                validation_status=target.validation_status,
                minimum_sequences=target.minimum_sequences,
                strong_sequences=target.strong_sequences,
            )
        )
    return tuple(specs)


def iter_target_specs() -> Iterable[TargetSpec]:
    yield from iter_alphabet_specs()
    yield from iter_sentence_specs()


def target_specs_as_dicts() -> list[dict[str, object]]:
    return [asdict(spec) for spec in iter_target_specs()]


def resolve_target(target_type: TargetType, target: str) -> TargetSpec:
    normalized = target.strip()
    if not normalized:
        raise ValueError("Target is required.")

    if target_type == "alphabet":
        lookup = {}
        for spec in iter_alphabet_specs():
            letter = spec.display_label.replace("Letra ", "")
            lookup[spec.target_id.lower()] = spec
            lookup[spec.data_label.lower()] = spec
            lookup[letter.lower()] = spec
            lookup[safe_label(letter)] = spec
        key = normalized.lower()
        if key in lookup:
            return lookup[key]
        raise ValueError(f"Unknown alphabet target: {target!r}")

    if target_type == "sentence":
        lookup = {spec.target_id.lower(): spec for spec in iter_sentence_specs()}
        key = normalized.lower()
        if key in lookup:
            return lookup[key]
        raise ValueError(f"Unknown sentence target: {target!r}")

    raise ValueError(f"Unsupported target type: {target_type!r}")


def save_target_manifest(output_dir: str | Path) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = target_specs_as_dicts()

    json_path = output_path / "lsm_targets_manifest.json"
    csv_path = output_path / "lsm_targets_manifest.csv"
    md_path = output_path / "lsm_targets_manifest.md"

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    alphabet_count = sum(1 for row in rows if row["target_type"] == "alphabet")
    sentence_count = sum(1 for row in rows if row["target_type"] == "sentence")
    sentence_rows = [row for row in rows if row["target_type"] == "sentence"]
    domains = sorted({str(row["domain"]) for row in sentence_rows})

    lines = [
        "# Manifest de objetivos LSM",
        "",
        f"- Objetivos de alfabeto: {alphabet_count}",
        f"- Objetivos de oraciones/intenciones: {sentence_count}",
        f"- Total: {len(rows)}",
        "",
        "## Uso",
        "",
        "- `data_dir` indica la carpeta principal.",
        "- `data_label` indica la subcarpeta exacta para guardar secuencias `.npy`.",
        "- `capture_prompt` es guía de captura, no prueba de traducción real.",
        "",
        "## Alfabeto LSM",
        "",
    ]
    for row in rows:
        if row["target_type"] != "alphabet":
            continue
        lines.append(f"- {row['target_type']} {row['target_id']}: {row['display_label']} -> {row['data_dir']}/{row['data_label']}")
    lines.extend(["", "## Oraciones/intenciones LSM", ""])
    for domain in domains:
        lines.extend(["", f"### {domain}", ""])
        for row in sentence_rows:
            if row["domain"] != domain:
                continue
            lines.append(
                f"- {row['target_id']}: {row['display_label']} -> {row['data_dir']}/{row['data_label']}"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}