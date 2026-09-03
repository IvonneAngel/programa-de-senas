from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from core.base.rutas import ruta_entrenamiento
from core.datos.cobertura_lsm import safe_label
from core.datos.manifiesto_objetivos import TargetSpec, iter_alphabet_specs, iter_sentence_specs, resolve_target
from core.datos.revisar_entrenamiento import normalize_label


ImportTargetType = Literal["word", "alphabet", "sentence"]


@dataclass(frozen=True, slots=True)
class LabelResolution:
    source_label: str
    normalized_source: str
    target_type: ImportTargetType
    status: str
    data_label: str | None = None
    target_id: str | None = None
    display_label: str | None = None
    reason: str | None = None
    mapping_source: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def _source_key(value: str) -> str:
    return safe_label(value)


def is_ambiguous_numeric_label(value: str | None) -> bool:
    key = _source_key(value or "")
    return bool(key and key.replace("_", "").isdigit())


def _sentence_text(spec: TargetSpec) -> str:
    prefix = f"{spec.target_id}: "
    if spec.display_label.startswith(prefix):
        return spec.display_label[len(prefix) :]
    return spec.display_label


def _add_alias(alias_map: dict[str, TargetSpec | None], alias: str, spec: TargetSpec) -> None:
    key = _source_key(alias)
    if not key:
        return
    existing = alias_map.get(key)
    if existing is None and key in alias_map:
        return
    if existing is not None and existing.target_id != spec.target_id:
        alias_map[key] = None
        return
    alias_map[key] = spec


def alphabet_alias_map() -> dict[str, TargetSpec | None]:
    aliases: dict[str, TargetSpec | None] = {}
    for spec in iter_alphabet_specs():
        letter = spec.display_label.replace("Letra ", "")
        for alias in (
            spec.target_id,
            spec.data_label,
            letter,
            f"letra_{letter}",
            f"letra {letter}",
            f"letter_{letter}",
            f"letter {letter}",
            f"lsm_{letter}",
        ):
            _add_alias(aliases, alias, spec)
    return aliases


def sentence_alias_map() -> dict[str, TargetSpec | None]:
    aliases: dict[str, TargetSpec | None] = {}
    for spec in iter_sentence_specs():
        _add_alias(aliases, spec.target_id, spec)
        _add_alias(aliases, _sentence_text(spec), spec)
    return aliases


def default_data_dir_for_import(target_type: ImportTargetType) -> str:
    root = ruta_entrenamiento()
    if target_type == "alphabet":
        return str(root / "abecedario")
    if target_type == "sentence":
        return str(root / "palabras y oraciones" / "oraciones")
    return str(root / "palabras y oraciones" / "otros")


def load_label_map(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    map_path = Path(path)
    if not map_path.exists():
        raise FileNotFoundError(f"Label map not found: {map_path}")

    if map_path.suffix.lower() == ".json":
        payload = json.loads(map_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON label map must be an object: {source_label: target_id}.")
        return {_source_key(str(source)): str(target) for source, target in payload.items() if _source_key(str(source))}

    rows: dict[str, str] = {}
    with map_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("CSV label map must include headers.")
        source_field = "source_label" if "source_label" in reader.fieldnames else "source"
        target_field = "target_id" if "target_id" in reader.fieldnames else "target"
        if source_field not in reader.fieldnames or target_field not in reader.fieldnames:
            raise ValueError("CSV label map needs source_label,target_id columns.")
        for row in reader:
            source = _source_key(str(row.get(source_field, "")))
            target = str(row.get(target_field, "")).strip()
            if source and target:
                rows[source] = target
    return rows


def _resolution_from_spec(
    source_label: str,
    normalized_source: str,
    target_type: ImportTargetType,
    spec: TargetSpec,
    *,
    mapping_source: str,
) -> LabelResolution:
    return LabelResolution(
        source_label=source_label,
        normalized_source=normalized_source,
        target_type=target_type,
        status="ready",
        data_label=spec.data_label,
        target_id=spec.target_id,
        display_label=spec.display_label,
        mapping_source=mapping_source,
    )


def _resolve_explicit_target(
    source_label: str,
    normalized_source: str,
    target_type: ImportTargetType,
    explicit_target: str,
) -> LabelResolution:
    if target_type == "word":
        label = normalize_label(explicit_target)
        if not label:
            return LabelResolution(
                source_label=source_label,
                normalized_source=normalized_source,
                target_type=target_type,
                status="unresolved",
                reason="explicit_word_label_empty",
                mapping_source="label_map",
            )
        return LabelResolution(
            source_label=source_label,
            normalized_source=normalized_source,
            target_type=target_type,
            status="ready",
            data_label=label,
            target_id=label,
            display_label=label.replace("_", " "),
            mapping_source="label_map",
        )

    try:
        spec = resolve_target(target_type, explicit_target)
    except ValueError as exc:
        return LabelResolution(
            source_label=source_label,
            normalized_source=normalized_source,
            target_type=target_type,
            status="unresolved",
            reason=f"label_map_target_out_of_contract:{exc}",
            mapping_source="label_map",
        )
    return _resolution_from_spec(source_label, normalized_source, target_type, spec, mapping_source="label_map")


def resolve_import_label(
    source_label: str | None,
    target_type: ImportTargetType,
    *,
    explicit_label_map: dict[str, str] | None = None,
) -> LabelResolution:
    raw = (source_label or "").strip()
    normalized_source = _source_key(raw)
    if not normalized_source:
        return LabelResolution(
            source_label=raw,
            normalized_source=normalized_source,
            target_type=target_type,
            status="unresolved",
            reason="empty_or_ambiguous_label",
        )

    explicit_map = explicit_label_map or {}
    if normalized_source in explicit_map:
        return _resolve_explicit_target(raw, normalized_source, target_type, explicit_map[normalized_source])

    if target_type == "word":
        if is_ambiguous_numeric_label(raw):
            return LabelResolution(
                source_label=raw,
                normalized_source=normalized_source,
                target_type=target_type,
                status="unresolved",
                reason="word_label_numeric_ambiguous",
            )
        label = normalize_label(raw)
        return LabelResolution(
            source_label=raw,
            normalized_source=normalized_source,
            target_type=target_type,
            status="ready",
            data_label=label,
            target_id=label,
            display_label=label.replace("_", " "),
            mapping_source="word_label",
        )

    aliases = alphabet_alias_map() if target_type == "alphabet" else sentence_alias_map()
    spec = aliases.get(normalized_source)
    if spec is None and normalized_source in aliases:
        return LabelResolution(
            source_label=raw,
            normalized_source=normalized_source,
            target_type=target_type,
            status="unresolved",
            reason="auto_alias_ambiguous",
            mapping_source="auto_alias",
        )
    if spec is not None:
        return _resolution_from_spec(raw, normalized_source, target_type, spec, mapping_source="auto_alias")

    reason = "alphabet_label_out_of_contract" if target_type == "alphabet" else "sentence_label_map_required"
    return LabelResolution(
        source_label=raw,
        normalized_source=normalized_source,
        target_type=target_type,
        status="unresolved",
        reason=reason,
    )