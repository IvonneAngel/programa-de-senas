"""Crea un manifiesto del corpus Zenodo dinámico tras verificar su MD5."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


LETTERS = frozenset(("J", "K", "Ñ", "Q", "X", "Z"))


@dataclass(frozen=True)
class DynamicClip:
    archive_path: str
    participant_id: int
    letter_lsm: str
    view: str
    repetition: int
    split_original: str


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324: se verifica el checksum oficial publicado, no seguridad
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_clip_path(archive_path: str, split_by_participant: dict[int, str] | None = None, view_override: str | None = None) -> DynamicClip | None:
    normalized = archive_path.replace("\\", "/")
    path_parts = normalized.split("/")
    split_matches = [part.lower() for part in path_parts[:-1] if part.lower() in {"train", "test"}]
    if len(split_matches) > 1:
        return None
    name = normalized.rsplit("/", maxsplit=1)[-1]
    if "." not in name:
        return None
    stem, extension = name.rsplit(".", maxsplit=1)
    if extension.lower() not in {"mp4", "mov", "avi", "mkv"}:
        return None
    match = re.fullmatch(r"S(?P<subject>\d+)-(?P<letter>[JKÑQXZ])-(?P<tail>.+)", stem, flags=re.IGNORECASE)
    if match is None:
        return None
    tail = re.fullmatch(r"(?P<view>.+)[.-](?P<repetition>\d+)", match.group("tail"), flags=re.IGNORECASE)
    if tail is None:
        return None
    participant_id = int(match.group("subject"))
    letter = match.group("letter").upper()
    view = view_override or tail.group("view").strip()
    repeat = int(tail.group("repetition"))
    if letter not in LETTERS or not view or participant_id < 1 or repeat < 1:
        raise ValueError(f"Nombre de video fuera de protocolo: {archive_path}")
    if split_matches:
        split = split_matches[0]
    elif split_by_participant and participant_id in split_by_participant:
        split = split_by_participant[participant_id]
    else:
        return None
    return DynamicClip(normalized, participant_id, letter, view, repeat, split)


def archive_paths(archive: Path) -> list[str]:
    completed = subprocess.run(["7z", "l", "-slt", str(archive)], check=True, text=True, capture_output=True)
    return [line.removeprefix("Path = ") for line in completed.stdout.splitlines() if line.startswith("Path = ")]


def split_map_from_manifest(path: Path | None) -> dict[int, str] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"participant_id", "split_original"}.issubset(rows[0]):
        raise ValueError("El manifiesto de mapeo de split no tiene columnas requeridas")
    mapping: dict[int, str] = {}
    for row in rows:
        participant_id = int(row["participant_id"])
        split = row["split_original"].lower()
        if split not in {"train", "test"}:
            raise ValueError("Split de participante fuera de protocolo")
        previous = mapping.setdefault(participant_id, split)
        if previous != split:
            raise ValueError("Participante con split contradictorio en el manifiesto de mapeo")
    if len(mapping) != 20 or sum(split == "train" for split in mapping.values()) != 18 or sum(split == "test" for split in mapping.values()) != 2:
        raise ValueError("El mapeo externo debe preservar 20 participantes y split oficial 18/2")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--declared-clips", type=int, default=600)
    parser.add_argument("--split-map-manifest", type=Path, help="Mapa por participante para archivos que no contienen carpetas train/test")
    parser.add_argument("--view-label", help="Etiqueta canónica de cámara cuando el archivo mezcla nombres locales")
    args = parser.parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    actual_md5 = md5_file(args.archive)
    if actual_md5.lower() != args.expected_md5.lower():
        raise ValueError(f"MD5 inválido: {actual_md5}; esperado {args.expected_md5}")
    split_by_participant = split_map_from_manifest(args.split_map_manifest)
    clips = sorted((parsed for path in archive_paths(args.archive) if (parsed := parse_clip_path(path, split_by_participant, args.view_label)) is not None), key=lambda item: (item.participant_id, item.letter_lsm, item.view, item.repetition))
    duplicate_keys = {(clip.participant_id, clip.letter_lsm, clip.view.casefold(), clip.repetition) for clip in clips}
    participants = {clip.participant_id for clip in clips}
    classes = {clip.letter_lsm for clip in clips}
    if not clips or classes != LETTERS or len(participants) != 20:
        raise ValueError(f"Inventario ilegible: clips={len(clips)}, participantes={sorted(participants)}, clases={sorted(classes)}")
    rows = [{**asdict(clip), "split_original": clip.split_original, "source_archive": args.archive.name} for clip in clips]
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.out_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["archive_path", "participant_id", "letter_lsm", "view", "repetition", "split_original", "source_archive"])
        writer.writeheader()
        writer.writerows(rows)
    by_split = {split: sum(row["split_original"] == split for row in rows) for split in ("train", "test")}
    declared_repetitions_only = all(clip.repetition <= 5 for clip in clips)
    inventory_matches_declaration = len(rows) == args.declared_clips and len(duplicate_keys) == len(rows) and declared_repetitions_only
    report = {
        "kind": "zenodo_dynamic_lsm_archive_manifest",
        "archive": args.archive.name,
        "md5": actual_md5,
        "clips": len(rows),
        "declared_clips": args.declared_clips,
        "unique_participant_letter_view_repetition": len(duplicate_keys),
        "inventory_matches_declaration": inventory_matches_declaration,
        "repetition_range": [min(clip.repetition for clip in clips), max(clip.repetition for clip in clips)],
        "declared_repetitions_only": declared_repetitions_only,
        "classes": sorted(classes),
        "participants": sorted(participants),
        "splits_original": by_split,
        "videos_extracted": False,
        "landmarks_extracted": False,
        "eligible_for_extraction_or_training": inventory_matches_declaration,
        "metrics_evaluated": False,
        "benchmark_210_words_touched": False,
        "s08_metrics_evaluated": False,
        "s09_metrics_evaluated": False,
        "split_map_manifest": str(args.split_map_manifest) if args.split_map_manifest else None,
        "view_label_override": args.view_label,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()