"""Construye un manifiesto externo por participante desde Zenodo 18330565."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from pathlib import PurePosixPath


MINIMUM_ARCHIVE_BYTES = 2_674_414_824
VIDEO_PATTERN = re.compile(r"^.+_(?P<participant>\d+)$")


def video_paths(archive: Path) -> list[str]:
    result = subprocess.run(["7z", "l", "-slt", str(archive)], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"No se puede listar 7z: {result.stderr[-500:]}")
    paths = [line.removeprefix("Path = ") for line in result.stdout.splitlines() if line.startswith("Path = ")]
    return [path for path in paths if path.lower().endswith((".mp4", ".mov")) and "/._" not in path and not Path(path).name.startswith("._")]


def row_from_path(internal_path: str) -> dict[str, str]:
    archive_path = PurePosixPath(internal_path)
    stem = archive_path.stem
    match = VIDEO_PATTERN.match(stem)
    if not match:
        raise ValueError(f"Nombre sin glosa_participante: {internal_path}")
    label, participant = archive_path.parent.name, int(match.group("participant"))
    if not label:
        raise ValueError(f"Glosa vacía: {internal_path}")
    if not 0 <= participant <= 11:
        raise ValueError(f"Participante fuera de 0–11: {internal_path}")
    participant_id = f"S{participant + 1:02d}"
    split = "train" if participant <= 7 else "validation" if participant <= 9 else "test"
    return {"sample_id": f"dynamic121_{label}_{participant_id}", "label_lsm": label, "participant_id": participant_id, "split_external": split, "video_internal_path": internal_path, "source_filename_stem": stem, "feature_status": "pending", "feature_path": "", "feature_error": ""}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True); parser.add_argument("--out-manifest", type=Path, required=True); parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()
    if not args.archive.is_file() or args.archive.stat().st_size < MINIMUM_ARCHIVE_BYTES:
        raise ValueError("Archivo Zenodo ausente o incompleto")
    rows = [row_from_path(path) for path in video_paths(args.archive)]
    if len(rows) != len({row["sample_id"] for row in rows}):
        raise ValueError("Videos duplicados por glosa/participante")
    participants = sorted({row["participant_id"] for row in rows})
    labels = sorted({row["label_lsm"] for row in rows})
    counts = Counter(row["split_external"] for row in rows)
    if participants != [f"S{index:02d}" for index in range(1, 13)] or len(labels) != 121 or any(counts[split] == 0 for split in ("train", "validation", "test")):
        raise ValueError(f"Inventario Zenodo inesperado: participantes={participants}, glosas={len(labels)}, splits={dict(counts)}")
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.out_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(sorted(rows, key=lambda row: (row["participant_id"], row["label_lsm"])))
    report = {"kind": "zenodo_dynamic121_manifest", "videos": len(rows), "expected_full_grid": 121 * 12, "missing_gloss_participant_slots": 121 * 12 - len(rows), "labels": len(labels), "participants": participants, "splits": dict(counts), "archive_only": True, "benchmark_210_words_touched": False, "s08_read": False, "s09_read": False}
    args.out_report.parent.mkdir(parents=True, exist_ok=True); args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__": main()