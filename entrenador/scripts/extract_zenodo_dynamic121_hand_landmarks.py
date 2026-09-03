"""Extrae landmarks de Zenodo 18330565 por video temporal y reanudable."""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as multiprocessing
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import mediapipe as mp
import numpy as np

from hand_landmark_common import FRAMES, LANDMARK_DIMENSIONS, extract_clip, feature_name


ARCHIVE_MIN_BYTES = 2_674_414_824
EXPECTED_VIDEOS, EXPECTED_LABELS = 1415, 121
ALLOWED_SUFFIXES = {".mp4", ".mov"}
_WORKER_HANDS: object | None = None


def valid_member(path: str) -> str:
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or parsed.suffix.lower() not in ALLOWED_SUFFIXES or len(parsed.parts) != 3:
        raise ValueError(f"Ruta Zenodo dinámica inválida: {path}")
    return path


def materialized_member_path(root: Path, member: str) -> Path:
    """Resuelve un miembro ya extraído sin permitir escapes del directorio temporal."""
    parsed = PurePosixPath(valid_member(member))
    candidate = root.joinpath(*parsed.parts)
    root_resolved, candidate_resolved = root.resolve(), candidate.resolve()
    if root_resolved not in candidate_resolved.parents:
        raise ValueError("Ruta materializada Zenodo fuera del directorio temporal")
    return candidate


@contextmanager
def temporary_video(archive: Path, member: str, temp_root: Path):
    valid_member(member); temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zenodo_dynamic121_", dir=temp_root) as directory:
        output = Path(directory) / f"clip{PurePosixPath(member).suffix.lower()}"
        with output.open("wb") as handle:
            completed = subprocess.run(["7z", "e", "-so", str(archive), member], stdout=handle, stderr=subprocess.PIPE, check=False)
        if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("No se pudo materializar video Zenodo temporal")
        yield output


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle: rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_VIDEOS or len({row["label_lsm"] for row in rows}) != EXPECTED_LABELS:
        raise ValueError("Manifiesto Zenodo dinámico incompleto")
    if {row["split_external"] for row in rows} != {"train", "validation", "test"}:
        raise ValueError("Split externo Zenodo inválido")
    for row in rows: valid_member(row["video_internal_path"])
    return rows


def reusable(row: dict[str, str], cache_root: Path) -> bool:
    candidate = cache_root / row.get("feature_path", "")
    if row.get("feature_status") not in {"ok", "insufficient_hand_evidence"} or not candidate.is_file(): return False
    values = np.load(candidate, allow_pickle=False)
    return values.shape == (FRAMES, LANDMARK_DIMENSIONS) and values.dtype == np.float32 and bool(np.isfinite(values).all())


def worker_init() -> None:
    """Inicializa una sola instancia MediaPipe por proceso para extracción por lotes."""
    global _WORKER_HANDS
    _WORKER_HANDS = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.50,
    )


def extract_materialized_row(task: tuple[dict[str, str], str, str]) -> dict[str, str]:
    """Extrae una fila desde un árbol temporal ya materializado; no conserva frames."""
    row, root_string, cache_string = task
    result = dict(row)
    result["feature_path"] = feature_name(row["video_internal_path"])
    try:
        video = materialized_member_path(Path(root_string), row["video_internal_path"])
        if not video.is_file() or video.stat().st_size == 0:
            raise FileNotFoundError("Video materializado Zenodo ausente")
        if _WORKER_HANDS is None:
            raise RuntimeError("Worker MediaPipe no inicializado")
        values, left, right, total, fps = extract_clip(video, _WORKER_HANDS)
        result.update({"feature_status": "ok" if left + right >= 5 else "insufficient_hand_evidence", "left_observed": str(left), "right_observed": str(right), "source_total_frames": str(total), "source_fps": f"{fps:.6f}", "feature_error": ""})
        np.save(Path(cache_string) / result["feature_path"], values, allow_pickle=False)
    except Exception as error:
        result.update({"feature_status": "error", "left_observed": "0", "right_observed": "0", "source_total_frames": "0", "source_fps": "0.000000", "feature_error": type(error).__name__})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--cache-root", type=Path, required=True); parser.add_argument("--out-manifest", type=Path, required=True); parser.add_argument("--out-report", type=Path, required=True); parser.add_argument("--temp-root", type=Path, required=True); parser.add_argument("--progress", type=Path, required=True); parser.add_argument("--resume", action="store_true"); parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--materialized-root", type=Path, help="Árbol temporal previamente extraído desde el archivador")
    parser.add_argument("--workers", type=int, default=1, help="Procesos de landmarks para --materialized-root")
    args = parser.parse_args()
    if not args.archive.is_file() or args.archive.stat().st_size < ARCHIVE_MIN_BYTES: raise ValueError("Archivo Zenodo dinámico ausente o incompleto")
    rows = load_rows(args.manifest); selected = rows[:args.limit] if args.limit else rows
    args.cache_root.mkdir(parents=True, exist_ok=True); args.progress.parent.mkdir(parents=True, exist_ok=True)
    prior: dict[str, dict[str, str]] = {}
    if args.resume and args.progress.is_file():
        for line in args.progress.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = {str(key): str(value) for key, value in json.loads(line).items()}; prior[entry["sample_id"]] = entry
    completed_rows, failures, resumed, pending_rows = [], [], 0, []
    for row in selected:
        old = prior.get(row["sample_id"])
        if old is not None and reusable(old, args.cache_root):
            completed_rows.append(old); resumed += 1
        else:
            pending_rows.append(row)

    def record(result: dict[str, str], progress: object) -> None:
        if result["feature_status"] == "error":
            failures.append({"sample_id": result["sample_id"], "error": result["feature_error"]})
        progress.write(json.dumps(result, ensure_ascii=False) + "\n"); progress.flush(); completed_rows.append(result)

    with args.progress.open("a", encoding="utf-8") as progress:
        if args.materialized_root is not None:
            if args.workers < 1:
                raise ValueError("--workers debe ser al menos 1")
            if not args.materialized_root.is_dir():
                raise ValueError("--materialized-root no existe o no es directorio")
            tasks = ((row, str(args.materialized_root), str(args.cache_root)) for row in pending_rows)
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=worker_init) as executor:
                for result in executor.map(extract_materialized_row, tasks):
                    record(result, progress)
        else:
            with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1, min_detection_confidence=0.50) as hands:
                for row in pending_rows:
                    result = dict(row); result["feature_path"] = feature_name(row["video_internal_path"])
                    try:
                        with temporary_video(args.archive, row["video_internal_path"], args.temp_root) as video:
                            values, left, right, total, fps = extract_clip(video, hands)
                        result.update({"feature_status": "ok" if left + right >= 5 else "insufficient_hand_evidence", "left_observed": str(left), "right_observed": str(right), "source_total_frames": str(total), "source_fps": f"{fps:.6f}", "feature_error": ""})
                        np.save(args.cache_root / result["feature_path"], values, allow_pickle=False)
                    except Exception as error:
                        result.update({"feature_status": "error", "left_observed": "0", "right_observed": "0", "source_total_frames": "0", "source_fps": "0.000000", "feature_error": type(error).__name__})
                    record(result, progress)
    fields = list(rows[0]) + ["left_observed", "right_observed", "source_total_frames", "source_fps"]
    for field in ("feature_path", "feature_status", "feature_error"):
        if field not in fields: fields.append(field)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.out_manifest.open("w", encoding="utf-8", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(completed_rows)
    report = {"kind": "zenodo_dynamic121_hand_landmarks", "rows": len(completed_rows), "ok": sum(row.get("feature_status") == "ok" for row in completed_rows), "insufficient_hand_evidence": sum(row.get("feature_status") == "insufficient_hand_evidence" for row in completed_rows), "errors": failures, "resumed": resumed, "split_counts": dict(Counter(row["split_external"] for row in completed_rows)), "feature_shape": [FRAMES, LANDMARK_DIMENSIONS], "video_extraction": "materialized_temporary_parallel" if args.materialized_root is not None else "one_member_temporary", "persistent_videos": False, "benchmark_210_words_touched": False, "s08_read": False, "s09_read": False}
    args.out_report.parent.mkdir(parents=True, exist_ok=True); args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__": main()