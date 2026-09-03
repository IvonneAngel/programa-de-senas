from __future__ import annotations

import json
import re
from collections import Counter
from types import SimpleNamespace
from pathlib import Path
from time import strftime
from typing import Any, Callable, Literal

import numpy as np

from core.base.configuracion import FRAME_COUNT, KEYPOINT_SIZE
from core.datos.conjunto_datos import validate_real_sequence
from core.vision.puntos import (
    RICH_KEYPOINT_SIZE,
    extract_hand_keypoints,
    extract_normalized_rich_keypoints,
    extract_rich_keypoints,
)
from core.vision.recursos_mediapipe import configure_mediapipe_resources, prepare_mediapipe_import_path
from core.datos.mapa_etiquetas import ImportTargetType, resolve_import_label
from core.datos.revisar_entrenamiento import normalize_label


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GENERIC_LABEL_PARTS = {
    "abecedario_images",
    "grabaciones",
    "dataset",
    "external_datasets",
    "images",
    "palabras_images",
    "preview_abecedario",
    "preview_palabras",
    "test",
    "todas_las_imagenes",
    "train",
    "valid",
    "validation",
    "zenodo_lsm",
}

FeatureSet = Literal["hands", "rich", "rich_normalized"]
SequenceBuilder = Callable[[list[Path], int, int], np.ndarray | None]


def keypoint_size_for_feature_set(feature_set: FeatureSet) -> int:
    if feature_set == "hands":
        return KEYPOINT_SIZE
    if feature_set in {"rich", "rich_normalized"}:
        return RICH_KEYPOINT_SIZE
    raise ValueError(f"Feature set no soportado: {feature_set!r}")


def image_files_in(path: Path) -> list[Path]:
    return sorted(file_path for file_path in path.iterdir() if file_path.suffix.lower() in IMAGE_EXTENSIONS and file_path.is_file())


def read_image(path: str | Path) -> np.ndarray | None:
    file_path = Path(path)
    data = np.fromfile(file_path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2_imdecode(data)


def cv2_imdecode(data: np.ndarray) -> np.ndarray | None:
    import cv2

    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def derive_label(relative_parts: tuple[str, ...]) -> str | None:
    for part in reversed(relative_parts):
        normalized = normalize_label(part)
        if normalized and normalized not in GENERIC_LABEL_PARTS:
            return normalized
    return None


def find_image_sequence_folders(dataset_dir: str | Path, *, min_images: int = 5) -> list[dict[str, Any]]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    rows: list[dict[str, Any]] = []
    for folder in sorted(path for path in root.rglob("*") if path.is_dir()):
        images = image_files_in(folder)
        if len(images) < min_images:
            continue
        relative_parts = folder.relative_to(root).parts
        if not relative_parts:
            continue
        label = derive_label(relative_parts)
        rows.append(
            {
                "folder": str(folder),
                "relative_folder": "/".join(relative_parts),
                "default_label": label,
                "label_status": "ready" if label else "ambiguous",
                "image_count": len(images),
            }
        )
    return rows


def sample_image_files(files: list[Path], frame_count: int) -> list[Path]:
    if not files:
        raise ValueError("No image files provided")
    if len(files) == frame_count:
        return files
    indices = np.linspace(0, len(files) - 1, frame_count).round().astype(int)
    return [files[int(index)] for index in indices]


def resample_sequence_frames(frames: list[np.ndarray], frame_count: int, keypoint_size: int) -> np.ndarray:
    if not frames:
        raise ValueError("No keypoint frames provided")
    source = np.stack([np.asarray(frame, dtype=np.float32).reshape(keypoint_size) for frame in frames])
    if source.shape[0] == frame_count:
        return source
    indices = np.linspace(0, source.shape[0] - 1, frame_count).round().astype(int)
    return source[indices].astype(np.float32)


def stable_sequence_path(output_root: str | Path, label: str, relative_folder: str) -> Path:
    safe_parts = [
        re.sub(r"[^a-zA-Z0-9_-]+", "_", part).strip("_")
        for part in Path(relative_folder).parts
        if part.strip()
    ]
    safe_name = "__".join(part for part in safe_parts if part) or "secuencia"
    return Path(output_root) / label / f"{safe_name}.npy"


def save_stable_sequence(output_root: str | Path, label: str, relative_folder: str, sequence: np.ndarray) -> Path:
    path = stable_sequence_path(output_root, label, relative_folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(sequence, dtype=np.float32))
    return path


def output_root_for_label(
    output_root: str | Path,
    label: str,
    target_type: ImportTargetType,
    word_category_map: dict[str, str] | None = None,
) -> Path:
    root = Path(output_root)
    if target_type == "word" and word_category_map is not None:
        category = word_category_map.get(label, "otros")
        return root / category
    return root


class MediapipeSequenceBuilder:
    def __init__(self, *, detect_frames: int = 0, feature_set: FeatureSet = "hands") -> None:
        prepare_mediapipe_import_path()
        import cv2
        import mediapipe as mp

        configure_mediapipe_resources(mp)
        self.cv2 = cv2
        self.detect_frames = detect_frames
        self.feature_set = feature_set
                # Tasks Hand+Pose (migrado)
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        hand_model = str(Path(__file__).resolve().parents[2] / "app" / "assets" / "models" / "hand_landmarker.task")
        pose_model = str(Path(__file__).resolve().parents[2] / "app" / "assets" / "models" / "pose_landmarker_lite.task")
        self.hand_landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=hand_model), running_mode=vision.RunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.30, min_tracking_confidence=0.30))
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=pose_model), running_mode=vision.RunningMode.IMAGE, num_poses=1, min_pose_detection_confidence=0.30, min_tracking_confidence=0.30))
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.35,
        )

    def _detect(self, rgb: np.ndarray) -> Any:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_result = self.hand_landmarker.detect(mp_image)
        pose_result = self.pose_landmarker.detect(mp_image)
        results = type('obj', (), {'hand_landmarks': hand_result.hand_landmarks, 'pose_landmarks': pose_result.pose_landmarks})()
        if results.left_hand_landmarks is not None or results.right_hand_landmarks is not None:
            return results

        hand_results = self.hands.process(rgb)
        hands = list(hand_results.multi_hand_landmarks or [])
        if not hands:
            return results
        return SimpleNamespace(
            left_hand_landmarks=hands[1] if len(hands) > 1 else None,
            right_hand_landmarks=hands[0],
            pose_landmarks=None,
            face_landmarks=None,
        )

    def __call__(self, files: list[Path], frame_count: int, keypoint_size: int) -> np.ndarray | None:
        selected = sample_image_files(files, self.detect_frames or frame_count)
        frames: list[np.ndarray] = []
        detected = 0
        for image_path in selected:
            image = read_image(image_path)
            if image is None:
                continue
            rgb = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
            results = self._detect(rgb)
            if results.left_hand_landmarks is not None or results.right_hand_landmarks is not None:
                detected += 1
            if self.feature_set == "rich":
                extractor = extract_rich_keypoints
            elif self.feature_set == "rich_normalized":
                extractor = extract_normalized_rich_keypoints
            else:
                extractor = extract_hand_keypoints
            frames.append(extractor(results))

        if len(frames) != len(selected):
            return None
        if detected < max(3, len(selected) // 5):
            return None
        return validate_real_sequence(resample_sequence_frames(frames, frame_count, keypoint_size), frame_count, keypoint_size)

    def close(self) -> None:
        self.hand_landmarker.close()
        self.pose_landmarker.close()
        self.hands.close()


def mediapipe_sequence_builder(files: list[Path], frame_count: int, keypoint_size: int) -> np.ndarray | None:
    builder = MediapipeSequenceBuilder()
    try:
        return builder(files, frame_count, keypoint_size)
    finally:
        builder.close()


def import_image_sequence_dataset(
    dataset_dir: str | Path,
    output_data_dir: str | Path,
    *,
    target_type: ImportTargetType = "word",
    explicit_label_map: dict[str, str] | None = None,
    frame_count: int = FRAME_COUNT,
    keypoint_size: int | None = None,
    feature_set: FeatureSet = "hands",
    min_images: int = 5,
    limit_per_label: int = 0,
    max_sequences: int = 0,
    detect_frames: int = 0,
    resume: bool = False,
    dry_run: bool = False,
    sequence_builder: SequenceBuilder | None = None,
    word_category_map: dict[str, str] | None = None,
    allowed_labels: set[str] | None = None,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    output_root = Path(output_data_dir)
    effective_keypoint_size = keypoint_size or keypoint_size_for_feature_set(feature_set)
    folders = find_image_sequence_folders(root, min_images=min_images)
    imported_counts: Counter[str] = Counter()
    skipped: list[dict[str, Any]] = []
    imported: list[dict[str, Any]] = []
    new_sequence_count = 0
    already_imported_sequence_count = 0
    stopped_reason = None
    builder = sequence_builder if sequence_builder is not None else (None if dry_run else MediapipeSequenceBuilder(detect_frames=detect_frames, feature_set=feature_set))

    try:
        for row in folders:
            resolution = resolve_import_label(
                row.get("default_label"),
                target_type,
                explicit_label_map=explicit_label_map,
            )
            if resolution.status != "ready" or not resolution.data_label:
                skipped.append({**row, "reason": resolution.reason or "label_unresolved"})
                skipped[-1]["label_resolution"] = resolution.to_dict()
                continue

            label = resolution.data_label
            if allowed_labels is not None and label not in allowed_labels:
                skipped.append({**row, "reason": "label_not_requested"})
                skipped[-1]["label_resolution"] = resolution.to_dict()
                continue

            if limit_per_label > 0 and imported_counts[label] >= limit_per_label:
                skipped.append({**row, "reason": "label_limit_reached"})
                skipped[-1]["label_resolution"] = resolution.to_dict()
                continue

            label_output_root = output_root_for_label(output_root, label, target_type, word_category_map)
            planned_path = stable_sequence_path(label_output_root, label, str(row["relative_folder"]))
            if resume and planned_path.exists():
                imported_counts[label] += 1
                already_imported_sequence_count += 1
                imported.append(
                    {
                        **row,
                        "label": label,
                        "target_id": resolution.target_id,
                        "label_resolution": resolution.to_dict(),
                        "sequence_path": str(planned_path),
                        "status": "already_imported",
                    }
                )
                continue

            if max_sequences > 0 and new_sequence_count >= max_sequences:
                stopped_reason = "max_sequences_reached"
                break

            folder = Path(str(row["folder"]))
            images = image_files_in(folder)
            if dry_run:
                imported_counts[label] += 1
                new_sequence_count += 1
                imported.append(
                    {
                        **row,
                        "label": label,
                        "target_id": resolution.target_id,
                        "label_resolution": resolution.to_dict(),
                        "sequence_path": None,
                        "status": "dry_run",
                    }
                )
                continue

            if builder is None:
                sequence = None
            else:
                sequence = builder(images, frame_count, effective_keypoint_size)
            if sequence is not None:
                try:
                    sequence = validate_real_sequence(sequence, frame_count, effective_keypoint_size)
                except ValueError:
                    sequence = None
            if sequence is None:
                skipped.append({**row, "reason": "hand_detection_failed"})
                skipped[-1]["label_resolution"] = resolution.to_dict()
                continue

            saved_path = save_stable_sequence(label_output_root, label, str(row["relative_folder"]), sequence)
            imported_counts[label] += 1
            new_sequence_count += 1
            imported.append(
                {
                    **row,
                    "label": label,
                    "target_id": resolution.target_id,
                    "label_resolution": resolution.to_dict(),
                    "sequence_path": str(saved_path),
                    "status": "imported",
                }
            )
    finally:
        closer = getattr(builder, "close", None) if builder is not None else None
        if callable(closer):
            closer()

    return {
        "generated_at": strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_dir": str(root),
        "output_data_dir": str(output_root),
        "target_type": target_type,
        "explicit_label_map_count": len(explicit_label_map or {}),
        "frame_count": frame_count,
        "keypoint_size": effective_keypoint_size,
        "feature_set": feature_set,
        "allowed_label_count": len(allowed_labels) if allowed_labels is not None else 0,
        "dry_run": dry_run,
        "max_sequences": max_sequences,
        "detect_frames": detect_frames,
        "resume": resume,
        "stopped_reason": stopped_reason,
        "source_sequence_folders": len(folders),
        "imported_sequence_count": len(imported),
        "new_sequence_count": new_sequence_count,
        "already_imported_sequence_count": already_imported_sequence_count,
        "skipped_sequence_count": len(skipped),
        "imported_counts": dict(sorted(imported_counts.items())),
        "imported_preview": imported[:30],
        "skipped_preview": skipped[:30],
    }


def save_import_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "lsm_image_dataset_import.json"
    md_path = output_path / "lsm_image_dataset_import.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    lines = [
        "# Importacion de Dataset LSM por Imagenes",
        "",
        f"Dataset: `{report['dataset_dir']}`",
        f"Salida: `{report['output_data_dir']}`",
        f"Tipo de objetivo: `{report.get('target_type', 'word')}`",
        f"Mapeos explicitos: {report.get('explicit_label_map_count', 0)}",
        f"Dry run: {'si' if report['dry_run'] else 'no'}",
        f"Carpetas de secuencia detectadas: {report['source_sequence_folders']}",
        f"Secuencias importadas: {report['imported_sequence_count']}",
        f"Secuencias omitidas: {report['skipped_sequence_count']}",
        "",
        "## Conteo por etiqueta",
        "",
    ]
    for label, count in report["imported_counts"].items():
        lines.append(f"- {label}: {count}")
    if not report["imported_counts"]:
        lines.append("- Ninguna secuencia importada.")
    lines.extend(
        [
            "",
            "## Nota",
            "",
            "Esto solo convierte datasets reales ya descargados/extrados. No inventa señas ni usa imagenes de referencia como entrenamiento.",
            "Para `alphabet` y `sentence`, las etiquetas deben resolver al contrato LSM; lo ambiguo se omite.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}