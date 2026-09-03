"""Exporta landmarks crudos de un clip local para paridad offline sin imágenes ni etiquetas."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


def natural_frames(source_dir: Path) -> list[Path]:
    return sorted(
        (path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() == ".jpg"),
        key=lambda path: int("".join(character for character in path.stem if character.isdigit()) or "0"),
    )


def sampled_indices(count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("El clip no contiene frames")
    return np.rint(np.linspace(0, count - 1, num=30)).astype(np.int64)


def encode_points(points: list[object], dimensions: int) -> str:
    values: list[str] = []
    for point in points:
        values.extend(format(float(getattr(point, axis)), ".9g") for axis in ("x", "y", "z"))
        if dimensions == 4:
            values.append(format(float(getattr(point, "visibility")), ".9g"))
    expected = 21 * 3 if dimensions == 3 else 33 * 4
    if len(values) != expected:
        raise ValueError(f"Se esperaban {expected} valores, se recibieron {len(values)}")
    return ",".join(values)


def empty_values(count: int) -> str:
    return ",".join("0" for _ in range(count))


def best_hands(result: object) -> dict[str, list[object]]:
    output: dict[str, tuple[float, list[object]]] = {}
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        if len(landmarks) != 21 or not handedness:
            continue
        category = handedness[0]
        side = str(category.category_name).lower().strip()
        if side not in {"left", "right"}:
            continue
        if side not in output or float(category.score) > output[side][0]:
            output[side] = (float(category.score), landmarks)
    return {side: item[1] for side, item in output.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.hand_model.is_file() or not args.pose_model.is_file():
        raise FileNotFoundError("No se encontraron modelos locales de landmarks")
    frames = natural_frames(args.source_dir)
    if not frames:
        raise FileNotFoundError("No hay frames JPEG en el directorio indicado")
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(args.hand_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.30,
        min_hand_presence_confidence=0.30,
        min_tracking_confidence=0.50,
    ))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(args.pose_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.30,
        min_pose_presence_confidence=0.30,
        min_tracking_confidence=0.50,
    ))
    lines: list[str] = []
    for index in sampled_indices(len(frames)):
        image_bgr = cv2.imread(str(frames[int(index)]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"No se pudo leer {frames[int(index)]}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        hands = best_hands(hand.detect(mp_image))
        poses = pose.detect(mp_image).pose_landmarks
        right = hands.get("right")
        left = hands.get("left")
        pose_points = poses[0] if poses and len(poses[0]) == 33 else None
        lines.append("|".join((
            "1" if right else "0",
            "1" if left else "0",
            "1" if pose_points else "0",
            encode_points(right, 3) if right else empty_values(63),
            encode_points(left, 3) if left else empty_values(63),
            encode_points(pose_points, 4) if pose_points else empty_values(132),
        )))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"frames": len(lines), "output": str(args.out), "contains_images": False, "contains_labels": False})


if __name__ == "__main__":
    main()