from __future__ import annotations

import json
from pathlib import Path
from time import strftime
from typing import Any

import cv2
import numpy as np

from core.base.mensajes import display_phrase
from core.interfaz.guia_practica import PRACTICE_GESTURES, render_reference_card


DEMO_WORDS = ("TE QUIERO", "HOLA", "GRACIAS", "SI", "NO", "AYUDA", "NADA")

BLUE = (226, 101, 37)
WHITE = (252, 248, 246)
INK = (47, 39, 34)
MUTED = (163, 148, 139)
SURFACE_DARK = (44, 37, 33)
WARNING = (42, 163, 245)


def _draw_text(
    frame: np.ndarray,
    text: str,
    xy: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    x, y = xy
    if color not in {INK, MUTED}:
        cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (38, 31, 26), max(1, thickness), cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _blend_rect(frame: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int], color: tuple[int, int, int], alpha: float) -> None:
    height, width = frame.shape[:2]
    x1 = int(np.clip(top_left[0], 0, width - 1))
    y1 = int(np.clip(top_left[1], 0, height - 1))
    x2 = int(np.clip(bottom_right[0], 0, width))
    y2 = int(np.clip(bottom_right[1], 0, height))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def _fitted_scale(text: str, max_width: int, base_scale: float, thickness: int, min_scale: float = 0.34) -> float:
    scale = base_scale
    while scale > min_scale:
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if text_width <= max_width:
            return scale
        scale -= 0.04
    return min_scale


def _draw_chip(
    frame: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    color: tuple[int, int, int] = BLUE,
    max_width: int | None = None,
    scale: float = 0.55,
) -> None:
    height, width = frame.shape[:2]
    x, y = xy
    pad_x = 10
    pad_y = 7
    text_scale = _fitted_scale(text, max(44, (max_width or width) - pad_x * 2), scale, 1)
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)
    chip_w = text_w + pad_x * 2
    chip_h = text_h + pad_y * 2
    x = int(np.clip(x, 8, max(8, width - chip_w - 8)))
    y = int(np.clip(y, 8, max(8, height - chip_h - 8)))
    cv2.rectangle(frame, (x + 2, y + 2), (x + chip_w + 2, y + chip_h + 2), (38, 31, 26), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + chip_w, y + chip_h), color, -1, cv2.LINE_AA)
    cv2.putText(frame, text, (x + pad_x, y + pad_y + text_h), cv2.FONT_HERSHEY_SIMPLEX, text_scale, WHITE, 1, cv2.LINE_AA)


def _draw_corner_box(frame: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box
    length = max(12, min(30, (x2 - x1) // 4, (y2 - y1) // 4))
    for start, end in [
        ((x1, y1), (x1 + length, y1)),
        ((x1, y1), (x1, y1 + length)),
        ((x2, y1), (x2 - length, y1)),
        ((x2, y1), (x2, y1 + length)),
        ((x1, y2), (x1 + length, y2)),
        ((x1, y2), (x1, y2 - length)),
        ((x2, y2), (x2 - length, y2)),
        ((x2, y2), (x2, y2 - length)),
    ]:
        cv2.line(frame, start, end, (38, 31, 26), 3, cv2.LINE_AA)
        cv2.line(frame, start, end, color, 1, cv2.LINE_AA)


def _practice_gestures_for_demo() -> list[dict[str, Any]]:
    wanted = set(DEMO_WORDS)
    return [gesture for gesture in PRACTICE_GESTURES if gesture["word"] in wanted]


def render_demo_frame(
    gesture: dict[str, Any],
    *,
    width: int = 960,
    height: int = 540,
    progress: float = 0.0,
) -> np.ndarray:
    frame = np.full((height, width, 3), (249, 245, 241), dtype=np.uint8)
    camera_w = int(width * 0.64)
    panel_x = camera_w + 16

    cv2.rectangle(frame, (18, 18), (camera_w, height - 18), (246, 241, 236), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (18, 18), (camera_w, height - 18), (176, 161, 150), 1, cv2.LINE_AA)
    _blend_rect(frame, (18, 18), (camera_w, 58), SURFACE_DARK, 0.76)
    _draw_text(frame, "Traductor LSM", (34, 45), 0.58, WHITE, 1)

    word = display_phrase(str(gesture["word"]))
    _draw_chip(frame, "demo reglas - no LSM real", (camera_w - 250, 29), color=WARNING, max_width=230, scale=0.43)
    _draw_text(frame, "sin camara", (34, height - 34), 0.47, MUTED, 1)

    card_width = max(120, min(300, camera_w - 110))
    card_height = max(90, min(220, height - 145))
    card = render_reference_card(gesture, width=card_width, height=card_height)
    card_x = int(camera_w * 0.5) - card.shape[1] // 2
    card_y = int(height * 0.40) - card.shape[0] // 2
    frame[card_y : card_y + card.shape[0], card_x : card_x + card.shape[1]] = card

    cx = card_x + card.shape[1] // 2
    cy = card_y + card.shape[0] // 2
    wobble = int(np.sin(progress * np.pi * 2) * 12)
    hand_box = (cx - 142 + wobble, cy - 106, cx + 142 + wobble, cy + 106)
    _draw_corner_box(frame, hand_box, BLUE)
    _draw_chip(frame, word, (hand_box[0], max(8, hand_box[1] - 36)), color=BLUE, max_width=220, scale=0.56)

    cv2.rectangle(frame, (panel_x, 18), (width - 18, height - 18), (255, 253, 250), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (panel_x, 18), (width - 18, height - 18), (216, 200, 190), 1, cv2.LINE_AA)
    _blend_rect(frame, (panel_x, 18), (width - 18, 66), SURFACE_DARK, 0.88)
    _draw_text(frame, "Vista de prueba", (panel_x + 18, 50), 0.64, WHITE, 1)

    panel_lines = [
        "Evidencia ligera",
        "Fuente: guia PNG",
        "Modo: demo heuristica",
        "Camara: no usada",
        "TensorFlow: pendiente multi-palabra",
        "",
        f"Palabra: {word}",
        str(gesture["hint"]),
    ]
    y = 102
    for line in panel_lines:
        if line:
            text_scale = _fitted_scale(line, max(40, width - panel_x - 44), 0.45, 1)
            _draw_text(frame, line, (panel_x + 18, y), text_scale, INK, 1)
        y += max(22, int(height * 0.06))

    bar_x = panel_x + 18
    bar_y = height - 66
    bar_w = width - panel_x - 56
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (220, 228, 238), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 8), BLUE, -1, cv2.LINE_AA)
    return frame


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if writer.isOpened():
        return writer, path

    fallback = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(fallback), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if writer.isOpened():
        return writer, fallback
    raise RuntimeError(f"No se pudo crear video demo: {path}")


def generate_no_camera_demo(
    video_dir: str | Path,
    report_dir: str | Path,
    *,
    fps: float = 6.0,
    seconds_per_gesture: float = 0.85,
    width: int = 960,
    height: int = 540,
) -> dict[str, Any]:
    video_path = Path(video_dir)
    report_path = Path(report_dir)
    video_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    timestamp = strftime("%Y%m%d_%H%M%S")
    target = video_path / f"demo_sin_camara_{timestamp}.mp4"
    writer, actual_video = _open_writer(target, max(1.0, fps), (width, height))
    gestures = _practice_gestures_for_demo()
    frames_per_gesture = max(1, int(max(0.2, seconds_per_gesture) * max(1.0, fps)))

    try:
        for gesture in gestures:
            for frame_index in range(frames_per_gesture):
                progress = (frame_index + 1) / frames_per_gesture
                writer.write(render_demo_frame(gesture, width=width, height=height, progress=progress))
    finally:
        writer.release()

    metadata = {
        "generated_at": strftime("%Y-%m-%d %H:%M:%S"),
        "video": str(actual_video),
        "bytes": actual_video.stat().st_size,
        "megabytes": round(actual_video.stat().st_size / (1024 * 1024), 3),
        "fps": max(1.0, fps),
        "seconds_per_gesture": seconds_per_gesture,
        "size": [width, height],
        "words": [gesture["word"] for gesture in gestures],
        "note": "Demo visual sin camara; no es LSM real y no sustituye entrenamiento TensorFlow multi-palabra con datos validados.",
    }

    json_path = report_path / "demo_sin_camara.json"
    md_path = report_path / "demo_sin_camara.md"
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
    lines = [
        "# Demo Sin Camara",
        "",
        f"Generado: {metadata['generated_at']}",
        f"Video: `{metadata['video']}`",
        f"Tamano: {metadata['megabytes']} MB",
        f"FPS: {metadata['fps']}",
        "",
        "## Palabras demostradas",
        "",
    ]
    for word in metadata["words"]:
        lines.append(f"- {word}")
    lines.extend(["", "## Nota", "", metadata["note"], ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    metadata["json_report"] = str(json_path)
    metadata["markdown_report"] = str(md_path)
    return metadata