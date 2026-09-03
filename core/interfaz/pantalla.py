from __future__ import annotations

import cv2
import numpy as np

from core.base.mensajes import (
    LOADING_MODEL_TEXT,
    LOW_CONFIDENCE_TEXT,
    LOW_QUALITY_TEXT,
    NO_REAL_MODEL_TEXT,
    SHOW_HAND_TEXT,
    WARMING_SEQUENCE_TEXT,
    display_phrase,
)
from core.interfaz.guia_practica import PRACTICE_GESTURES, gesture_by_word, render_reference_card


GREEN = (58, 188, 94)
BLUE = (255, 90, 20)  # BGR azul vivo - fix bolita invisible
WHITE = (252, 248, 246)
SHADOW = (38, 31, 26)
CYAN = (204, 176, 44)
INK = (47, 39, 34)
MUTED = (163, 148, 139)
SURFACE_DARK = (44, 37, 33)
WARNING = (42, 163, 245)


def landmark_xy(point, width: int, height: int) -> tuple[int, int]:
    return int(np.clip(point.x * width, 0, width - 1)), int(np.clip(point.y * height, 0, height - 1))


def hand_box(results, width: int, height: int) -> tuple[int, int, int, int] | None:
    """hand box."""
    points: list[tuple[int, int]] = []
    for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
        if hand is None:
            continue
        for point in hand.landmark:
            points.append(landmark_xy(point, width, height))
    if not points:
        return None
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    pad = 22
    return max(0, min(xs) - pad), max(0, min(ys) - pad), min(width - 1, max(xs) + pad), min(height - 1, max(ys) + pad)


def face_box(results, width: int, height: int) -> tuple[int, int, int, int] | None:
    if results.face_landmarks is None:
        return None
    points = [landmark_xy(point, width, height) for point in results.face_landmarks.landmark]
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    pad = 8
    return max(0, min(xs) - pad), max(0, min(ys) - pad), min(width - 1, max(xs) + pad), min(height - 1, max(ys) + pad)


def draw_hand_overlay(frame: np.ndarray, hand_landmarks) -> None:
    if hand_landmarks is None:
        return

    import mediapipe as mp

    height, width = frame.shape[:2]
    HAND_CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
    for start, end in HAND_CONNECTIONS:
        ax, ay = landmark_xy(hand_landmarks.landmark[start], width, height)
        bx, by = landmark_xy(hand_landmarks.landmark[end], width, height)
        cv2.line(frame, (ax, ay), (bx, by), SHADOW, 3, cv2.LINE_AA)
        cv2.line(frame, (ax, ay), (bx, by), WHITE, 1, cv2.LINE_AA)

    for point in hand_landmarks.landmark:
        x, y = landmark_xy(point, width, height)
        # FIX bolita azul invisible: r=3 muy pequeña, ahora con borde contrastado SHADOW+WHITE+BLUE
        cv2.circle(frame, (x, y), 7, SHADOW, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 6, WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 4, BLUE, -1, cv2.LINE_AA)


def draw_text(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int], scale: float, thickness: int) -> None:
    x, y = origin
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, SHADOW, max(1, thickness), cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def blend_rect(frame: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int], color: tuple[int, int, int], alpha: float) -> None:
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


def fitted_scale(text: str, max_width: int, base_scale: float, thickness: int, min_scale: float = 0.38) -> float:
    scale = base_scale
    while scale > min_scale:
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if text_width <= max_width:
            return scale
        scale -= 0.04
    return min_scale


def draw_chip(
    frame: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    color: tuple[int, int, int] = BLUE,
    max_width: int | None = None,
    scale: float = 0.55,
    thickness: int = 1,
) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    x, y = xy
    pad_x = 12
    pad_y = 8
    max_text_width = max(60, (max_width or width) - pad_x * 2)
    actual_scale = fitted_scale(text, max_text_width, scale, thickness)
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, actual_scale, thickness)
    chip_w = text_w + pad_x * 2
    chip_h = text_h + pad_y * 2
    x = int(np.clip(x, 8, max(8, width - chip_w - 8)))
    y = int(np.clip(y, 8, max(8, height - chip_h - 8)))
    cv2.rectangle(frame, (x + 2, y + 2), (x + chip_w + 2, y + chip_h + 2), SHADOW, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + chip_w, y + chip_h), color, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        text,
        (x + pad_x, y + pad_y + text_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        actual_scale,
        WHITE,
        thickness,
        cv2.LINE_AA,
    )
    return x, y, x + chip_w, y + chip_h


def draw_focus_box(frame: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int] = BLUE) -> None:
    x1, y1, x2, y2 = box
    length = max(18, min(38, (x2 - x1) // 4, (y2 - y1) // 4))
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
        cv2.line(frame, start, end, SHADOW, 3, cv2.LINE_AA)
        cv2.line(frame, start, end, color, 1, cv2.LINE_AA)


INTERNAL_STATUS_TEXTS = {
    display_phrase(NO_REAL_MODEL_TEXT),
    display_phrase(LOW_CONFIDENCE_TEXT),
    display_phrase(LOW_QUALITY_TEXT),
    display_phrase(LOADING_MODEL_TEXT),
    display_phrase(WARMING_SEQUENCE_TEXT),
    display_phrase(SHOW_HAND_TEXT),
}


def should_draw_translation(text: str) -> bool:
    display_text = display_phrase(text)
    return bool(display_text and display_text not in INTERNAL_STATUS_TEXTS)


def draw_translation_label(frame: np.ndarray, text: str, box: tuple[int, int, int, int]) -> None:
    height, width = frame.shape[:2]
    display_text = display_phrase(text)
    if not should_draw_translation(display_text):
        return

    max_width = min(360, width - 24)
    scale = fitted_scale(display_text, max_width, 0.66, 1, min_scale=0.42)
    text_w, text_h = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
    x = int(np.clip(box[0], 8, max(8, width - text_w - 8)))
    y = box[1] - 14 if box[1] > text_h + 24 else box[3] + text_h + 14
    y = int(np.clip(y, text_h + 8, height - 8))
    draw_text(frame, display_text, (x, y), BLUE, scale, 1)


def model_mode_text(trained: bool, allow_rule_demo: bool) -> tuple[str, tuple[int, int, int]]:
    if trained:
        return "modelo entrenado", GREEN
    if allow_rule_demo:
        return "demo reglas - no LSM real", WARNING
    return "sin modelo LSM real", MUTED


def draw_interface(
    frame: np.ndarray,
    results,
    translation: str,
    fingers: int | None,
    trained: bool,
    *,
    allow_rule_demo: bool = False,
) -> None:
    height, width = frame.shape[:2]
    box = hand_box(results, width, height)
    if box is not None:
        draw_translation_label(frame, translation, box)


def build_practice_cards(word: str) -> list[np.ndarray]:
    if word.strip().lower() in {"", "auto", "rotar", "rotate"}:
        gestures = PRACTICE_GESTURES
    else:
        gestures = [gesture_by_word(word)]
    return [render_reference_card(gesture, width=360, height=260) for gesture in gestures]


def compose_practice_view(frame: np.ndarray, cards: list[np.ndarray], elapsed_seconds: float, rotate_seconds: float) -> np.ndarray:
    if not cards:
        return frame
    interval = max(1.0, rotate_seconds)
    card = cards[int(elapsed_seconds // interval) % len(cards)]
    height, width = frame.shape[:2]
    panel_width = max(300, min(390, width // 3))
    panel = np.full((height, panel_width, 3), (235, 240, 247), dtype=np.uint8)
    blend_rect(panel, (0, 0), (panel_width, height), (252, 253, 255), 0.7)
    cv2.rectangle(panel, (0, 0), (1, height), (199, 210, 224), 1, cv2.LINE_AA)
    draw_text(panel, "guia visual", (18, 34), INK, 0.58, 1)
    draw_text(panel, "demo, no LSM certificado", (18, 62), MUTED, 0.43, 1)
    target_w = panel_width - 32
    target_h = int(target_w * card.shape[0] / card.shape[1])
    max_card_h = max(72, height - 118)
    if target_h > max_card_h:
        target_h = max_card_h
        target_w = int(target_h * card.shape[1] / card.shape[0])
    target_w = max(80, min(panel_width - 32, target_w))
    target_h = max(60, min(max_card_h, target_h))
    resized = cv2.resize(card, (target_w, target_h), interpolation=cv2.INTER_AREA)
    y = min(max(74, (height - resized.shape[0]) // 2), max(8, height - resized.shape[0] - 44))
    x = max(16, (panel_width - resized.shape[1]) // 2)
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    progress = (elapsed_seconds % interval) / interval
    bar_x = 18
    bar_y = min(height - 30, y + resized.shape[0] + 22)
    bar_w = panel_width - 36
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + 5), (207, 216, 229), -1, cv2.LINE_AA)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 5), BLUE, -1, cv2.LINE_AA)
    return np.hstack([frame, panel])