from __future__ import annotations

import math
from collections import deque

import numpy as np

from core.interfaz.pantalla import face_box
from core.vision.reglas_gestos import classify_base_gesture, has_horizontal_wave


def point_distance(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

def angle_degrees(a, b, c) -> float:
    ab = np.array([a.x - b.x, a.y - b.y, a.z - b.z], dtype=np.float32)
    cb = np.array([c.x - b.x, c.y - b.y, c.z - b.z], dtype=np.float32)
    denom = float(np.linalg.norm(ab) * np.linalg.norm(cb))
    if denom <= 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(ab, cb) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))

def finger_states(hand_landmarks) -> dict[str, bool]:
    points = hand_landmarks.landmark
    wrist = points[0]

    states = {
        "index": angle_degrees(points[5], points[6], points[8]) > 150 and point_distance(points[8], wrist) > point_distance(points[6], wrist),
        "middle": angle_degrees(points[9], points[10], points[12]) > 150 and point_distance(points[12], wrist) > point_distance(points[10], wrist),
        "ring": angle_degrees(points[13], points[14], points[16]) > 150 and point_distance(points[16], wrist) > point_distance(points[14], wrist),
        "pinky": angle_degrees(points[17], points[18], points[20]) > 150 and point_distance(points[20], wrist) > point_distance(points[18], wrist),
        "thumb": angle_degrees(points[2], points[3], points[4]) > 145 and point_distance(points[4], wrist) > point_distance(points[3], wrist),
    }
    return states

def hand_center_pixels(hand_landmarks, width: int, height: int) -> tuple[int, int]:
    points = hand_landmarks.landmark
    x = int(np.mean([point.x for point in points]) * width)
    y = int(np.mean([point.y for point in points]) * height)
    return int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))

def near_face(hand_landmarks, results, width: int, height: int) -> bool:
    face = face_box(results, width, height)
    if face is None:
        return False

    cx, cy = hand_center_pixels(hand_landmarks, width, height)
    face_height = max(1, face[3] - face[1])
    face_width = max(1, face[2] - face[0])
    x_min = face[0] - int(face_width * 0.55)
    x_max = face[2] + int(face_width * 0.55)
    y_min = face[1] + int(face_height * 0.30)
    y_max = face[3] + int(face_height * 0.45)
    return x_min <= cx <= x_max and y_min <= cy <= y_max

def basic_translation(
    states: dict[str, bool],
    hand_landmarks,
    results,
    width: int,
    height: int,
    center_history: deque[tuple[float, float]],
) -> str:
    return classify_base_gesture(
        states,
        near_face=near_face(hand_landmarks, results, width, height),
        waving=has_horizontal_wave(center_history),
    )

def choose_primary_hand(results):
    return results.right_hand_landmarks or results.left_hand_landmarks

def resolve_runtime_mode(camera_opened: bool, *, demo_mode: bool, camera_optional: bool) -> str:
    if demo_mode:
        return "demo"
    if camera_opened:
        return "camera"
    if camera_optional:
        return "demo"
    raise RuntimeError("No pude abrir la camara.")
