from __future__ import annotations

from collections.abc import Sequence

from core.base.mensajes import SHOW_HAND_TEXT


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def count_fingers(states: dict[str, bool]) -> int:
    return sum(1 for name in FINGER_NAMES if states.get(name, False))


def only(states: dict[str, bool], *extended_names: str) -> bool:
    wanted = set(extended_names)
    return all(bool(states.get(name, False)) == (name in wanted) for name in FINGER_NAMES)


def has_horizontal_wave(history: Sequence[tuple[float, float]]) -> bool:
    if len(history) < 10:
        return False
    xs = [point[0] for point in history]
    ys = [point[1] for point in history]
    return (max(xs) - min(xs)) > 0.08 and (max(ys) - min(ys)) < 0.18


def classify_base_gesture(
    states: dict[str, bool],
    *,
    near_face: bool = False,
    waving: bool = False,
) -> str:
    fingers = count_fingers(states)

    if only(states, "thumb", "index", "pinky"):
        return "TE QUIERO"
    if only(states, "thumb"):
        return "SI"
    if only(states, "index"):
        return "NO"
    if only(states, "thumb", "index", "middle"):
        return "AYUDA"

    if fingers >= 4 and near_face:
        return "GRACIAS"
    if fingers >= 4 and waving:
        return "HOLA"
    if fingers >= 4:
        return "HOLA"

    return {
        0: "NADA",
        1: "UNO",
        2: "DOS",
        3: "TRES",
        4: "CUATRO",
    }.get(fingers, SHOW_HAND_TEXT)