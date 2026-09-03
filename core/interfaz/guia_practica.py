from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.base.mensajes import display_phrase


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
BLUE = (226, 101, 37)
WHITE = (252, 248, 246)
INK = (47, 39, 34)
MUTED = (145, 126, 115)
SURFACE = (252, 249, 247)
SURFACE_DARK = (44, 37, 33)
TEAL = (15, 118, 110)

PRACTICE_GESTURES: list[dict[str, Any]] = [
    {
        "word": "TE QUIERO",
        "extended": ("thumb", "index", "pinky"),
        "hint": "Pulgar, indice y menique arriba",
        "context": "base",
    },
    {
        "word": "HOLA",
        "extended": ("thumb", "index", "middle", "ring", "pinky"),
        "hint": "Mano abierta y movimiento corto",
        "context": "mover",
    },
    {
        "word": "GRACIAS",
        "extended": ("thumb", "index", "middle", "ring", "pinky"),
        "hint": "Mano abierta cerca del rostro",
        "context": "rostro",
    },
    {
        "word": "SI",
        "extended": ("thumb",),
        "hint": "Solo pulgar extendido",
        "context": "base",
    },
    {
        "word": "NO",
        "extended": ("index",),
        "hint": "Solo indice extendido",
        "context": "base",
    },
    {
        "word": "AYUDA",
        "extended": ("thumb", "index", "middle"),
        "hint": "Pulgar, indice y medio",
        "context": "base",
    },
    {
        "word": "NADA",
        "extended": (),
        "hint": "Mano cerrada",
        "context": "base",
    },
    {
        "word": "UNO",
        "extended": ("index",),
        "hint": "Un dedo extendido",
        "context": "numero",
    },
    {
        "word": "DOS",
        "extended": ("index", "middle"),
        "hint": "Dos dedos extendidos",
        "context": "numero",
    },
    {
        "word": "TRES",
        "extended": ("thumb", "index", "middle"),
        "hint": "Tres dedos extendidos",
        "context": "numero",
    },
    {
        "word": "CUATRO",
        "extended": ("index", "middle", "ring", "pinky"),
        "hint": "Cuatro dedos extendidos",
        "context": "numero",
    },
]


def slugify_word(word: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", word.lower()).strip("_")
    return slug or "gesto"


def gesture_by_word(word: str) -> dict[str, Any]:
    normalized = word.strip().upper()
    for gesture in PRACTICE_GESTURES:
        if gesture["word"] == normalized:
            return gesture
    raise KeyError(f"No practice gesture configured for {word!r}")


def _finger_segments(
    name: str,
    extended: bool,
    palm_center: tuple[int, int],
    scale: float,
) -> list[tuple[int, int]]:
    cx, cy = palm_center
    finger_offsets = {
        "thumb": (-70, -6),
        "index": (-34, -72),
        "middle": (0, -84),
        "ring": (34, -72),
        "pinky": (65, -55),
    }
    base_dx, base_dy = finger_offsets[name]
    base = (int(cx + base_dx * scale), int(cy + base_dy * scale))

    if name == "thumb":
        if extended:
            return [
                (int(cx - 48 * scale), int(cy + 18 * scale)),
                (int(cx - 82 * scale), int(cy - 18 * scale)),
                (int(cx - 112 * scale), int(cy - 42 * scale)),
            ]
        return [
            (int(cx - 42 * scale), int(cy + 18 * scale)),
            (int(cx - 70 * scale), int(cy + 12 * scale)),
            (int(cx - 56 * scale), int(cy - 18 * scale)),
        ]

    if extended:
        length = {
            "index": 115,
            "middle": 132,
            "ring": 112,
            "pinky": 92,
        }[name]
        tip = (base[0], int(base[1] - length * scale))
        mid = (base[0], int((base[1] + tip[1]) / 2))
        return [base, mid, tip]

    bend = {
        "index": (-28, -20),
        "middle": (-12, -22),
        "ring": (16, -18),
        "pinky": (30, -12),
    }[name]
    mid = (int(base[0] + bend[0] * scale), int(base[1] + bend[1] * scale))
    tip = (int(mid[0] + bend[0] * 0.65 * scale), int(mid[1] + 45 * scale))
    return [base, mid, tip]


def _draw_polyline(image: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int], thickness: int) -> None:
    for start, end in zip(points, points[1:]):
        cv2.line(image, start, end, color, thickness, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, point, max(3, thickness // 2), (245, 248, 252), -1, cv2.LINE_AA)
        cv2.circle(image, point, max(2, thickness // 3), color, -1, cv2.LINE_AA)


def _fit_scale(text: str, max_width: int, base_scale: float, thickness: int, min_scale: float = 0.34) -> float:
    scale = base_scale
    while scale > min_scale:
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if text_width <= max_width:
            return scale
        scale -= 0.04
    return min_scale


def _put_fitted_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    max_width: int,
    *,
    base_scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    scale = _fit_scale(text, max_width, base_scale, thickness)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_reference_card(gesture: dict[str, Any], width: int = 420, height: int = 300) -> np.ndarray:
    card = np.full((height, width, 3), SURFACE, dtype=np.uint8)
    cv2.rectangle(card, (0, 0), (width - 1, height - 1), (203, 212, 226), 1, cv2.LINE_AA)
    cv2.rectangle(card, (0, 0), (width, 50), SURFACE_DARK, -1, cv2.LINE_AA)

    _put_fitted_text(card, display_phrase(str(gesture["word"])), (18, 33), max(40, width - 150), base_scale=0.78, color=WHITE, thickness=2)

    badge_text = str(gesture["context"]).upper()
    badge_scale = _fit_scale(badge_text, max(36, width // 3), 0.42, 1)
    badge_size = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, badge_scale, 1)[0]
    badge_x = width - badge_size[0] - 28
    cv2.rectangle(card, (badge_x - 8, 15), (width - 14, 36), TEAL, -1, cv2.LINE_AA)
    cv2.putText(card, badge_text, (badge_x - 2, 31), cv2.FONT_HERSHEY_SIMPLEX, badge_scale, WHITE, 1, cv2.LINE_AA)

    scale = min(width / 420, height / 300)
    palm_center = (width // 2, int(height * 0.67))
    palm_radius = int(58 * scale)
    cv2.ellipse(card, palm_center, (palm_radius, int(palm_radius * 0.9)), 0, 0, 360, (226, 232, 240), -1, cv2.LINE_AA)
    cv2.ellipse(card, palm_center, (palm_radius, int(palm_radius * 0.9)), 0, 0, 360, MUTED, 2, cv2.LINE_AA)
    cv2.rectangle(
        card,
        (int(palm_center[0] - 35 * scale), int(palm_center[1] + 42 * scale)),
        (int(palm_center[0] + 36 * scale), int(palm_center[1] + 85 * scale)),
        (226, 232, 240),
        -1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        card,
        (int(palm_center[0] - 35 * scale), int(palm_center[1] + 42 * scale)),
        (int(palm_center[0] + 36 * scale), int(palm_center[1] + 85 * scale)),
        MUTED,
        2,
        cv2.LINE_AA,
    )

    extended = set(gesture["extended"])
    for finger in FINGER_NAMES:
        points = _finger_segments(finger, finger in extended, palm_center, scale)
        color = BLUE if finger in extended else MUTED
        _draw_polyline(card, points, color, max(5, int(9 * scale)))

    if gesture["context"] == "mover":
        y = int(height * 0.33)
        cv2.arrowedLine(card, (int(width * 0.28), y), (int(width * 0.72), y), TEAL, 3, cv2.LINE_AA, tipLength=0.08)
        cv2.arrowedLine(card, (int(width * 0.72), y + 18), (int(width * 0.28), y + 18), TEAL, 3, cv2.LINE_AA, tipLength=0.08)
    elif gesture["context"] == "rostro":
        cv2.circle(card, (int(width * 0.22), int(height * 0.45)), int(34 * scale), (235, 238, 244), 2, cv2.LINE_AA)
        cv2.putText(card, "cara", (int(width * 0.16), int(height * 0.45) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, MUTED, 1, cv2.LINE_AA)

    hint = str(gesture["hint"])
    cv2.rectangle(card, (12, height - 44), (width - 12, height - 12), (247, 240, 235), -1, cv2.LINE_AA)
    _put_fitted_text(card, hint, (20, height - 22), max(40, width - 40), base_scale=0.46, color=INK, thickness=1)

    return card


def render_contact_sheet(cards: list[np.ndarray], columns: int = 3, gap: int = 16) -> np.ndarray:
    if not cards:
        raise ValueError("At least one card is required")
    height, width = cards[0].shape[:2]
    rows = int(np.ceil(len(cards) / columns))
    sheet_h = rows * height + (rows + 1) * gap
    sheet_w = columns * width + (columns + 1) * gap
    sheet = np.full((sheet_h, sheet_w, 3), (238, 242, 247), dtype=np.uint8)

    for index, card in enumerate(cards):
        row = index // columns
        col = index % columns
        y = gap + row * (height + gap)
        x = gap + col * (width + gap)
        sheet[y : y + height, x : x + width] = card
    return sheet


def save_practice_guide(output_dir: str | Path, preview_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    preview_path = Path(preview_dir)
    card_dir = preview_path / "guia_gestos"
    output_path.mkdir(parents=True, exist_ok=True)
    card_dir.mkdir(parents=True, exist_ok=True)

    cards = []
    card_records = []
    for gesture in PRACTICE_GESTURES:
        card = render_reference_card(gesture)
        cards.append(card)
        card_path = card_dir / f"{slugify_word(gesture['word'])}.png"
        cv2.imwrite(str(card_path), card)
        card_records.append(
            {
                "word": gesture["word"],
                "hint": gesture["hint"],
                "context": gesture["context"],
                "image": str(card_path),
            }
        )

    sheet = render_contact_sheet(cards)
    sheet_path = preview_path / "guia_gestos_base.png"
    cv2.imwrite(str(sheet_path), sheet)

    markdown_path = output_path / "guia_gestos.md"
    lines = [
        "# Guia Visual de Gestos Base",
        "",
        "Esta guia es para practicar sin abrir camara. Son imagenes PNG generadas por el proyecto.",
        "",
        "Advertencia: esta guia es demo heuristica, no certifica Lengua de Senas Mexicana real.",
        "Para reconocimiento real se requiere modelo entrenado con datos LSM validados.",
        "",
        f"Imagen general: `{sheet_path}`",
        "",
    ]
    for record in card_records:
        lines.extend(
            [
                f"## {record['word']}",
                "",
                f"- Como formarla: {record['hint']}",
                f"- Contexto: {record['context']}",
                f"- Imagen: `{record['image']}`",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "sheet": str(sheet_path),
        "markdown": str(markdown_path),
        "cards": card_records,
        "word_count": len(card_records),
    }