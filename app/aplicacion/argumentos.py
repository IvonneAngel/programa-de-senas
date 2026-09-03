from __future__ import annotations

import argparse

from core.base.configuracion import CONFIDENCE_THRESHOLD, FRAME_COUNT
from core.vision.puntos import RICH_KEYPOINT_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=FRAME_COUNT)
    parser.add_argument("--keypoint-size", type=int, default=RICH_KEYPOINT_SIZE)
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--min-confidence-margin", type=float, default=0.12)
    parser.add_argument("--recognition-mode", choices=["auto", "word", "alphabet", "sentence"], default="auto")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--record-process", action="store_true")
    parser.add_argument("--record-seconds", type=int, default=90)
    parser.add_argument("--record-fps", type=float, default=10.0)
    parser.add_argument("--record-scale", type=float, default=0.5)
    parser.add_argument("--stability-window", type=int, default=7)
    parser.add_argument("--stability-votes", type=int, default=4)
    parser.add_argument("--max-runtime-seconds", type=float, default=0.0)
    parser.add_argument("--no-log-events", action="store_true")
    parser.add_argument("--demo-mode", action="store_true")
    parser.add_argument("--camera-optional", action="store_true")
    parser.add_argument(
        "--allow-rule-demo",
        action="store_true",
    )
    parser.add_argument("--practice-guide", action="store_true")
    parser.add_argument("--practice-word", default="Te quiero")
    parser.add_argument("--practice-rotate-seconds", type=float, default=5.0)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()
