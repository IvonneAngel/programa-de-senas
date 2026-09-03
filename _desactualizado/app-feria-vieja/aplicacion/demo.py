from __future__ import annotations

import argparse
import time

import cv2

from aplicacion.constantes import WINDOW_HEIGHT, WINDOW_NAME, WINDOW_WIDTH
from aplicacion.grabacion import ProcessRecorder
from core.interfaz.demo_video import DEMO_WORDS, render_demo_frame
from core.interfaz.guia_practica import PRACTICE_GESTURES
from core.sistema.eventos import RuntimeEventLogger


def demo_gestures() -> list[dict]:
    wanted = set(DEMO_WORDS)
    return [gesture for gesture in PRACTICE_GESTURES if gesture["word"] in wanted]

def run_demo_loop(args: argparse.Namespace, model_ready: bool, labels: list[str], event_logger: RuntimeEventLogger) -> None:
    """run demo loop."""
    gestures = demo_gestures()
    if not gestures:
        raise RuntimeError("No hay gestos demo configurados.")

    if args.smoke_test:
        frame = render_demo_frame(gestures[0], width=960, height=540, progress=1.0)
        print("smoke-test-demo-ok", frame.shape, "model_ready", model_ready)
        return

    app_started_at = time.monotonic()
    recorder = ProcessRecorder(args, event_logger)
    event_logger.start(
        {
            "runtime_mode": "demo",
            "model_ready": model_ready,
            "labels": labels,
            "recognition_mode": args.recognition_mode,
            "record_process": args.record_process,
            "record_seconds": args.record_seconds,
            "record_fps": args.record_fps,
            "record_scale": args.record_scale,
            "max_runtime_seconds": args.max_runtime_seconds,
        }
    )

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
        while True:
            elapsed = time.monotonic() - app_started_at
            if args.max_runtime_seconds > 0 and elapsed >= args.max_runtime_seconds:
                break

            gesture_index = int(elapsed // max(0.5, args.practice_rotate_seconds)) % len(gestures)
            local_progress = (elapsed % max(0.5, args.practice_rotate_seconds)) / max(0.5, args.practice_rotate_seconds)
            gesture = gestures[gesture_index]
            display_frame = render_demo_frame(gesture, width=960, height=540, progress=local_progress)
            event_logger.log_translation(gesture["word"], fingers=None, source="demo")

            recorder.update(display_frame)
            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(33) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        recorder.close()
        event_logger.finish()
        cv2.destroyAllWindows()