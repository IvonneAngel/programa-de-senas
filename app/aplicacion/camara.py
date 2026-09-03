from __future__ import annotations

import argparse
from pathlib import Path
from time import strftime

import cv2
import numpy as np

from core.base.rutas import ruta_videos


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def read_frame(cap: cv2.VideoCapture, attempts: int = 60) -> np.ndarray | None:
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        cv2.waitKey(45)
    return None

def create_process_recorder(frame: np.ndarray, args: argparse.Namespace):
    if not args.record_process:
        return None, None, None

    output_dir = ruta_videos()
    output_dir.mkdir(parents=True, exist_ok=True)

    height, width = frame.shape[:2]
    scale = float(np.clip(args.record_scale, 0.25, 1.0))
    out_width = max(320, int(width * scale))
    out_height = max(180, int(height * scale))
    if out_width % 2:
        out_width += 1
    if out_height % 2:
        out_height += 1

    timestamp = strftime("%Y%m%d_%H%M%S")
    mp4_path = output_dir / f"traductor_proceso_{timestamp}.mp4"
    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, float(args.record_fps)),
        (out_width, out_height),
    )
    if writer.isOpened():
        print(f"video_proceso={mp4_path}", flush=True)
        return writer, mp4_path, (out_width, out_height)

    avi_path = output_dir / f"traductor_proceso_{timestamp}.avi"
    writer = cv2.VideoWriter(
        str(avi_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        max(1.0, float(args.record_fps)),
        (out_width, out_height),
    )
    if writer.isOpened():
        print(f"video_proceso={avi_path}", flush=True)
        return writer, avi_path, (out_width, out_height)

    print("video_proceso=no_disponible", flush=True)
    return None, None, None
