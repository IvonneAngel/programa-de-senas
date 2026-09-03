from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from aplicacion.camara import create_process_recorder
from core.sistema.eventos import RuntimeEventLogger


class ProcessRecorder:
    def __init__(self, args: argparse.Namespace, event_logger: RuntimeEventLogger):
        """  init  ."""
        self.args = args
        self.event_logger = event_logger
        self.writer = None
        self.video_path: Path | None = None
        self.video_size: tuple[int, int] | None = None
        self.started_at: float | None = None
        self.last_frame_at = 0.0

    """update."""
    def update(self, frame) -> None:
        if not self.args.record_process:
            return

        if self.writer is None and self.video_size is None:
            self.writer, self.video_path, self.video_size = create_process_recorder(frame, self.args)
            self.event_logger.attach_video(self.video_path)
            self.started_at = time.monotonic()

        if self.writer is None or self.video_size is None or self.started_at is None:
            return

        now = time.monotonic()
        if now - self.started_at > max(1, self.args.record_seconds):
            self.close()
            return

        interval = 1.0 / max(1.0, float(self.args.record_fps))
        if now - self.last_frame_at >= interval:
            self.writer.write(cv2.resize(frame, self.video_size, interpolation=cv2.INTER_AREA))
            self.last_frame_at = now

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None