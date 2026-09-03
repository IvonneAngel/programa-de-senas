from __future__ import annotations

import json
import time
from pathlib import Path
from time import strftime
from typing import Any


class RuntimeEventLogger:
    def __init__(
        self,
        root_dir: str | Path,
        *,
        app_name: str = "traductor_de_senas",
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.root_dir = Path(root_dir)
        self.app_name = app_name
        self.session_id = strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.root_dir / f"session_{self.session_id}"
        self.events_path = self.session_dir / "events.jsonl"
        self.manifest_path = self.session_dir / "manifest.json"
        self._started = False
        self._last_translation = ""
        self._start_monotonic = 0.0

    def start(self, metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled or self._started:
            return
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._start_monotonic = time.monotonic()
        manifest = {
            "session_id": self.session_id,
            "app_name": self.app_name,
            "started_at": strftime("%Y-%m-%d %H:%M:%S"),
            "events_path": str(self.events_path),
            "metadata": metadata or {},
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
        self._started = True
        self.write_event("session_start", {"metadata": metadata or {}})

    def attach_video(self, video_path: str | Path | None) -> None:
        if not self.enabled or not self._started or video_path is None:
            return
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["video_path"] = str(video_path)
        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
        self.write_event("video_attached", {"video_path": str(video_path)})

    def write_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if not self.enabled or not self._started:
            return
        record = {
            "timestamp": strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(time.monotonic() - self._start_monotonic, 3),
            "event_type": event_type,
            "payload": payload,
        }
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")

    def log_translation(
        self,
        translation: str,
        *,
        fingers: int | None,
        source: str,
        confidence: float | None = None,
    ) -> None:
        if not self.enabled or translation == self._last_translation:
            return
        self._last_translation = translation
        payload: dict[str, Any] = {
            "translation": translation,
            "fingers": fingers,
            "source": source,
        }
        if confidence is not None:
            payload["confidence"] = round(float(confidence), 4)
        self.write_event("translation_change", payload)

    def finish(self) -> None:
        if not self.enabled or not self._started:
            return
        self.write_event("session_end", {})
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["finished_at"] = strftime("%Y-%m-%d %H:%M:%S")
        manifest["duration_seconds"] = round(time.monotonic() - self._start_monotonic, 3)
        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
        self._started = False


def list_sessions(root_dir: str | Path, limit: int = 10) -> list[dict[str, Any]]:
    """list sessions."""
    root = Path(root_dir)
    if not root.exists():
        return []
    manifests = sorted(root.glob("session_*/manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    sessions: list[dict[str, Any]] = []
    for manifest_path in manifests[:limit]:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        events_path = Path(manifest.get("events_path", ""))
        event_count = 0
        if events_path.exists():
            with events_path.open("r", encoding="utf-8") as fh:
                event_count = sum(1 for _ in fh)
        sessions.append(
            {
                "session_id": manifest.get("session_id", manifest_path.parent.name),
                "manifest_path": str(manifest_path),
                "events_path": str(events_path) if events_path else "",
                "video_path": manifest.get("video_path"),
                "started_at": manifest.get("started_at"),
                "finished_at": manifest.get("finished_at"),
                "duration_seconds": manifest.get("duration_seconds"),
                "event_count": event_count,
            }
        )
    return sessions