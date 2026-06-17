"""
recorder.py — Video kaydı, snapshot ve log export.

  * VideoRecorder: annotated (üzerine çizim yapılmış) frame'leri MP4'e yazar.
  * save_snapshot: anlık frame'i timestamp adıyla PNG kaydeder.
  * export_log: detection event'lerini JSON veya CSV olarak yazar.

Tüm çıktılar config.export_dir altına gider.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class VideoRecorder:
    """Thread-safe MP4 kaydedici. Annotated frame'leri yazar."""

    def __init__(self, export_dir: Path) -> None:
        self.export_dir = export_dir
        self._writer: Optional[cv2.VideoWriter] = None
        self._lock = threading.Lock()
        self.recording = False
        self.current_path: Optional[Path] = None
        self._fps = 30.0
        self._size: Optional[tuple[int, int]] = None

    def start(self, frame: np.ndarray, fps: float = 30.0) -> Path:
        """Verilen ilk frame'in boyutuna göre yazıcıyı başlatır."""
        with self._lock:
            if self.recording:
                return self.current_path  # zaten kayıtta
            h, w = frame.shape[:2]
            self._size = (w, h)
            self._fps = max(fps, 1.0)
            path = self.export_dir / f"recording_{_timestamp()}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(path), fourcc, self._fps, self._size
            )
            self.recording = True
            self.current_path = path
            return path

    def write(self, frame: np.ndarray) -> None:
        with self._lock:
            if self.recording and self._writer is not None:
                # Boyut sabit olmalı; gerekirse yeniden boyutlandır
                if self._size and (frame.shape[1], frame.shape[0]) != self._size:
                    frame = cv2.resize(frame, self._size)
                self._writer.write(frame)

    def stop(self) -> Optional[Path]:
        with self._lock:
            if self._writer is not None:
                self._writer.release()
            self._writer = None
            self.recording = False
            path = self.current_path
            self.current_path = None
            return path

    def get_state(self) -> dict:
        return {
            "recording": self.recording,
            "path": str(self.current_path) if self.current_path else None,
        }


def save_snapshot(frame: np.ndarray, export_dir: Path) -> Path:
    """Anlık frame'i PNG olarak kaydeder, yolunu döndürür."""
    path = export_dir / f"snapshot_{_timestamp()}.png"
    cv2.imwrite(str(path), frame)
    return path


def export_log(
    events: list[dict], export_dir: Path, fmt: str = "json"
) -> Path:
    """Detection event listesini JSON ya da CSV olarak yazar."""
    ts = _timestamp()
    if fmt == "csv":
        path = export_dir / f"detections_{ts}.csv"
        if events:
            keys = sorted({k for e in events for k in e.keys()})
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(events)
        else:
            path.write_text("", encoding="utf-8")
    else:
        path = export_dir / f"detections_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, ensure_ascii=False)
    return path
