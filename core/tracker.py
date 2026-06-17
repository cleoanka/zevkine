"""
tracker.py — Track iz (trail) yöneticisi.

Kalıcı track ID'ler ByteTrack tarafından (inference.py içindeki model.track)
üretilir. Bu modül ise her track ID için son N merkez pozisyonunu tutar;
frontend bunları kuyruk (trail) olarak çizer.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from .detection import Detection


class TrackTrails:
    def __init__(self, trail_length: int = 30) -> None:
        self.trail_length = trail_length
        # track_id -> deque[(nx, ny)]  normalize pozisyonlar
        self._trails: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.trail_length)
        )
        self._last_seen: dict[int, float] = {}

    def configure(self, trail_length: int) -> None:
        if trail_length != self.trail_length:
            self.trail_length = trail_length
            # Yeni uzunlukla deque'leri yeniden oluştur
            new: dict[int, deque] = defaultdict(
                lambda: deque(maxlen=self.trail_length)
            )
            for tid, pts in self._trails.items():
                d = deque(pts, maxlen=self.trail_length)
                new[tid] = d
            self._trails = new

    def update(
        self, detections: list[Detection], frame_shape: tuple[int, int]
    ) -> None:
        h, w = frame_shape[:2]
        now = time.monotonic()
        for det in detections:
            if det.track_id is None:
                continue
            cx, cy = det.center
            self._trails[det.track_id].append((cx / max(w, 1), cy / max(h, 1)))
            self._last_seen[det.track_id] = now

        # Görünmeyen track izlerini temizle
        stale = [tid for tid, ts in self._last_seen.items() if now - ts > 2.0]
        for tid in stale:
            self._trails.pop(tid, None)
            self._last_seen.pop(tid, None)

    def get_trails(self) -> dict[int, list[list[float]]]:
        """Frontend için: {track_id: [[nx, ny], ...]}."""
        return {tid: [list(p) for p in pts] for tid, pts in self._trails.items()}

    def active_count(self) -> int:
        return len(self._trails)

    def reset(self) -> None:
        self._trails.clear()
        self._last_seen.clear()
