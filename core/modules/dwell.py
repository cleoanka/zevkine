"""
dwell.py — Dwell Time (sahnede kalma süresi) modülü.

Her track ID'nin sahnede ilk göründüğü zamanı kaydeder; her frame'de
geçen süreyi hesaplar. Eşik (saniye) aşan objeler highlight edilir.
"""

from __future__ import annotations

import time

from ..detection import Detection


class DwellModule:
    def __init__(self, threshold_seconds: float = 5.0) -> None:
        self.threshold_seconds = threshold_seconds
        # track_id -> ilk görülme zamanı (monotonic)
        self._first_seen: dict[int, float] = {}
        self._last_seen: dict[int, float] = {}
        self.last_error: str | None = None

    def configure(self, threshold_seconds: float | None = None) -> None:
        if threshold_seconds is not None:
            self.threshold_seconds = float(threshold_seconds)

    def update(self, detections: list[Detection], frame_shape: tuple[int, int]) -> None:
        try:
            now = time.monotonic()
            for det in detections:
                if det.track_id is None:
                    continue
                tid = det.track_id
                if tid not in self._first_seen:
                    self._first_seen[tid] = now
                self._last_seen[tid] = now
                dwell = now - self._first_seen[tid]
                det.dwell_seconds = dwell
                det.is_dwelling = dwell >= self.threshold_seconds

            # 3 sn'dir görünmeyen track'leri unut (yeniden gelirse sıfırlanır)
            stale = [tid for tid, ts in self._last_seen.items() if now - ts > 3.0]
            for tid in stale:
                self._first_seen.pop(tid, None)
                self._last_seen.pop(tid, None)
        except Exception as exc:
            self.last_error = str(exc)

    def get_state(self) -> dict:
        return {
            "threshold_seconds": self.threshold_seconds,
            "active_tracks": len(self._first_seen),
        }
