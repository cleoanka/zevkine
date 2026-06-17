"""
anomaly.py — Basit anomali tespiti modülü.

Son N saniyedeki obje sayısının hareketli ortalamasını tutar. Anlık obje
sayısı bu ortalamanın `multiplier` katını aşarsa anomaly bayrağı kalkar ve
dashboard'da bir banner gösterilir.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

from ..detection import Detection


class AnomalyModule:
    def __init__(
        self, window_seconds: float = 30.0, multiplier: float = 2.0
    ) -> None:
        self.window_seconds = window_seconds
        self.multiplier = multiplier
        # (zaman, sayı) örnekleri
        self._samples: deque = deque()
        self.is_anomaly = False
        self.current_count = 0
        self.baseline = 0.0
        self.last_error: Optional[str] = None

    def configure(
        self,
        window_seconds: Optional[float] = None,
        multiplier: Optional[float] = None,
    ) -> None:
        if window_seconds is not None:
            self.window_seconds = float(window_seconds)
        if multiplier is not None:
            self.multiplier = float(multiplier)

    def update(
        self, detections: list[Detection], frame_shape: tuple[int, int]
    ) -> None:
        try:
            now = time.monotonic()
            count = len(detections)
            self.current_count = count
            self._samples.append((now, count))

            # Pencere dışındaki örnekleri at
            cutoff = now - self.window_seconds
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()

            if len(self._samples) >= 2:
                counts = [c for _, c in self._samples]
                self.baseline = sum(counts) / len(counts)
                # Ortalama çok düşükse gürültüden kaçın (en az 1 referans)
                threshold = max(self.baseline * self.multiplier, 1.0)
                self.is_anomaly = count > threshold and self.baseline >= 1.0
            else:
                self.baseline = float(count)
                self.is_anomaly = False
        except Exception as exc:
            self.last_error = str(exc)

    def get_state(self) -> dict:
        return {
            "is_anomaly": self.is_anomaly,
            "current_count": self.current_count,
            "baseline": round(self.baseline, 1),
            "multiplier": self.multiplier,
            "window_seconds": self.window_seconds,
        }
