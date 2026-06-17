"""
speed.py — Perspektif tabanlı hız tahmini modülü.

Kullanıcı canvas'ta iki yatay referans çizgisi sürükler ve aralarındaki
gerçek dünya mesafesini metre cinsinden girer. Track ID'ye bağlı olarak
frame-to-frame piksel deltası ölçülür ve metre/saniye'ye çevrilir.

UYARI: Bu kalibrasyonsuz/yarı-kalibre bir tahmindir. Düz, sabit bir
perspektif varsayar. Sonuçlar yaklaşıktır ve `calibrated` bayrağı ile
birlikte raporlanır.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from ..detection import Detection


class SpeedModule:
    def __init__(self, real_distance_meters: float = 5.0) -> None:
        # İki referans çizgisinin normalize y konumları (0..1)
        self.line_a_y: float = 0.4
        self.line_b_y: float = 0.6
        self.real_distance_meters = real_distance_meters
        self.calibrated = False  # kullanıcı mesafe girdiyse True

        # track_id -> son (zaman, normalize y, normalize x)
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=5))
        self.last_error: str | None = None

    def configure(
        self,
        line_a_y: float | None = None,
        line_b_y: float | None = None,
        real_distance_meters: float | None = None,
    ) -> None:
        if line_a_y is not None:
            self.line_a_y = float(line_a_y)
        if line_b_y is not None:
            self.line_b_y = float(line_b_y)
        if real_distance_meters is not None:
            self.real_distance_meters = float(real_distance_meters)
            self.calibrated = True

    def update(self, detections: list[Detection], frame_shape: tuple[int, int]) -> None:
        """Her track için referans çizgileri arası piksel hızını m/s'ye çevirir."""
        try:
            h, w = frame_shape[:2]
            now = time.monotonic()
            span = abs(self.line_b_y - self.line_a_y) or 1e-6
            # Normalize y birim başına gerçek metre
            meters_per_unit_y = self.real_distance_meters / span

            seen: set[int] = set()
            for det in detections:
                if det.track_id is None:
                    continue
                seen.add(det.track_id)
                cx, cy = det.center
                ny = cy / max(h, 1)
                nx = cx / max(w, 1)
                hist = self._history[det.track_id]
                hist.append((now, ny, nx))

                if len(hist) >= 2:
                    t0, y0, _ = hist[0]
                    t1, y1, _ = hist[-1]
                    dt = t1 - t0
                    if dt > 1e-3:
                        dy_units = abs(y1 - y0)
                        meters = dy_units * meters_per_unit_y
                        det.speed_mps = meters / dt

            # Görünmeyen track geçmişini temizle (bellek sızıntısı önle)
            stale = [tid for tid in self._history if tid not in seen]
            for tid in stale:
                # Birkaç frame görünmezse sil
                if self._history[tid] and now - self._history[tid][-1][0] > 2.0:
                    del self._history[tid]
        except Exception as exc:
            self.last_error = str(exc)

    def get_state(self) -> dict:
        return {
            "line_a_y": self.line_a_y,
            "line_b_y": self.line_b_y,
            "real_distance_meters": self.real_distance_meters,
            "calibrated": self.calibrated,
        }
