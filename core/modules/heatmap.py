"""
heatmap.py — Crowd Density Heatmap modülü.

Son N saniyedeki tüm bbox merkezlerini birikimli bir ısı haritasına yazar,
Gaussian blur ile yumuşatır ve renkli bir overlay (JET) üretir. Overlay
PNG (base64) olarak frontend'e gönderilir ve opacity slider ile karıştırılır.

Performans: ısı haritası inference çözünürlüğünden bağımsız, sabit küçük bir
ızgarada (accumulator) tutulur; her frame decay ile sönümlenir.
"""

from __future__ import annotations

import base64
import time

import cv2
import numpy as np

from ..detection import Detection


class HeatmapModule:
    def __init__(
        self,
        grid_size: tuple[int, int] = (160, 90),
        window_seconds: float = 5.0,
        blur_kernel: int = 51,
        decay: float = 0.92,
    ) -> None:
        # accumulator (yükseklik, genişlik) — float32 yoğunluk
        self.grid_w, self.grid_h = grid_size
        self.window_seconds = window_seconds
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.decay = decay

        self._acc = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self._last_ts = time.monotonic()
        self.last_error: str | None = None

    def reset(self) -> None:
        self._acc[:] = 0.0

    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
    ) -> None:
        """Detection merkezlerini accumulator'a ekler ve zamanla sönümler."""
        try:
            h, w = frame_shape[:2]
            # Zaman bazlı decay: pencere boyunca eski katkı azalır
            now = time.monotonic()
            dt = now - self._last_ts
            self._last_ts = now
            # Pencereye göre normalize edilmiş decay faktörü
            factor = self.decay ** (dt / (self.window_seconds / 30.0 + 1e-6))
            self._acc *= np.clip(factor, 0.0, 1.0)

            for det in detections:
                cx, cy = det.center
                gx = int(cx / max(w, 1) * (self.grid_w - 1))
                gy = int(cy / max(h, 1) * (self.grid_h - 1))
                if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                    self._acc[gy, gx] += 1.0
        except Exception as exc:
            self.last_error = str(exc)

    def render_overlay(self, out_size: tuple[int, int]) -> str | None:
        """
        Isı haritasını renkli PNG (base64 data-uri) olarak döndürür.
        out_size: (genişlik, yükseklik) display boyutu.
        Boşsa None döner (overlay çizilmez).
        """
        try:
            if self._acc.max() <= 1e-6:
                return None

            norm = self._acc / (self._acc.max() + 1e-6)
            norm = (norm * 255).astype(np.uint8)
            norm = cv2.GaussianBlur(norm, (self.blur_kernel, self.blur_kernel), 0)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

            out_w, out_h = out_size
            colored = cv2.resize(
                colored, (out_w, out_h), interpolation=cv2.INTER_LINEAR
            )
            ok, buf = cv2.imencode(".png", colored)
            if not ok:
                return None
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            return f"data:image/png;base64,{b64}"
        except Exception as exc:
            self.last_error = str(exc)
            return None
