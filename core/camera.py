"""
camera.py — Kamera yönetimi.

CameraManager:
  * Sistemdeki kameraları enumerate eder (OpenCV index 0..N probe).
  * Seçili kamerayı ayrı bir thread'de okur (producer).
  * Lock-free benzeri davranış için deque(maxlen=2) kullanır; böylece
    latency birikmez — tüketici (inference) her zaman en taze frame'i alır.
  * Stream kesmeden hot-swap (kamera/çözünürlük değişimi) destekler.

macOS notu: Continuity Camera (iPhone), AVFoundation üzerinden normal bir
OpenCV index'i olarak görünür ve probe sırasında otomatik listelenir.
"""

from __future__ import annotations

import platform
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

# OpenCV'nin kendi thread havuzu Python thread'leriyle çakışmasın
cv2.setNumThreads(0)

# Yaygın çözünürlük ön ayarları (genişlik, yükseklik)
RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    # "native" özel olarak ele alınır: kameranın varsayılanı bırakılır
}


@dataclass
class CameraInfo:
    """Enumerate edilen bir kameranın tanımı."""

    id: int
    name: str
    width: int
    height: int
    fps: float

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "resolution": f"{self.width}x{self.height}",
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 1),
        }


@dataclass
class _Stats:
    """Kamera thread'inin canlı istatistikleri."""

    captured_frames: int = 0
    dropped_frames: int = 0
    measured_fps: float = 0.0
    buffer_fill: float = 0.0  # deque doluluk oranı 0..1
    last_error: str | None = None
    _fps_samples: deque = field(default_factory=lambda: deque(maxlen=30))


def _backend_for_platform() -> int:
    """Platforma uygun OpenCV capture backend'i seçer."""
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


class CameraManager:
    """
    Tek aktif kamerayı yöneten producer thread.

    Kullanım:
        cam = CameraManager()
        cam.list_cameras()         # mevcut kameralar
        cam.start(index=0)         # yakalamayı başlat
        frame = cam.read()         # en taze frame (BGR np.ndarray) ya da None
        cam.switch(index=1)        # kesintisiz geçiş
        cam.stop()
    """

    def __init__(self, max_probe: int = 10) -> None:
        self._max_probe = max_probe
        self._backend = _backend_for_platform()

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._lock = threading.RLock()

        # En taze 2 frame: maxlen=2 -> eski frame otomatik düşer (latency yok)
        self._queue: deque = deque(maxlen=2)

        self.current_index: int | None = None
        self.current_resolution: str = "720p"
        self.target_fps: float = 30.0
        self.stats = _Stats()

    # ----- enumerate ----------------------------------------------------------

    def list_cameras(self) -> list[CameraInfo]:
        """
        0..max_probe arası index'leri açmayı dener.
        Açılan her cihaz için çözünürlük/fps bilgisini toplar.

        Not: Bu işlem aktif kamerayı açmaz/kapatmaz; geçici handle kullanır.
        Zaten açık olan index için mevcut handle bilgisini kullanır.
        """
        cameras: list[CameraInfo] = []
        for idx in range(self._max_probe + 1):
            if idx == self.current_index and self._cap is not None:
                # Aktif kamerayı tekrar açmaya çalışma; mevcut handle'ı oku
                w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self._cap.get(cv2.CAP_PROP_FPS) or self.target_fps
                cameras.append(CameraInfo(idx, self._guess_name(idx), w, h, fps))
                continue

            cap = cv2.VideoCapture(idx, self._backend)
            if cap is not None and cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    cameras.append(CameraInfo(idx, self._guess_name(idx), w, h, fps))
            if cap is not None:
                cap.release()
        return cameras

    @staticmethod
    def _guess_name(idx: int) -> str:
        """
        İnsan-okunur kamera adı tahmini. macOS'ta düşük index genelde
        dahili FaceTime kamerası; Continuity Camera sonradan eklenir.
        """
        system = platform.system()
        if system == "Darwin":
            if idx == 0:
                return "FaceTime HD Camera (dahili)"
            return f"Camera {idx} (Continuity / USB olabilir)"
        return f"Camera {idx}"

    # ----- yaşam döngüsü ------------------------------------------------------

    def start(
        self,
        index: int,
        resolution: str = "720p",
        target_fps: float = 30.0,
    ) -> bool:
        """Belirtilen kamerayı açar ve okuma thread'ini başlatır."""
        with self._lock:
            self._open_capture(index, resolution, target_fps)
            if self._cap is None or not self._cap.isOpened():
                self.stats.last_error = f"Kamera {index} açılamadı"
                return False

            self.current_index = index
            self.current_resolution = resolution
            self.target_fps = target_fps
            self._queue.clear()
            self.stats = _Stats()

            self._running.set()
            self._thread = threading.Thread(
                target=self._capture_loop, name="camera-capture", daemon=True
            )
            self._thread.start()
            return True

    def _open_capture(self, index: int, resolution: str, target_fps: float) -> None:
        """Yeni bir VideoCapture açar ve özelliklerini ayarlar."""
        cap = cv2.VideoCapture(index, self._backend)
        if resolution in RESOLUTION_PRESETS:
            w, h = RESOLUTION_PRESETS[resolution]
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        # Sürücü tarafı buffer'ı küçült -> gecikme azalır
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._cap = cap

    def switch(
        self,
        index: int,
        resolution: str | None = None,
        target_fps: float | None = None,
    ) -> bool:
        """
        Aktif kamerayı kesintisiz değiştirir. Önceki kaynak düzgün release
        edilir. Stream broadcaster'ı durdurmaya gerek yoktur; read() yeni
        kameradan frame dönmeye başlar.
        """
        with self._lock:
            res = resolution or self.current_resolution
            fps = target_fps if target_fps is not None else self.target_fps
            self.stop()
            return self.start(index, res, fps)

    def stop(self) -> None:
        """Okuma thread'ini durdurur ve kamerayı serbest bırakır."""
        self._running.clear()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._queue.clear()
        self.current_index = None

    # ----- producer loop ------------------------------------------------------

    def _capture_loop(self) -> None:
        """
        Ayrı thread: sürekli frame okur ve en taze 2 frame'i deque'de tutar.
        target_fps'i aşan hız frame drop ile dengelenir; böylece tüketici
        gerçek zamanlı kalır.
        """
        min_interval = 1.0 / max(self.target_fps, 1.0)
        last_capture = 0.0

        while self._running.is_set():
            cap = self._cap
            if cap is None:
                break

            ok, frame = cap.read()
            now = time.monotonic()
            if not ok or frame is None:
                self.stats.last_error = "Frame okunamadı (kamera koptu?)"
                # Kısa bekleme: meşgul döngüyü önle (main thread değil)
                time.sleep(0.01)
                continue

            # FPS sınırlama: hedef aralıktan erkense bu frame'i at
            if now - last_capture < min_interval:
                self.stats.dropped_frames += 1
                continue
            last_capture = now

            # deque(maxlen=2): doluysa en eski otomatik düşer
            if len(self._queue) == self._queue.maxlen:
                self.stats.dropped_frames += 1
            self._queue.append(frame)

            # İstatistik güncelle
            self.stats.captured_frames += 1
            self.stats._fps_samples.append(now)
            self._update_fps()
            self.stats.buffer_fill = len(self._queue) / float(self._queue.maxlen or 1)

    def _update_fps(self) -> None:
        """Son örneklerden anlık yakalama FPS'i hesaplar."""
        samples = self.stats._fps_samples
        if len(samples) >= 2:
            span = samples[-1] - samples[0]
            if span > 0:
                self.stats.measured_fps = (len(samples) - 1) / span

    # ----- consumer -----------------------------------------------------------

    def read(self) -> np.ndarray | None:
        """
        En taze frame'i döndürür (kopya). Frame yoksa None.
        Tüketici (inference thread) bunu çağırır.
        """
        with self._lock:
            if not self._queue:
                return None
            return self._queue[-1].copy()

    def is_running(self) -> bool:
        return self._running.is_set()

    def get_stats(self) -> dict:
        return {
            "current_index": self.current_index,
            "resolution": self.current_resolution,
            "target_fps": self.target_fps,
            "measured_fps": round(self.stats.measured_fps, 1),
            "captured_frames": self.stats.captured_frames,
            "dropped_frames": self.stats.dropped_frames,
            "buffer_fill": round(self.stats.buffer_fill, 2),
            "last_error": self.stats.last_error,
        }
