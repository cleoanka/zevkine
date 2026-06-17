"""
inference.py — YOLO inference motoru.

YOLOEngine:
  * Cihazı otomatik seçer (MPS > CUDA > CPU) ya da config'ten zorlar.
  * Model'i ultralytics ile yükler; runtime'da hot-swap edilebilir.
  * ByteTrack ile entegre (model.track) — kalıcı track ID üretir.
  * torch.inference_mode() her zaman aktif.
  * Inference latency ve FPS sayaçlarını tutar.

Bağımlılık (ultralytics/torch) yüklü değilse import patlamasın diye
yükleme tembel (lazy) yapılır; hata durumu durum bilgisine yazılır.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from .detection import COCO_CLASSES, Detection


def _select_device(preference: str) -> str:
    """
    Cihaz önceliği: auto -> mps > cuda > cpu.
    torch yoksa 'cpu' döner (inference zaten başarısız olur ama çökmez).
    """
    try:
        import torch
    except Exception:
        return "cpu"

    pref = (preference or "auto").lower()
    if pref in ("mps", "cuda", "cpu"):
        # Zorlanan cihaz kullanılabilir değilse cpu'ya düş
        if pref == "mps" and not getattr(torch.backends, "mps", None):
            return "cpu"
        if pref == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return pref

    # auto
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class YOLOEngine:
    """YOLO modelini saran, thread-safe inference motoru."""

    def __init__(
        self,
        model_name: str = "yolo11x.pt",
        device_pref: str = "auto",
        confidence: float = 0.35,
        iou: float = 0.45,
        inference_size: int = 640,
        half: bool = False,
        use_tracking: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = _select_device(device_pref)
        self.confidence = confidence
        self.iou = iou
        self.inference_size = inference_size
        self.half = half and self.device != "cpu"
        self.use_tracking = use_tracking

        self._model = None
        self._lock = threading.RLock()
        self._latency_samples: deque = deque(maxlen=30)
        self.last_error: Optional[str] = None
        self.ready = False

        self.load_model(model_name)

    # ----- model yönetimi -----------------------------------------------------

    def load_model(self, model_name: str) -> bool:
        """Model'i yükler (hot-swap için de kullanılır)."""
        with self._lock:
            try:
                from ultralytics import YOLO

                model = YOLO(model_name)
                # Cihaza taşı; bazı backend'lerde to() gerekebilir
                try:
                    model.to(self.device)
                except Exception:
                    pass
                self._model = model
                self.model_name = model_name
                self.ready = True
                self.last_error = None
                return True
            except Exception as exc:  # ultralytics/torch yoksa ya da indirme hatası
                self.ready = False
                self.last_error = f"Model yüklenemedi: {exc}"
                return False

    def set_device(self, device_pref: str) -> None:
        with self._lock:
            self.device = _select_device(device_pref)
            self.half = self.half and self.device != "cpu"
            if self._model is not None:
                try:
                    self._model.to(self.device)
                except Exception:
                    pass

    def update_thresholds(
        self, confidence: Optional[float] = None, iou: Optional[float] = None
    ) -> None:
        with self._lock:
            if confidence is not None:
                self.confidence = float(confidence)
            if iou is not None:
                self.iou = float(iou)

    # ----- inference ----------------------------------------------------------

    def predict(
        self,
        frame: np.ndarray,
        class_filter: Optional[list[int]] = None,
    ) -> list[Detection]:
        """
        Tek frame üzerinde inference yapar.
        use_tracking=True ise model.track ile kalıcı track ID üretilir.
        class_filter verilirse sadece o sınıf id'leri döner.

        Dönüş: Detection listesi (bbox display çözünürlüğünde, yani giriş
        frame boyutunda — ultralytics orijinal ölçeğe geri döndürür).
        """
        with self._lock:
            if not self.ready or self._model is None:
                return []

            start = time.perf_counter()
            try:
                import torch

                with torch.inference_mode():
                    if self.use_tracking:
                        results = self._model.track(
                            frame,
                            persist=True,
                            conf=self.confidence,
                            iou=self.iou,
                            imgsz=self.inference_size,
                            half=self.half,
                            device=self.device,
                            classes=class_filter,
                            tracker="bytetrack.yaml",
                            verbose=False,
                        )
                    else:
                        results = self._model.predict(
                            frame,
                            conf=self.confidence,
                            iou=self.iou,
                            imgsz=self.inference_size,
                            half=self.half,
                            device=self.device,
                            classes=class_filter,
                            verbose=False,
                        )
            except Exception as exc:
                self.last_error = f"Inference hatası: {exc}"
                return []
            finally:
                self._latency_samples.append(time.perf_counter() - start)

            return self._parse_results(results)

    def _parse_results(self, results) -> list[Detection]:
        """ultralytics sonucunu Detection listesine çevirir."""
        detections: list[Detection] = []
        if not results:
            return detections

        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = (
            boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
        )

        names = getattr(res, "names", None) or {
            i: n for i, n in enumerate(COCO_CLASSES)
        }

        for i in range(len(xyxy)):
            cid = int(cls_ids[i])
            detections.append(
                Detection(
                    class_id=cid,
                    class_name=names.get(cid, str(cid)),
                    confidence=float(confs[i]),
                    bbox=tuple(float(v) for v in xyxy[i]),
                    track_id=int(track_ids[i]) if track_ids is not None else None,
                )
            )
        return detections

    def reset_tracker(self) -> None:
        """Track ID'leri sıfırlamak için modeli yeniden yükle (basit yol)."""
        self.load_model(self.model_name)

    # ----- istatistik ---------------------------------------------------------

    @property
    def latency_ms(self) -> float:
        if not self._latency_samples:
            return 0.0
        return (sum(self._latency_samples) / len(self._latency_samples)) * 1000.0

    @property
    def inference_fps(self) -> float:
        lat = self.latency_ms
        return 1000.0 / lat if lat > 0 else 0.0

    def get_stats(self) -> dict:
        return {
            "model": self.model_name,
            "device": self.device,
            "half": self.half,
            "confidence": self.confidence,
            "iou": self.iou,
            "tracking": self.use_tracking,
            "latency_ms": round(self.latency_ms, 1),
            "inference_fps": round(self.inference_fps, 1),
            "ready": self.ready,
            "last_error": self.last_error,
        }
