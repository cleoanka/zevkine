"""
pipeline.py — OmniVision'ın kalbi.

Producer (kamera thread) ile tüketici (inference thread) arasında köprü kurar:
  kamera frame -> preprocess -> YOLO inference -> ByteTrack -> modüller ->
  annotate (kayıt için) -> JPEG encode -> en güncel "payload" olarak sakla.

WebSocket broadcaster ve REST API bu payload'ı okur. Hiçbir time.sleep()
main thread'de değildir; inference kendi thread'inde döner.
"""

from __future__ import annotations

import base64
import threading
import time
from collections import deque
from typing import Any, Optional

import cv2
import numpy as np
import psutil

from .camera import CameraManager
from .config import Config
from .detection import COCO_CLASSES, Detection
from .inference import YOLOEngine
from .modules import (
    AnomalyModule,
    CrossingModule,
    DwellModule,
    HeatmapModule,
    SpeedModule,
    ZonesModule,
)
from .recorder import VideoRecorder, export_log, save_snapshot
from .tracker import TrackTrails


class Pipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        ms = config.get("module_settings", {})

        self.camera = CameraManager(max_probe=10)
        self.engine = YOLOEngine(
            model_name=config.get("model", "yolo11x.pt"),
            device_pref=config.get("device", "auto"),
            confidence=config.get("confidence", 0.35),
            iou=config.get("iou", 0.45),
            inference_size=config.get("inference_size", 640),
            half=config.get("half_precision", False),
            use_tracking=config.get("modules.tracking", True),
        )

        # Modüller
        self.trails = TrackTrails(ms.get("trail_length", 30))
        self.heatmap = HeatmapModule(
            window_seconds=ms.get("heatmap_window_seconds", 5.0),
            blur_kernel=ms.get("heatmap_blur_kernel", 51),
            decay=ms.get("heatmap_decay", 0.92),
        )
        self.zones = ZonesModule()
        self.speed = SpeedModule(ms.get("speed_real_distance_meters", 5.0))
        self.dwell = DwellModule(ms.get("dwell_threshold_seconds", 5.0))
        self.crossing = CrossingModule()
        self.anomaly = AnomalyModule(
            window_seconds=ms.get("anomaly_window_seconds", 30.0),
            multiplier=ms.get("anomaly_multiplier", 2.0),
        )

        self.recorder = VideoRecorder(config.export_dir)

        # Sınıf filtresi: None -> tümü
        self.class_filter: Optional[list[int]] = None

        # En güncel çıktı (thread-safe)
        self._payload_lock = threading.RLock()
        self._latest_payload: dict[str, Any] = {}
        self._latest_jpeg: Optional[bytes] = None

        # Detection event log (ring buffer)
        self._event_log: deque = deque(maxlen=5000)

        # Pipeline thread kontrolü
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_counter = 0
        self._start_time = time.monotonic()
        self._pipeline_fps_samples: deque = deque(maxlen=30)

    # ----- yaşam döngüsü ------------------------------------------------------

    def start(self) -> None:
        """Kamerayı ve inference thread'ini başlatır."""
        idx = self.config.get("camera_index", 0)
        self.camera.start(
            index=idx,
            resolution="720p",
            target_fps=self.config.get("target_fps", 30),
        )
        self._running.set()
        self._thread = threading.Thread(
            target=self._loop, name="inference-pipeline", daemon=True
        )
        self._thread.start()

    def shutdown(self) -> None:
        self._running.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.recorder.stop()
        self.camera.stop()

    # ----- ana döngü ----------------------------------------------------------

    def _loop(self) -> None:
        target_fps = self.config.get("target_fps", 30)
        min_interval = 1.0 / max(target_fps, 1.0)
        frame_skip = max(int(self.config.get("frame_skip", 1)), 1)
        jpeg_quality = int(self.config.get("jpeg_quality", 80))
        last_iter = 0.0
        last_dets: list[Detection] = []

        while self._running.is_set():
            now = time.monotonic()
            if now - last_iter < min_interval:
                # Meşgul-bekleme yerine kısa uyku (bu thread main değil)
                time.sleep(0.001)
                continue
            last_iter = now

            frame = self.camera.read()
            if frame is None:
                time.sleep(0.005)
                continue

            self._frame_counter += 1
            h, w = frame.shape[:2]
            frame_shape = (h, w)

            # Frame skip: bazı frame'lerde inference atlanır, önceki sonuç çizilir
            run_inference = (self._frame_counter % frame_skip) == 0
            if run_inference:
                try:
                    dets = self.engine.predict(frame, self.class_filter)
                except Exception:
                    dets = []
                last_dets = dets
            else:
                dets = last_dets

            # Modülleri çalıştır (her biri kendi hatasını yutar)
            self._run_modules(dets, frame_shape)

            # Kayıt için annotated frame üret
            annotated = self._annotate(frame.copy(), dets)
            if self.recorder.recording:
                self.recorder.write(annotated)

            # JPEG encode (ham frame -> frontend canvas overlay'leri kendi çizer)
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            jpeg_bytes = buf.tobytes() if ok else None

            # Event log'a yaz
            self._log_events(dets)

            # Pipeline FPS
            self._pipeline_fps_samples.append(now)

            # Payload oluştur
            payload = self._build_payload(dets, frame_shape, jpeg_bytes)
            with self._payload_lock:
                self._latest_payload = payload
                self._latest_jpeg = jpeg_bytes

    def _run_modules(
        self, dets: list[Detection], frame_shape: tuple[int, int]
    ) -> None:
        mods = self.config.get("modules", {})
        if mods.get("tracking", True):
            self.trails.update(dets, frame_shape)
        if mods.get("heatmap", True):
            self.heatmap.update(dets, frame_shape)
        if mods.get("zones", True):
            self.zones.update(dets, frame_shape)
        if mods.get("speed", False):
            self.speed.update(dets, frame_shape)
        if mods.get("dwell", True):
            self.dwell.update(dets, frame_shape)
        if mods.get("crossing", True):
            self.crossing.update(dets, frame_shape)
        if mods.get("anomaly", True):
            self.anomaly.update(dets, frame_shape)

    # ----- annotate (kayıt için sunucu tarafı çizim) --------------------------

    def _annotate(
        self, frame: np.ndarray, dets: list[Detection]
    ) -> np.ndarray:
        """Kayıt edilen MP4 için bbox/label çizimi (BGR)."""
        for det in dets:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            r, g, b = det.color
            color = (b, g, r)  # OpenCV BGR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = det.class_name
            if det.track_id is not None:
                label += f" #{det.track_id}"
            label += f" {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1
            )
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return frame

    # ----- event log ----------------------------------------------------------

    def _log_events(self, dets: list[Detection]) -> None:
        ts = time.time()
        for det in dets:
            self._event_log.append(
                {
                    "timestamp": round(ts, 3),
                    "class_name": det.class_name,
                    "class_id": det.class_id,
                    "confidence": round(det.confidence, 3),
                    "track_id": det.track_id,
                }
            )

    # ----- payload ------------------------------------------------------------

    def _build_payload(
        self,
        dets: list[Detection],
        frame_shape: tuple[int, int],
        jpeg_bytes: Optional[bytes],
    ) -> dict[str, Any]:
        h, w = frame_shape
        mods = self.config.get("modules", {})

        image_uri = None
        if jpeg_bytes is not None:
            b64 = base64.b64encode(jpeg_bytes).decode("ascii")
            image_uri = f"data:image/jpeg;base64,{b64}"

        heatmap_uri = None
        if mods.get("heatmap", True):
            heatmap_uri = self.heatmap.render_overlay((w, h))

        return {
            "type": "frame",
            "image": image_uri,
            "width": w,
            "height": h,
            "detections": [d.to_dict() for d in dets],
            "trails": (
                self.trails.get_trails() if mods.get("tracking", True) else {}
            ),
            "zones": self.zones.get_state(),
            "crossing": self.crossing.get_state(),
            "speed": self.speed.get_state(),
            "dwell": self.dwell.get_state(),
            "anomaly": self.anomaly.get_state(),
            "heatmap": heatmap_uri,
            "stats": self.get_stats(),
        }

    def get_latest_payload(self) -> dict[str, Any]:
        with self._payload_lock:
            return dict(self._latest_payload)

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._payload_lock:
            return self._latest_jpeg

    # ----- istatistik / sistem durumu -----------------------------------------

    @property
    def pipeline_fps(self) -> float:
        s = self._pipeline_fps_samples
        if len(s) >= 2 and (s[-1] - s[0]) > 0:
            return (len(s) - 1) / (s[-1] - s[0])
        return 0.0

    def get_stats(self) -> dict:
        proc = psutil.Process()
        mem = psutil.virtual_memory()
        gpu_mem = self._gpu_memory_mb()
        return {
            "pipeline_fps": round(self.pipeline_fps, 1),
            "objects": len(self.get_latest_payload().get("detections", [])),
            "active_tracks": self.trails.active_count(),
            "camera": self.camera.get_stats(),
            "engine": self.engine.get_stats(),
            "recorder": self.recorder.get_state(),
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_percent": mem.percent,
                "ram_used_mb": round(mem.used / 1024 / 1024),
                "process_mem_mb": round(proc.memory_info().rss / 1024 / 1024),
                "gpu_mem_mb": gpu_mem,
                "uptime_seconds": round(time.monotonic() - self._start_time),
            },
        }

    @staticmethod
    def _gpu_memory_mb() -> Optional[float]:
        try:
            import torch

            if torch.cuda.is_available():
                return round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "current_allocated_memory"):
                return round(mps.current_allocated_memory() / 1024 / 1024, 1)
        except Exception:
            return None
        return None

    # ----- runtime kontrol (API'den çağrılır) ---------------------------------

    def set_class_filter(self, class_ids: Optional[list[int]]) -> None:
        self.class_filter = class_ids if class_ids else None

    def apply_config(self, partial: dict) -> dict:
        """Runtime config güncellemesini ilgili bileşenlere yansıtır."""
        merged = self.config.update(partial)

        if "confidence" in partial or "iou" in partial:
            self.engine.update_thresholds(
                merged.get("confidence"), merged.get("iou")
            )
        if "device" in partial:
            self.engine.set_device(merged["device"])
        if "model" in partial and partial["model"] != self.engine.model_name:
            self.engine.load_model(partial["model"])
        if "modules" in partial:
            self.engine.use_tracking = merged.get("modules", {}).get(
                "tracking", True
            )

        ms = merged.get("module_settings", {})
        self.trails.configure(ms.get("trail_length", self.trails.trail_length))
        self.dwell.configure(ms.get("dwell_threshold_seconds"))
        self.anomaly.configure(
            ms.get("anomaly_window_seconds"), ms.get("anomaly_multiplier")
        )
        return merged

    def get_event_log(self, minutes: float = 5.0) -> list[dict]:
        cutoff = time.time() - minutes * 60
        return [e for e in self._event_log if e["timestamp"] >= cutoff]

    # ----- export / kayıt eylemleri -------------------------------------------

    def snapshot(self) -> Optional[str]:
        jpeg = self.get_latest_jpeg()
        if jpeg is None:
            return None
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        path = save_snapshot(frame, self.config.export_dir)
        return str(path)

    def export_detection_log(self, minutes: float, fmt: str) -> str:
        events = self.get_event_log(minutes)
        path = export_log(events, self.config.export_dir, fmt)
        return str(path)

    def toggle_recording(self) -> dict:
        if self.recorder.recording:
            path = self.recorder.stop()
            return {"recording": False, "path": str(path) if path else None}
        jpeg = self.get_latest_jpeg()
        if jpeg is None:
            return {"recording": False, "error": "Henüz frame yok"}
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        path = self.recorder.start(frame, fps=self.config.get("target_fps", 30))
        return {"recording": True, "path": str(path)}
