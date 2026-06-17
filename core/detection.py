"""
detection.py — Paylaşılan veri modelleri ve COCO sınıf yardımcıları.

Tüm pipeline boyunca dolaşan hafif, serileştirilebilir Detection nesnesi
burada tanımlanır. Modüller bu nesneleri okur ve zenginleştirir.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

# COCO 80 sınıf isimleri (ultralytics ile aynı sıra)
COCO_CLASSES: list[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def class_color(class_id: int) -> tuple[int, int, int]:
    """
    Her sınıf için sabit ama ayırt edici bir renk üretir (golden-ratio HSV).
    Dönüş: (R, G, B) 0..255.
    """
    golden = 0.61803398875
    hue = (class_id * golden) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


@dataclass
class Detection:
    """Tek bir tespit. Pipeline boyunca modüllerce zenginleştirilir."""

    class_id: int
    class_name: str
    confidence: float
    # bbox: piksel koordinatları [x1, y1, x2, y2] (display çözünürlüğünde)
    bbox: tuple[float, float, float, float]
    track_id: int | None = None

    # YOLO26 çok-görevli (multi-task) çıktıları — model destekliyorsa dolar
    # keypoints: [(x, y, conf), ...] (pose modelleri, COCO-17 iskelet)
    keypoints: list[tuple[float, float, float]] | None = None
    # mask: segmentasyon poligonu [(x, y), ...] (seg modelleri)
    mask: list[tuple[float, float]] | None = None

    # Modüllerin doldurduğu ek alanlar
    speed_mps: float | None = None  # speed modülü
    dwell_seconds: float | None = None  # dwell modülü
    is_dwelling: bool = False
    in_zones: list[int] = field(default_factory=list)  # zones modülü

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def color(self) -> tuple[int, int, int]:
        return class_color(self.class_id)

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.bbox
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "track_id": self.track_id,
            "color": self.color,
            "keypoints": (
                [[round(x, 1), round(y, 1), round(c, 2)] for x, y, c in self.keypoints]
                if self.keypoints is not None
                else None
            ),
            "mask": (
                [[round(x, 1), round(y, 1)] for x, y in self.mask]
                if self.mask is not None
                else None
            ),
            "speed_mps": (
                round(self.speed_mps, 2) if self.speed_mps is not None else None
            ),
            "dwell_seconds": (
                round(self.dwell_seconds, 1) if self.dwell_seconds is not None else None
            ),
            "is_dwelling": self.is_dwelling,
            "in_zones": self.in_zones,
        }
