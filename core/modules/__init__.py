"""OmniVision analiz modülleri.

Her modül bağımsızdır, kendi durumunu tutar ve kendi hatasını yakalar;
biri çökerse pipeline'ı durdurmaz. Tüm modüller `update(detections, frame)`
arayüzünü paylaşır ve görselleştirme için overlay verisi üretir.
"""

from .heatmap import HeatmapModule
from .zones import ZonesModule
from .speed import SpeedModule
from .dwell import DwellModule
from .crossing import CrossingModule
from .anomaly import AnomalyModule

__all__ = [
    "HeatmapModule",
    "ZonesModule",
    "SpeedModule",
    "DwellModule",
    "CrossingModule",
    "AnomalyModule",
]
