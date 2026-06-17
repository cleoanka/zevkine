"""OmniVision analiz modülleri.

Her modül bağımsızdır, kendi durumunu tutar ve kendi hatasını yakalar;
biri çökerse pipeline'ı durdurmaz. Tüm modüller `update(detections, frame)`
arayüzünü paylaşır ve görselleştirme için overlay verisi üretir.
"""

from .anomaly import AnomalyModule
from .crossing import CrossingModule
from .dwell import DwellModule
from .heatmap import HeatmapModule
from .speed import SpeedModule
from .zones import ZonesModule

__all__ = [
    "HeatmapModule",
    "ZonesModule",
    "SpeedModule",
    "DwellModule",
    "CrossingModule",
    "AnomalyModule",
]
