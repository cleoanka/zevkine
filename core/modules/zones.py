"""
zones.py — Zone (poligon bölge) analizi modülü.

Kullanıcı canvas'ta poligon çizer; her zone için anlık obje sayısı tutulur.
Sayı, zone'un `threshold` değerini aşarsa ihlal (violation) bayrağı kalkar
ve frontend o zone'u kırmızıya boyar. Zone'lar JSON olarak export/import
edilebilir.

Koordinatlar normalize (0..1) tutulur; böylece çözünürlük değişse de
zone'lar geçerli kalır.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..detection import Detection


@dataclass
class Zone:
    id: int
    name: str
    # normalize köşe noktaları: [[x,y], ...]  (0..1)
    points: list[list[float]]
    threshold: int = 5
    count: int = 0
    violated: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "points": self.points,
            "threshold": self.threshold,
            "count": self.count,
            "violated": self.violated,
        }


def _point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
    """Ray-casting algoritması ile nokta-poligon testi (normalize uzayda)."""
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


class ZonesModule:
    def __init__(self) -> None:
        self._zones: dict[int, Zone] = {}
        self._next_id = 1
        self.last_error: Optional[str] = None

    # ----- CRUD ---------------------------------------------------------------

    def add_zone(
        self, points: list[list[float]], name: str = "", threshold: int = 5
    ) -> Zone:
        zid = self._next_id
        self._next_id += 1
        zone = Zone(
            id=zid,
            name=name or f"Zone {zid}",
            points=points,
            threshold=threshold,
        )
        self._zones[zid] = zone
        return zone

    def remove_zone(self, zone_id: int) -> bool:
        return self._zones.pop(zone_id, None) is not None

    def clear(self) -> None:
        self._zones.clear()

    def export(self) -> list[dict]:
        return [z.to_dict() for z in self._zones.values()]

    def import_zones(self, data: list[dict]) -> None:
        """JSON'dan zone'ları yükler; id çakışmalarını yeniden numaralar."""
        self.clear()
        self._next_id = 1
        for item in data:
            self.add_zone(
                points=item.get("points", []),
                name=item.get("name", ""),
                threshold=int(item.get("threshold", 5)),
            )

    # ----- güncelleme ---------------------------------------------------------

    def update(
        self, detections: list[Detection], frame_shape: tuple[int, int]
    ) -> None:
        """Her zone için içindeki obje sayısını ve ihlal durumunu günceller."""
        try:
            h, w = frame_shape[:2]
            for zone in self._zones.values():
                zone.count = 0

            for det in detections:
                cx, cy = det.center
                nx, ny = cx / max(w, 1), cy / max(h, 1)
                for zone in self._zones.values():
                    if _point_in_polygon(nx, ny, zone.points):
                        zone.count += 1
                        det.in_zones.append(zone.id)

            for zone in self._zones.values():
                zone.violated = zone.count > zone.threshold
        except Exception as exc:
            self.last_error = str(exc)

    def get_state(self) -> list[dict]:
        return self.export()
