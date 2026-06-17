"""
crossing.py — Line Crossing Counter (çizgi geçiş sayacı) modülü.

Kullanıcı canvas'ta bir çizgi (iki nokta) çizer. Her track ID için
çizginin hangi tarafında olduğu izlenir; taraf değiştirdiğinde geçiş
sayılır. Yön bilgisi (A->B ve B->A) ayrı ayrı tutulur.

Koordinatlar normalize (0..1).
"""

from __future__ import annotations

from ..detection import Detection


def _side(px: float, py: float, line: dict) -> float:
    """
    Noktanın çizginin hangi tarafında olduğunu işaret olarak döndürür.
    Çizgi: (x1,y1)->(x2,y2). Pozitif/negatif -> iki farklı yarı düzlem.
    """
    x1, y1 = line["x1"], line["y1"]
    x2, y2 = line["x2"], line["y2"]
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


class CrossingModule:
    def __init__(self) -> None:
        # normalize çizgi; varsayılan ekran ortasında yatay
        self.line: dict = {"x1": 0.2, "y1": 0.5, "x2": 0.8, "y2": 0.5}
        self.count_a_to_b = 0
        self.count_b_to_a = 0
        # track_id -> son işaret (taraf)
        self._last_side: dict[int, float] = {}
        self.last_error: str | None = None

    def configure(self, line: dict | None = None) -> None:
        if line:
            self.line = {
                "x1": float(line.get("x1", self.line["x1"])),
                "y1": float(line.get("y1", self.line["y1"])),
                "x2": float(line.get("x2", self.line["x2"])),
                "y2": float(line.get("y2", self.line["y2"])),
            }

    def reset(self) -> None:
        self.count_a_to_b = 0
        self.count_b_to_a = 0
        self._last_side.clear()

    def update(self, detections: list[Detection], frame_shape: tuple[int, int]) -> None:
        try:
            h, w = frame_shape[:2]
            for det in detections:
                if det.track_id is None:
                    continue
                cx, cy = det.center
                nx, ny = cx / max(w, 1), cy / max(h, 1)
                s = _side(nx, ny, self.line)
                prev = self._last_side.get(det.track_id)
                if prev is not None and prev != 0 and s != 0:
                    if prev < 0 and s > 0:
                        self.count_a_to_b += 1
                    elif prev > 0 and s < 0:
                        self.count_b_to_a += 1
                self._last_side[det.track_id] = s
        except Exception as exc:
            self.last_error = str(exc)

    def get_state(self) -> dict:
        return {
            "line": self.line,
            "count_a_to_b": self.count_a_to_b,
            "count_b_to_a": self.count_b_to_a,
            "total": self.count_a_to_b + self.count_b_to_a,
        }
