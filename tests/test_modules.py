"""Analiz modüllerinin saf (deterministik) mantığı için testler.

Kamera/model/torch gerektirmez; yalnızca Detection nesneleriyle çalışır.
opencv-headless + numpy kuruluysa core.modules import edilebilir.
"""

from core.detection import Detection
from core.modules.anomaly import AnomalyModule
from core.modules.crossing import CrossingModule, _side
from core.modules.dwell import DwellModule
from core.modules.zones import ZonesModule, _point_in_polygon
from core.tracker import TrackTrails

SHAPE = (720, 1280)  # (h, w)


def _det(cx, cy, track_id=None, w=1280, h=720):
    """Merkezi (cx,cy) olan küçük bir bbox üretir."""
    return Detection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=(cx - 5, cy - 5, cx + 5, cy + 5),
        track_id=track_id,
    )


# ----- zones -----------------------------------------------------------------


def test_point_in_polygon():
    square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert _point_in_polygon(0.5, 0.5, square) is True
    assert _point_in_polygon(1.5, 0.5, square) is False
    assert _point_in_polygon(0.5, 0.5, [[0, 0], [1, 0]]) is False  # <3 nokta


def test_zone_counts_and_violation():
    z = ZonesModule()
    # Ekranın sol yarısını kaplayan zone (normalize)
    z.add_zone([[0, 0], [0.5, 0], [0.5, 1], [0, 1]], name="sol", threshold=1)
    # sol yarıda 2 obje, sağ yarıda 1 obje
    dets = [_det(100, 100), _det(200, 200), _det(1000, 400)]
    z.update(dets, SHAPE)
    state = z.export()[0]
    assert state["count"] == 2
    assert state["violated"] is True  # 2 > eşik(1)


def test_zone_export_import_round_trip():
    z = ZonesModule()
    z.add_zone([[0, 0], [1, 0], [1, 1]], name="t", threshold=3)
    data = z.export()
    z2 = ZonesModule()
    z2.import_zones(data)
    assert z2.export()[0]["threshold"] == 3
    assert z2.export()[0]["points"] == [[0, 0], [1, 0], [1, 1]]


# ----- crossing --------------------------------------------------------------


def test_side_sign_differs_across_line():
    line = {"x1": 0.2, "y1": 0.5, "x2": 0.8, "y2": 0.5}
    above = _side(0.5, 0.4, line)
    below = _side(0.5, 0.6, line)
    assert (above < 0) != (below < 0)


def test_crossing_counts_directional():
    c = CrossingModule()
    c.configure({"x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 0.5})
    # track 1: üstten alta (A->B)
    c.update([_det(640, 200, track_id=1)], SHAPE)
    c.update([_det(640, 500, track_id=1)], SHAPE)
    assert c.count_a_to_b + c.count_b_to_a == 1
    # track 1: alttan üste (geri dönüş) -> diğer yön
    c.update([_det(640, 200, track_id=1)], SHAPE)
    state = c.get_state()
    assert state["total"] == 2


# ----- anomaly ---------------------------------------------------------------


def test_anomaly_flags_spike():
    a = AnomalyModule(window_seconds=60, multiplier=2.0)
    # Baseline ~2 obje
    for _ in range(10):
        a.update([_det(1, 1), _det(2, 2)], SHAPE)
    assert a.is_anomaly is False
    # Ani sıçrama: 10 obje
    a.update([_det(i, i) for i in range(10)], SHAPE)
    assert a.is_anomaly is True


# ----- dwell -----------------------------------------------------------------


def test_dwell_accumulates_time():
    d = DwellModule(threshold_seconds=0.0)
    dets = [_det(100, 100, track_id=5)]
    d.update(dets, SHAPE)
    assert dets[0].dwell_seconds is not None
    assert dets[0].is_dwelling is True  # eşik 0 -> hemen dwelling


# ----- tracker trails --------------------------------------------------------


def test_trails_accumulate_normalized_positions():
    t = TrackTrails(trail_length=5)
    for x in (100, 200, 300):
        t.update([_det(x, 360, track_id=9)], SHAPE)
    trails = t.get_trails()
    assert 9 in trails
    assert len(trails[9]) == 3
    # normalize: ilk x = 100/1280
    assert abs(trails[9][0][0] - 100 / 1280) < 1e-6
    assert t.active_count() == 1
