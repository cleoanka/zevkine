"""Detection veri modeli ve COCO yardımcıları için testler."""

from core.detection import COCO_CLASSES, Detection, class_color


def test_coco_has_80_classes():
    assert len(COCO_CLASSES) == 80
    assert COCO_CLASSES[0] == "person"
    assert COCO_CLASSES[2] == "car"


def test_detection_center():
    d = Detection(0, "person", 0.9, (10, 10, 50, 80))
    assert d.center == (30.0, 45.0)


def test_detection_color_is_stable_and_rgb():
    c1 = class_color(0)
    c2 = class_color(0)
    assert c1 == c2  # aynı sınıf -> aynı renk
    assert all(0 <= v <= 255 for v in c1)
    assert class_color(0) != class_color(5)  # farklı sınıf -> farklı renk


def test_detection_to_dict_round_trips_fields():
    d = Detection(2, "car", 0.876, (1.0, 2.0, 3.0, 4.0), track_id=7)
    out = d.to_dict()
    assert out["class_name"] == "car"
    assert out["track_id"] == 7
    assert out["confidence"] == 0.876
    assert out["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert isinstance(out["color"], tuple) or isinstance(out["color"], list)
