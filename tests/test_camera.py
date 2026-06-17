"""Kamera yardımcıları için testler (gerçek cihaz açmadan)."""

from core.camera import RESOLUTION_PRESETS, CameraInfo


def test_resolution_presets():
    assert RESOLUTION_PRESETS["720p"] == (1280, 720)
    assert RESOLUTION_PRESETS["1080p"] == (1920, 1080)
    assert RESOLUTION_PRESETS["480p"] == (640, 480)


def test_camera_info_to_dict():
    info = CameraInfo(id=0, name="Test Cam", width=1280, height=720, fps=30.0)
    out = info.to_dict()
    assert out["id"] == 0
    assert out["resolution"] == "1280x720"
    assert out["fps"] == 30.0
