"""Config yükleme, nokta-yol erişimi ve derin birleştirme testleri."""

from pathlib import Path

from core.config import Config

SAMPLE = """
model: yolo26x.pt
confidence: 0.35
modules:
  tracking: true
  speed: false
nested:
  a:
    b: 1
"""


def _write_cfg(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(SAMPLE, encoding="utf-8")
    return p


def test_dotted_get(tmp_path):
    cfg = Config(_write_cfg(tmp_path))
    assert cfg.get("model") == "yolo26x.pt"
    assert cfg.get("modules.tracking") is True
    assert cfg.get("modules.speed") is False
    assert cfg.get("nested.a.b") == 1
    assert cfg.get("yok.bir.sey", "default") == "default"


def test_deep_merge_update_preserves_siblings(tmp_path):
    cfg = Config(_write_cfg(tmp_path))
    cfg.update({"modules": {"speed": True}}, persist=False)
    # speed güncellendi ama tracking korundu
    assert cfg.get("modules.speed") is True
    assert cfg.get("modules.tracking") is True


def test_update_persists_to_disk(tmp_path):
    path = _write_cfg(tmp_path)
    cfg = Config(path)
    cfg.update({"confidence": 0.5}, persist=True)
    reloaded = Config(path)
    assert reloaded.get("confidence") == 0.5
