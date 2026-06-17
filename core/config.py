"""
config.py — Konfigürasyon yönetimi.

config.yaml dosyasını okur, runtime'da güncellenebilir bir konfigürasyon
nesnesi sağlar ve değişiklikleri tekrar diske yazar. Tüm magic number'lar
buradan gelir; kod içinde sabit değer kullanılmaz.
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any

import yaml

# Proje kök dizini (bu dosya core/ altında olduğu için iki üst)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """İç içe dict'leri özyinelemeli birleştirir (override öncelikli)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """
    Thread-safe konfigürasyon sarmalayıcısı.

    Inference thread'i okurken API thread'i güncelleyebileceği için
    erişimler bir kilit ile korunur. `data` her zaman tutarlı bir
    snapshot döndürür.
    """

    def __init__(self, path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        self._path = Path(path)
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self.reload()

    # ----- yükleme / kaydetme -------------------------------------------------

    def reload(self) -> None:
        """Dosyadan yeniden okur. Dosya yoksa varsayılanlarla başlar."""
        with self._lock:
            if self._path.exists():
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = yaml.safe_load(f) or {}
            else:
                self._data = {}

    def save(self) -> None:
        """Mevcut konfigürasyonu YAML olarak diske yazar."""
        with self._lock:
            with open(self._path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    self._data,
                    f,
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                )

    # ----- erişim -------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Nokta ile ayrılmış anahtar yolu destekler: 'modules.tracking'."""
        with self._lock:
            node: Any = self._data
            for part in key.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    return default
            return copy.deepcopy(node)

    @property
    def data(self) -> dict[str, Any]:
        """Tüm konfigürasyonun derin kopyası (güvenli okuma)."""
        with self._lock:
            return copy.deepcopy(self._data)

    def update(self, partial: dict[str, Any], persist: bool = True) -> dict[str, Any]:
        """
        Kısmi konfigürasyonu mevcut değerlerle birleştirir.
        persist=True ise diske yazar. Güncellenmiş snapshot döner.
        """
        with self._lock:
            self._data = _deep_merge(self._data, partial)
            if persist:
                self.save()
            return copy.deepcopy(self._data)

    @property
    def export_dir(self) -> Path:
        """Export dizini, ~ expand edilmiş ve oluşturulmuş halde."""
        raw = self.get("export_dir", "~/OmniVision/exports")
        path = Path(raw).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path


# Uygulama genelinde paylaşılan tekil konfigürasyon nesnesi
config = Config()
