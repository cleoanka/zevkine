# Katkı Rehberi

OmniVision'a katkıda bulunduğun için teşekkürler! 🛰️

## Geliştirme ortamı

```bash
pip install -e ".[dev]"     # ruff + pytest + headless OpenCV
pre-commit install          # (opsiyonel) commit öncesi otomatik lint
```

## Kod standartları

- **Python 3.11+**, tip ipuçları (type hints) tercih edilir.
- **Magic number yok** — sabitler `config.yaml`'a gider.
- Her analiz modülü kendi hatasını yakalar; pipeline'ı çökertmez.
- Main thread'de `time.sleep()` kullanma.
- Kod yorumları Türkçe olabilir.

## Göndermeden önce

```bash
make lint     # ruff check .
make test     # pytest
make compile  # py_compile sağlık kontrolü
```

CI (`.github/workflows/ci.yml`) bu üçünü her PR'da otomatik çalıştırır.

## Commit & PR

- Anlamlı, açıklayıcı commit mesajları yaz.
- Yeni modül eklerken `core/modules/` desenini takip et ve bir test ekle.
- PR açıklamasında neyi/neden değiştirdiğini özetle.
