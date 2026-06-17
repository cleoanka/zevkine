#!/bin/bash
# ============================================================
#  OmniVision — SessionStart hook (Claude Code on the web)
#  Lint ve testlerin web oturumlarında çalışabilmesi için
#  gerekli (hafif) bağımlılıkları kurar. Ağır torch/ultralytics
#  kurulmaz; inference.py bunları tembel (lazy) import eder.
# ============================================================
set -euo pipefail

# Sadece uzak (Claude Code on the web) ortamında çalış
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

echo "[omnivision] geliştirme bağımlılıkları kuruluyor..."
# pip yükseltmesi kritik değil; bazı sistem-yönetimli pip'lerde başarısız
# olabilir, bu durumda sessizce devam et.
python -m pip install --upgrade pip >/dev/null 2>&1 || true

# Idempotent: pip zaten kuruluysa hızlıca atlar
pip install \
  pyyaml numpy "opencv-python-headless>=4.9" psutil \
  fastapi "uvicorn[standard]" pydantic pillow websockets \
  pytest ruff

# Proje kökünü PYTHONPATH'e ekle ki testler import edebilsin
echo 'export PYTHONPATH="."' >> "$CLAUDE_ENV_FILE"

echo "[omnivision] hazır. 'pytest' ve 'ruff check .' kullanılabilir."
