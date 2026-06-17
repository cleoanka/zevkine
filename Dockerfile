# ============================================================
#  OmniVision — Docker imajı (headless / sunucu dağıtımı)
#  Not: macOS Continuity Camera ve MPS yalnızca yerel (bare-metal)
#  macOS'ta çalışır. Bu imaj Linux/CPU veya CUDA dağıtımı içindir.
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV runtime bağımlılıkları (headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Önce bağımlılıklar → katman cache'i
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

# Sunucuda kamera /dev/video* olarak --device ile bağlanmalıdır
CMD ["python", "main.py"]
