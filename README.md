# 🛰️ OMNIVISION

**macOS için production-grade, gerçek zamanlı çok modlu CV analiz platformu.**
YOLO11x / YOLOv8x üzerine inşa edilmiş; iPhone (Continuity Camera), dahili ve
USB kameralarla çalışan yerel bir AI vision workstation.

Bu bir "object detection demo" değil — kamera yönetimi, çoklu analiz modülü,
canlı dashboard, kayıt ve export içeren bütünleşik bir sistemdir.

---

## ✨ Özellikler

| Modül | Açıklama |
|-------|----------|
| 🎯 **Object Detection** | COCO 80 sınıf, sınıfa özel sabit renkler, confidence + label |
| 🆔 **ByteTrack** | Kalıcı track ID + iz (trail) görselleştirme |
| 🟩 **Zone Analizi** | Canvas'ta poligon çiz, obje say, eşik aşımında ihlal alarmı, JSON export/import |
| 🚗 **Hız Tahmini** | İki referans çizgi + gerçek mesafe → m/s (kalibrasyonsuz mod uyarısı) |
| 🔥 **Crowd Heatmap** | Birikimli yoğunluk ısı haritası, Gaussian blur, opacity slider |
| ⏱️ **Dwell Time** | Track bazlı sahnede kalma süresi, eşik aşımında highlight |
| ↔️ **Line Crossing** | Çizgi geçiş sayacı, yönlü (A→B / B→A) |
| ⚠️ **Anomaly** | Obje sayısı hareketli ortalamanın 2x üzerine çıkınca banner |

Ek olarak: canlı kamera hot-swap, çözünürlük seçici, model hot-swap,
runtime confidence/IoU slider'ları, sınıf filtresi, video kaydı (MP4),
snapshot (PNG), log export (JSON/CSV) ve sistem durumu paneli (CPU/RAM/GPU).

---

## 🏗️ Mimari

```
iPhone / Dahili / USB Kamera
        ↓
 OpenCV Capture Thread (deque maxlen=2 → latency birikmez)
        ↓
 YOLO Inference (ultralytics, MPS/CUDA/CPU otomatik)
        ↓
 ByteTrack + Analiz Modülleri (zone/speed/heatmap/dwell/crossing/anomaly)
        ↓
 ┌─────────────────────────────────────┐
 │ WebSocket Broadcaster + MJPEG        │
 │ REST API (/api/*)                    │
 └─────────────────────────────────────┘
        ↓
 Browser Dashboard (Canvas render — npm/node gerekmez)
```

Producer (kamera) ve consumer (inference) ayrı thread'lerde çalışır;
main thread'de hiçbir `time.sleep()` yoktur. Her modül kendi hatasını
yakalar — biri çökse sistem ayakta kalır.

---

## 📦 Kurulum

```bash
# Python 3.11+ önerilir
pip install -r requirements.txt
```

İlk çalıştırmada `ultralytics` model ağırlığını (örn. `yolo11x.pt`)
otomatik indirir.

## ▶️ Çalıştırma

```bash
python main.py
# Tarayıcıda:
open http://localhost:8000
```

## 📱 iPhone'u Continuity Camera olarak kullanma (macOS 13+)

1. iPhone ve Mac aynı Apple ID'de, Bluetooth/Wi-Fi açık olsun.
2. iPhone'u Mac'e yaklaştır → otomatik kamera kaynağı olur.
3. Dashboard'da **CAMERA** sekmesinden iPhone'u seç, **GEÇİŞ YAP**.

---

## ⚙️ Konfigürasyon

Tüm parametreler `config.yaml` dosyasından gelir (magic number yok).
Runtime'da `POST /api/config` ile güncellenir ve dosyaya yazılır.

```yaml
model: yolo11x.pt
device: auto          # auto | mps | cuda | cpu
inference_size: 640
target_fps: 30
confidence: 0.35
iou: 0.45
modules:
  tracking: true
  heatmap: true
  zones: true
  speed: false
  dwell: true
  crossing: true
  anomaly: true
```

---

## 🗂️ Dosya Yapısı

```
omnivision/
├── main.py              # FastAPI app + lifespan
├── config.yaml          # Tüm konfigürasyon
├── requirements.txt
├── core/
│   ├── config.py        # Thread-safe config
│   ├── camera.py        # CameraManager (enumerate + capture thread)
│   ├── inference.py     # YOLOEngine (MPS/CUDA/CPU, ByteTrack)
│   ├── detection.py     # Detection veri modeli + COCO
│   ├── tracker.py       # Track iz (trail) yöneticisi
│   ├── recorder.py      # MP4 / snapshot / log export
│   ├── pipeline.py      # Producer-consumer orkestrasyon
│   └── modules/
│       ├── heatmap.py   ├── zones.py    ├── speed.py
│       ├── dwell.py     ├── crossing.py └── anomaly.py
├── api/
│   ├── routes.py        # /api/* REST
│   └── websocket.py     # WS + MJPEG yayıncı
└── static/
    └── index.html       # Tek dosya dashboard (CSS+JS inline)
```

---

## 🔌 API Özeti

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/api/stats` | Canlı pipeline + sistem istatistikleri |
| GET/POST | `/api/config` | Konfigürasyon oku / güncelle |
| GET | `/api/cameras` | Kameraları listele |
| POST | `/api/cameras/switch` | Kamera hot-swap |
| POST | `/api/filter/classes` | Sınıf filtresi |
| GET/POST/DELETE | `/api/zones` | Zone CRUD |
| POST | `/api/crossing/line` | Geçiş çizgisi ayarla |
| POST | `/api/speed/config` | Hız kalibrasyonu |
| POST | `/api/record/toggle` | Kaydı başlat/durdur |
| POST | `/api/snapshot` | PNG snapshot |
| POST | `/api/export/log` | JSON/CSV log export |
| WS | `/ws` | Canlı frame + meta yayını |
| GET | `/mjpeg` | Klasik MJPEG akışı |

---

## ⚡ Performans Notları (macOS)

- Apple Silicon'da **MPS** otomatik seçilir; `half_precision` toggle ile FP16.
- `torch.inference_mode()` her zaman aktif.
- Frame queue `deque(maxlen=2)` → gecikme birikmez, daima en taze frame.
- `cv2.setNumThreads(0)` → Python thread havuzuyla çakışma yok.
- `frame_skip` ile düşük donanımda her N. frame işlenir.

---

> Hiçbir özellik atlanmadı. Efsane olsun diye yapıldı. 🚀
