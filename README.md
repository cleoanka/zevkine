<div align="center">

# 🛰️ OMNIVISION

### macOS için gerçek zamanlı, çok modlu yapay zekâ görüş istasyonu

**YOLO26** üzerine kurulu · iPhone Continuity Camera destekli · tek komutla çalışan yerel CV platformu

<br/>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLO26](https://img.shields.io/badge/YOLO26-NMS--free-00ff88?style=for-the-badge&logo=ultralytics&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![macOS](https://img.shields.io/badge/macOS-MPS-000000?style=for-the-badge&logo=apple&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-e8e8f0?style=for-the-badge)

<br/>

`person · car · bicycle · dog · …` &nbsp; **+** &nbsp; `track · zone · speed · heatmap · dwell · crossing · anomaly`

</div>

---

> **OmniVision bir "object detection demo" değildir.**
> Kamera yönetimi, ByteTrack, yedi farklı analiz modülü, canlı dashboard, video
> kaydı ve export içeren **production-grade** bir yerel AI vision workstation'dır.
> Tamamen kendi makinende çalışır — buluta hiçbir görüntü gitmez.

<br/>

## 📑 İçindekiler

- [✨ Özellikler](#-özellikler)
- [🏗️ Mimari](#️-mimari)
- [📦 Kurulum](#-kurulum)
- [▶️ Çalıştırma](#️-çalıştırma)
- [📱 iPhone Continuity Camera](#-iphoneu-continuity-camera-olarak-kullanma-macos-13)
- [⚙️ Konfigürasyon](#️-konfigürasyon)
- [🗂️ Dosya Yapısı](#️-dosya-yapısı)
- [🔌 API](#-api-özeti)
- [⚡ Performans](#-performans-notları-macos)
- [🧪 Geliştirme & Test](#-geliştirme--test)

<br/>

## ✨ Özellikler

<table>
<tr><th align="left">Modül</th><th align="left">Ne yapar?</th></tr>
<tr><td>🎯 <b>Object Detection</b></td><td>COCO 80 sınıf · sınıfa özel sabit renkler · confidence + label</td></tr>
<tr><td>🆔 <b>ByteTrack</b></td><td>Kalıcı track ID + iz (trail) görselleştirme</td></tr>
<tr><td>🟩 <b>Zone Analizi</b></td><td>Canvas'ta poligon çiz · obje say · eşik aşımında ihlal alarmı · JSON export/import</td></tr>
<tr><td>🚗 <b>Hız Tahmini</b></td><td>İki referans çizgi + gerçek mesafe → m/s (kalibrasyonsuz mod uyarısı)</td></tr>
<tr><td>🔥 <b>Crowd Heatmap</b></td><td>Birikimli yoğunluk ısı haritası · Gaussian blur · opacity slider</td></tr>
<tr><td>⏱️ <b>Dwell Time</b></td><td>Track bazlı sahnede kalma süresi · eşik aşımında highlight</td></tr>
<tr><td>↔️ <b>Line Crossing</b></td><td>Çizgi geçiş sayacı · yönlü (A→B / B→A)</td></tr>
<tr><td>⚠️ <b>Anomaly</b></td><td>Obje sayısı hareketli ortalamanın 2x üzerine çıkınca banner</td></tr>
</table>

**Ek olarak:** canlı kamera hot-swap · çözünürlük seçici · model hot-swap
(YOLO26 / YOLO11 / YOLOv8) · runtime confidence & IoU slider'ları · sınıf filtresi ·
video kaydı (MP4) · snapshot (PNG) · log export (JSON/CSV) · sistem durumu paneli
(CPU / RAM / GPU / uptime).

<br/>

## 🏗️ Mimari

```
   iPhone / Dahili / USB Kamera
              │
              ▼
   ┌─────────────────────────────┐
   │  OpenCV Capture Thread       │   deque(maxlen=2) → latency birikmez
   └─────────────────────────────┘
              │  en taze frame
              ▼
   ┌─────────────────────────────┐
   │  YOLO26 Inference (NMS-free) │   MPS > CUDA > CPU otomatik
   │  + ByteTrack                 │   torch.inference_mode()
   └─────────────────────────────┘
              │  Detection[]
              ▼
   ┌─────────────────────────────┐
   │  Analiz Modülleri            │   zone · speed · heatmap
   │  (her biri izole, hatayı     │   dwell · crossing · anomaly
   │   kendi yakalar)             │
   └─────────────────────────────┘
              │
     ┌────────┴─────────┐
     ▼                  ▼
 WebSocket /ws      REST /api/*
 + MJPEG /mjpeg
     │
     ▼
 Browser Dashboard (Canvas render — npm/node YOK)
```

Producer (kamera) ve consumer (inference) **ayrı thread'lerde** çalışır;
main thread'de hiçbir `time.sleep()` yoktur. Her modül kendi hatasını yakalar —
biri çökse sistem ayakta kalır.

<br/>

## 📦 Kurulum

```bash
# Python 3.11+ önerilir
pip install -r requirements.txt
```

İlk çalıştırmada `ultralytics`, model ağırlığını (örn. `yolo26x.pt`) otomatik indirir.

<br/>

## ▶️ Çalıştırma

```bash
python main.py
```

Ardından tarayıcıda:

```
http://localhost:8000
```

<br/>

## 📱 iPhone'u Continuity Camera olarak kullanma (macOS 13+)

1. iPhone ve Mac **aynı Apple ID**'de, Bluetooth/Wi-Fi açık olsun.
2. iPhone'u Mac'e yaklaştır → otomatik kamera kaynağı olur.
3. Dashboard'da **CAMERA** sekmesinden iPhone'u seç, **GEÇİŞ YAP**.

> iPhone genelde 60 fps, dahili FaceTime kamerası 30 fps hedefiyle çalışır.

<br/>

## ⚙️ Konfigürasyon

Tüm parametreler `config.yaml` dosyasından gelir (kod içinde magic number yok).
Runtime'da `POST /api/config` ile güncellenir ve dosyaya yazılır.

```yaml
model: yolo26x.pt     # NMS-free; yolo11x.pt / yolov8x.pt da olabilir
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

<br/>

## 🗂️ Dosya Yapısı

```
omnivision/
├── main.py              # FastAPI app + lifespan
├── config.yaml          # Tüm konfigürasyon
├── pyproject.toml       # Ruff + pytest + paket metadata
├── requirements.txt
├── core/
│   ├── config.py        # Thread-safe config
│   ├── camera.py        # CameraManager (enumerate + capture thread)
│   ├── inference.py     # YOLOEngine (YOLO26, MPS/CUDA/CPU, ByteTrack)
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
├── static/
│   └── index.html       # Tek dosya dashboard (CSS+JS inline)
└── tests/               # pytest — saf mantık testleri
```

<br/>

## 🔌 API Özeti

| Method | Endpoint | Açıklama |
|:------:|----------|----------|
| `GET` | `/api/stats` | Canlı pipeline + sistem istatistikleri |
| `GET` `POST` | `/api/config` | Konfigürasyon oku / güncelle |
| `GET` | `/api/cameras` | Kameraları listele |
| `POST` | `/api/cameras/switch` | Kamera hot-swap |
| `POST` | `/api/filter/classes` | Sınıf filtresi |
| `GET` `POST` `DELETE` | `/api/zones` | Zone CRUD |
| `POST` | `/api/crossing/line` | Geçiş çizgisi ayarla |
| `POST` | `/api/speed/config` | Hız kalibrasyonu |
| `POST` | `/api/record/toggle` | Kaydı başlat/durdur |
| `POST` | `/api/snapshot` | PNG snapshot |
| `POST` | `/api/export/log` | JSON/CSV log export |
| `WS` | `/ws` | Canlı frame + meta yayını |
| `GET` | `/mjpeg` | Klasik MJPEG akışı |

<br/>

## ⚡ Performans Notları (macOS)

- **Apple Silicon (MPS)** otomatik seçilir; `half_precision` toggle ile FP16.
- **YOLO26 NMS-free** → daha hafif post-processing, düşük gecikme.
- `torch.inference_mode()` her zaman aktif.
- Frame queue `deque(maxlen=2)` → gecikme birikmez, daima en taze frame.
- `cv2.setNumThreads(0)` → Python thread havuzuyla çakışma yok.
- `frame_skip` ile düşük donanımda her N. frame işlenir.

<br/>

## 🧪 Geliştirme & Test

```bash
pip install -e ".[dev]"   # ruff + pytest + headless OpenCV
ruff check .              # lint
pytest                    # saf mantık testleri
```

CI (`.github/workflows/ci.yml`) her push & PR'da **ruff + py_compile + pytest**
çalıştırır. Claude Code on the web oturumları için `.claude/hooks/session-start.sh`
bağımlılıkları otomatik kurar.

<br/>

<div align="center">

**Hiçbir özellik atlanmadı. Tamamen yerel. Tamamen senin.** 🚀

<sub>MIT License · YOLO26 by Ultralytics</sub>

</div>
