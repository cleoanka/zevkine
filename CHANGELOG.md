# Changelog

Bu projedeki dikkate değer tüm değişiklikler bu dosyada belgelenir.
Format [Keep a Changelog](https://keepachangelog.com/) esaslıdır ve proje
[Semantic Versioning](https://semver.org/) kullanır.

## [Unreleased]

### Added
- **YOLO26** varsayılan model (NMS-free, end-to-end inference).
- **Pose tahmini** — YOLO26-pose modelleriyle COCO-17 iskelet görselleştirme.
- **Segmentasyon** — YOLO26-seg modelleriyle yarı saydam maske overlay.
- Model seçicide pose/seg varyantları.
- Docker / docker-compose desteği (Linux/CPU/CUDA dağıtımı).
- `Makefile` geliştirme kısayolları.
- GitHub Actions CI (ruff + py_compile + pytest).
- pre-commit konfigürasyonu, issue/PR şablonları, CONTRIBUTING rehberi.

## [0.1.0]

### Added
- macOS gerçek zamanlı çok modlu CV vision workstation ilk sürümü.
- Object detection, ByteTrack, zone analizi, hız tahmini, crowd heatmap,
  dwell time, line crossing, anomaly tespiti.
- Canlı dashboard, video kaydı, snapshot, log export.
- iPhone Continuity Camera desteği.
