"""
routes.py — REST API endpoint'leri.

Tüm /api/* uçları burada. Pipeline nesnesine request.app.state.pipeline
üzerinden erişilir. Her uç, hatayı yakalayıp anlamlı bir HTTP cevabı döner.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.detection import COCO_CLASSES

router = APIRouter(prefix="/api")


def _pipeline(request: Request):
    return request.app.state.pipeline


# ----- modeller (request gövdeleri) ------------------------------------------


class ConfigUpdate(BaseModel):
    # Serbest biçimli kısmi config; pipeline doğrular
    data: dict[str, Any]


class CameraSwitch(BaseModel):
    index: int
    resolution: Optional[str] = None
    target_fps: Optional[float] = None


class ClassFilter(BaseModel):
    class_ids: Optional[list[int]] = None


class ZoneCreate(BaseModel):
    points: list[list[float]]
    name: str = ""
    threshold: int = 5


class ZoneImport(BaseModel):
    zones: list[dict]


class LineConfig(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class SpeedConfig(BaseModel):
    line_a_y: Optional[float] = None
    line_b_y: Optional[float] = None
    real_distance_meters: Optional[float] = None


class LogExport(BaseModel):
    minutes: float = 5.0
    format: str = "json"


# ----- genel ------------------------------------------------------------------


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "app": "OmniVision"}


@router.get("/classes")
def classes() -> dict:
    """COCO sınıf listesi (id + isim) — class filter UI için."""
    return {"classes": [{"id": i, "name": n} for i, n in enumerate(COCO_CLASSES)]}


@router.get("/stats")
def stats(request: Request) -> dict:
    return _pipeline(request).get_stats()


# ----- konfigürasyon ----------------------------------------------------------


@router.get("/config")
def get_config(request: Request) -> dict:
    return _pipeline(request).config.data


@router.post("/config")
def update_config(request: Request, body: ConfigUpdate) -> dict:
    merged = _pipeline(request).apply_config(body.data)
    return {"status": "ok", "config": merged}


# ----- kameralar --------------------------------------------------------------


@router.get("/cameras")
def list_cameras(request: Request) -> dict:
    cams = _pipeline(request).camera.list_cameras()
    return {"cameras": [c.to_dict() for c in cams]}


@router.post("/cameras/switch")
def switch_camera(request: Request, body: CameraSwitch) -> dict:
    cam = _pipeline(request).camera
    ok = cam.switch(body.index, body.resolution, body.target_fps)
    if not ok:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Kamera {body.index} açılamadı"},
        )
    return {"status": "ok", "camera": cam.get_stats()}


# ----- sınıf filtresi ---------------------------------------------------------


@router.post("/filter/classes")
def set_class_filter(request: Request, body: ClassFilter) -> dict:
    _pipeline(request).set_class_filter(body.class_ids)
    return {"status": "ok", "class_ids": body.class_ids}


# ----- zone'lar ---------------------------------------------------------------


@router.get("/zones")
def get_zones(request: Request) -> dict:
    return {"zones": _pipeline(request).zones.export()}


@router.post("/zones")
def add_zone(request: Request, body: ZoneCreate) -> dict:
    zone = _pipeline(request).zones.add_zone(
        body.points, body.name, body.threshold
    )
    return {"status": "ok", "zone": zone.to_dict()}


@router.delete("/zones/{zone_id}")
def remove_zone(request: Request, zone_id: int) -> dict:
    ok = _pipeline(request).zones.remove_zone(zone_id)
    return {"status": "ok" if ok else "not_found"}


@router.post("/zones/clear")
def clear_zones(request: Request) -> dict:
    _pipeline(request).zones.clear()
    return {"status": "ok"}


@router.post("/zones/import")
def import_zones(request: Request, body: ZoneImport) -> dict:
    _pipeline(request).zones.import_zones(body.zones)
    return {"status": "ok", "zones": _pipeline(request).zones.export()}


# ----- çizgi geçiş & hız ------------------------------------------------------


@router.post("/crossing/line")
def set_crossing_line(request: Request, body: LineConfig) -> dict:
    _pipeline(request).crossing.configure(body.model_dump())
    return {"status": "ok", "crossing": _pipeline(request).crossing.get_state()}


@router.post("/crossing/reset")
def reset_crossing(request: Request) -> dict:
    _pipeline(request).crossing.reset()
    return {"status": "ok"}


@router.post("/speed/config")
def set_speed_config(request: Request, body: SpeedConfig) -> dict:
    _pipeline(request).speed.configure(
        body.line_a_y, body.line_b_y, body.real_distance_meters
    )
    return {"status": "ok", "speed": _pipeline(request).speed.get_state()}


# ----- kayıt & export ---------------------------------------------------------


@router.post("/record/toggle")
def toggle_record(request: Request) -> dict:
    return _pipeline(request).toggle_recording()


@router.post("/snapshot")
def snapshot(request: Request) -> dict:
    path = _pipeline(request).snapshot()
    if path is None:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Henüz frame yok"},
        )
    return {"status": "ok", "path": path}


@router.post("/export/log")
def export_log_endpoint(request: Request, body: LogExport) -> dict:
    fmt = body.format if body.format in ("json", "csv") else "json"
    path = _pipeline(request).export_detection_log(body.minutes, fmt)
    return {"status": "ok", "path": path}


@router.get("/log")
def get_log(request: Request, minutes: float = 5.0) -> dict:
    return {"events": _pipeline(request).get_event_log(minutes)}


@router.post("/tracker/reset")
def reset_tracker(request: Request) -> dict:
    p = _pipeline(request)
    p.engine.reset_tracker()
    p.trails.reset()
    return {"status": "ok"}
