"""
websocket.py — WebSocket frame yayıncısı.

İstemci /ws'e bağlanır; sunucu hedef FPS'te en güncel payload'ı (frame +
detection metası + overlay verileri) JSON olarak gönderir. Disconnect
graceful biçimde ele alınır; bir istemcinin kopması diğerlerini etkilemez.

Ayrıca /mjpeg uç noktası, canvas kullanmak istemeyen istemciler için
klasik multipart MJPEG stream sağlar.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

ws_router = APIRouter()


@ws_router.websocket("/ws")
async def stream_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    pipeline = websocket.app.state.pipeline
    target_fps = pipeline.config.get("target_fps", 30)
    interval = 1.0 / max(target_fps, 1.0)

    try:
        while True:
            payload = pipeline.get_latest_payload()
            if payload:
                # default=str: beklenmedik tip serileştirmede çökmesin
                await websocket.send_text(json.dumps(payload, default=str))
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        # İstemci kapattı — sessizce çık
        return
    except Exception:
        # Diğer hatalar: bağlantıyı kapatmayı dene, sistemi çökertme
        try:
            await websocket.close()
        except Exception:
            pass


@ws_router.get("/mjpeg")
def stream_mjpeg(request: Request) -> StreamingResponse:
    """Klasik multipart/x-mixed-replace MJPEG akışı (ham frame'ler)."""
    pipeline = request.app.state.pipeline
    target_fps = pipeline.config.get("target_fps", 30)
    interval = 1.0 / max(target_fps, 1.0)

    async def generate():
        boundary = b"--frame"
        while True:
            jpeg = pipeline.get_latest_jpeg()
            if jpeg is not None:
                yield (
                    boundary
                    + b"\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )
            await asyncio.sleep(interval)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
