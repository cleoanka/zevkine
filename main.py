"""
main.py — OmniVision FastAPI uygulaması.

Başlatma:
    python main.py
Sonra tarayıcıda:
    http://localhost:8000

Yaşam döngüsü (lifespan):
  * Başlangıçta config okunur, Pipeline kurulur ve kamera + inference
    thread'leri başlatılır.
  * Kapanışta thread'ler düzgün durdurulur, kamera serbest bırakılır.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router as api_router
from api.websocket import ws_router
from core.config import config
from core.pipeline import Pipeline

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ----- başlangıç -----
    pipeline = Pipeline(config)
    pipeline.start()
    app.state.pipeline = pipeline
    try:
        yield
    finally:
        # ----- kapanış -----
        pipeline.shutdown()


app = FastAPI(title="OmniVision", version="1.0.0", lifespan=lifespan)

# Yerel kullanım için CORS açık (tek makinede çalışır)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(ws_router)


@app.get("/")
def index():
    """Tek dosya dashboard'u döner."""
    return FileResponse(STATIC_DIR / "index.html")


# Statik varlıklar (gerekirse) — index.html her şeyi inline taşır
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def run() -> None:
    server_cfg = config.get("server", {})
    uvicorn.run(
        app,
        host=server_cfg.get("host", "0.0.0.0"),
        port=int(server_cfg.get("port", 8000)),
        log_level="info",
    )


if __name__ == "__main__":
    run()
