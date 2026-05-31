"""
FastAPI web server: thin wrapper around OCRPipeline with WebSocket progress.

Provides endpoints for PDF/image OCR processing, runtime configuration,
model discovery, and job history tracking.
"""

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pdf_ocr.api.routers import config, ocr, websocket

# ---------------------------------------------------------------------------
# Static files directory
# ---------------------------------------------------------------------------
_STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

app.include_router(config.router)
app.include_router(ocr.router)
app.include_router(websocket.router)

@app.get("/")
async def read_index():
    """Serve the single-page frontend."""
    return FileResponse(_STATIC_DIR / "index.html")

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Local LLM PDF OCR web server (FastAPI + WebSocket progress).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    args = parser.parse_args()
    uvicorn.run(
        "pdf_ocr.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

if __name__ == "__main__":
    main()
