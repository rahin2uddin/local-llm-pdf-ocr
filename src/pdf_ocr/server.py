"""
FastAPI web server: thin wrapper around OCRPipeline with WebSocket progress.

Provides endpoints for PDF/image OCR processing, runtime configuration,
model discovery, and job history tracking.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

ASGIReceive = Callable[[], Awaitable[dict[str, Any]]]
ASGISend = Callable[[dict[str, Any]], Awaitable[None]]
ASGIScope = dict[str, Any]

# ---------------------------------------------------------------------------
# Static files directory
# ---------------------------------------------------------------------------
_STATIC_DIR = Path(__file__).parent / "static"


_WEB_EXTRA_MESSAGE = (
    "The web server requires the optional web dependencies. Install them with "
    "`uv sync --extra web` for a source checkout, or "
    "`pip install 'local-llm-pdf-ocr[web]'` for an installed package."
)


class ASGIApplication(Protocol):
    async def __call__(
        self,
        scope: ASGIScope,
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None: ...


def _load_optional_module(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Cannot start local-llm-pdf-ocr-server because `{exc.name}` is not "
            f"installed. {_WEB_EXTRA_MESSAGE}"
        ) from exc


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


def create_app() -> ASGIApplication:
    """Create the FastAPI app after optional web dependencies are available."""
    fastapi = _load_optional_module("fastapi")
    staticfiles = _load_optional_module("fastapi.staticfiles")

    from pdf_ocr.api.routers import config, ocr, websocket

    web_app = fastapi.FastAPI()
    web_app.mount(
        "/static",
        staticfiles.StaticFiles(directory=str(_STATIC_DIR)),
        name="static",
    )

    web_app.include_router(config.router)
    web_app.include_router(ocr.router)
    web_app.include_router(websocket.router)
    web_app.get("/")(read_index)

    return cast(ASGIApplication, web_app)


class LazyASGIApp:
    """ASGI proxy that defers FastAPI imports until the server is used."""

    def __init__(self, factory: Callable[[], ASGIApplication]) -> None:
        self._factory = factory
        self._app: ASGIApplication | None = None

    def _load(self) -> ASGIApplication:
        if self._app is None:
            self._app = self._factory()
        return self._app

    async def __call__(
        self,
        scope: ASGIScope,
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        await self._load()(scope, receive, send)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)


app = LazyASGIApp(create_app)


async def read_index() -> Any:
    """Serve the single-page frontend."""
    responses = _load_optional_module("fastapi.responses")
    return responses.FileResponse(_STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_host(value: str) -> str:
    host = value.strip()
    if not host:
        raise argparse.ArgumentTypeError("host must not be empty")
    return host


def _parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc

    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Local LLM PDF OCR web server (FastAPI + WebSocket progress).",
    )
    parser.add_argument(
        "--host",
        type=_parse_host,
        default="127.0.0.1",
        help="Bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=_parse_port,
        default=8000,
        help="Bind port (default: 8000)",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload (development)"
    )
    args = parser.parse_args(argv)

    try:
        uvicorn = _load_optional_module("uvicorn")
        app._load()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    uvicorn.run(
        "pdf_ocr.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
