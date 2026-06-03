from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType
from typing import Any

import pytest


def test_server_module_import_does_not_import_web_dependencies(
    monkeypatch: pytest.MonkeyPatch,
):
    original_import = builtins.__import__
    blocked_roots = {"fastapi", "uvicorn"}

    def block_web_imports(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".", maxsplit=1)[0] in blocked_roots:
            raise AssertionError(f"{name} was imported at module import time")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block_web_imports)
    sys.modules.pop("pdf_ocr.server", None)

    server = importlib.import_module("pdf_ocr.server")

    assert isinstance(server.app, server.LazyASGIApp)


def test_create_app_reports_missing_web_extra(monkeypatch: pytest.MonkeyPatch):
    from pdf_ocr import server

    original_import_module = importlib.import_module

    def missing_fastapi(name: str, package: str | None = None) -> ModuleType:
        if name == "fastapi":
            raise ModuleNotFoundError("No module named 'fastapi'", name="fastapi")
        return original_import_module(name, package)

    monkeypatch.setattr(server.importlib, "import_module", missing_fastapi)

    with pytest.raises(RuntimeError) as exc_info:
        server.create_app()

    message = str(exc_info.value)
    assert "fastapi" in message
    assert "uv sync --extra web" in message
    assert "local-llm-pdf-ocr[web]" in message


def test_main_reports_missing_uvicorn_as_system_exit(monkeypatch: pytest.MonkeyPatch):
    from pdf_ocr import server

    original_import_module = importlib.import_module

    def missing_uvicorn(name: str, package: str | None = None) -> ModuleType:
        if name == "uvicorn":
            raise ModuleNotFoundError("No module named 'uvicorn'", name="uvicorn")
        return original_import_module(name, package)

    monkeypatch.setattr(server.importlib, "import_module", missing_uvicorn)

    with pytest.raises(SystemExit) as exc_info:
        server.main([])

    message = str(exc_info.value)
    assert "uvicorn" in message
    assert "uv sync --extra web" in message
