from __future__ import annotations

import asyncio
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from pdf_ocr.api.routers import config, ocr
from pdf_ocr.api.services.security import UploadValidationError, save_validated_upload
from pdf_ocr.utils.security import is_ssrf_target


class _AsyncUpload:
    def __init__(self, data: bytes):
        self._data = data
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._data) - self._offset
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def _api_client() -> TestClient:
    app = FastAPI()
    app.include_router(config.router)
    app.include_router(ocr.router)
    return TestClient(app)


def _public_dns(host: str, port, *args, **kwargs):
    if host == "api.openai.com":
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.18.3.161", 443))]
    raise socket.gaierror(-2, "Name or service not known")


def _process_form() -> dict[str, str]:
    return {
        "client_id": "same-client",
        "api_base": "http://api.openai.com/v1",
        "api_key": "test-key",
        "model": "openai/test-model",
        "pipeline_mode": "hybrid",
        "dpi": "200",
        "concurrency": "1",
        "dense_mode": "auto",
        "dense_threshold": "60",
        "refine": "true",
        "max_image_dim": "1024",
        "self_correction": "false",
        "binarize": "false",
        "dual_engine": "false",
        "spellcheck": "none",
        "cross_page": "false",
    }


def _pdf_upload() -> tuple[str, bytes, str]:
    return ("input.pdf", b"%PDF-1.4\n%%EOF\n", "application/pdf")


def test_ssrf_fails_closed_and_requires_explicit_local_allowance():
    with patch.dict(os.environ, {}, clear=True):
        with patch("pdf_ocr.utils.security.socket.getaddrinfo") as getaddrinfo:
            getaddrinfo.side_effect = _public_dns
            assert is_ssrf_target("http://api.openai.com/v1") is False
            assert is_ssrf_target("localhost:1234/v1") is True
            assert is_ssrf_target("ftp://api.openai.com/v1") is True
            assert is_ssrf_target(None) is True

    with patch.dict(os.environ, {}, clear=True):
        with patch("pdf_ocr.utils.security.socket.getaddrinfo") as getaddrinfo:
            getaddrinfo.side_effect = socket.gaierror(-2, "Name or service not known")
            assert is_ssrf_target("http://does-not-resolve.example/v1") is True

    with patch.dict(os.environ, {"ALLOW_SSRF_LOCAL": "true"}, clear=True):
        assert is_ssrf_target("http://127.0.0.1:1234/v1") is False
        assert is_ssrf_target("http://metadata.google.internal/v1") is True


def test_config_update_rejects_string_booleans_and_local_api_base():
    client = _api_client()

    response = client.post("/api/config", json={"refine": "false"})
    assert response.status_code == 422

    with patch.dict(os.environ, {}, clear=True):
        response = client.post(
            "/api/config", json={"api_base": "http://127.0.0.1:1234/v1"}
        )
    assert response.status_code == 403
    assert "127.0.0.1" not in response.json()["error"]


def test_upload_validation_uses_streaming_limit_and_content_signature():
    async def run_checks():
        with pytest.raises(UploadValidationError) as too_large:
            await save_validated_upload(
                _AsyncUpload(b"%PDF-1.4\n" + b"x" * 16), max_bytes=8
            )
        assert too_large.value.status_code == 413

        with pytest.raises(UploadValidationError) as bad_type:
            await save_validated_upload(_AsyncUpload(b"not a pdf"), max_bytes=1024)
        assert bad_type.value.status_code == 415

    asyncio.run(run_checks())


def test_process_issues_opaque_text_artifact_ids_and_prevents_client_id_lookup(
    tmp_path,
):
    class DummyPipeline:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, input_path, output_path, **kwargs):
            Path(output_path).write_bytes(b"%PDF-1.4\n%%EOF\n")
            return {0: ["safe text"]}

    client = _api_client()

    with (
        patch("pdf_ocr.utils.security.socket.getaddrinfo", side_effect=_public_dns),
        patch("pdf_ocr.api.routers.ocr.OCRPipeline", DummyPipeline),
        patch(
            "pdf_ocr.api.routers.ocr.OCRProcessor",
            lambda *args, **kwargs: SimpleNamespace(),
        ),
        patch(
            "pdf_ocr.api.routers.ocr.HybridAligner",
            lambda *args, **kwargs: SimpleNamespace(),
        ),
        patch(
            "pdf_ocr.api.routers.ocr.PDFHandler",
            lambda *args, **kwargs: SimpleNamespace(),
        ),
    ):
        first = client.post(
            "/process", data=_process_form(), files={"file": _pdf_upload()}
        )
        second = client.post(
            "/process", data=_process_form(), files={"file": _pdf_upload()}
        )

    assert first.status_code == 200
    assert second.status_code == 200
    first_id = first.headers["X-Text-Artifact-Id"]
    second_id = second.headers["X-Text-Artifact-Id"]
    assert first_id != second_id
    assert len(first_id) == 32

    assert client.get("/text/same-client").status_code == 404
    text_response = client.get(f"/text/{first_id}")
    assert text_response.status_code == 200
    assert text_response.json() == {"0": ["safe text"]}


def test_translate_error_response_does_not_expose_internal_exception():
    async def fail_completion(*args, **kwargs):
        raise RuntimeError("secret-api-key leaked by provider")

    client = _api_client()
    with (
        patch("pdf_ocr.utils.security.socket.getaddrinfo", side_effect=_public_dns),
        patch("litellm.acompletion", fail_completion),
    ):
        response = client.post(
            "/api/translate",
            json={
                "text": "hello",
                "target_language": "Spanish",
                "api_base": "http://api.openai.com/v1",
                "model": "openai/test-model",
                "api_key": "secret-api-key",
            },
        )

    assert response.status_code == 500
    payload = json.dumps(response.json())
    assert "secret-api-key" not in payload
    assert "provider" not in payload


def test_static_js_has_no_html_injection_sinks():
    static_js = Path("src/pdf_ocr/static/js")
    for path in static_js.glob("*.js"):
        source = path.read_text(encoding="utf-8")
        assert "innerHTML" not in source
        assert "insertAdjacentHTML" not in source
        assert "outerHTML" not in source
