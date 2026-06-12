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

from local_deepl.api.routers import ai, config, ocr, websocket, jobs, artifacts, translation, extraction, state
from local_deepl.api.services.artifacts import TextArtifactStore
from local_deepl.api.services.security import (
    UploadValidationError,
    save_validated_upload,
)
from local_deepl.core.document import DocumentResult
from local_deepl.utils.security import is_ssrf_target


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
    app.include_router(ai.router)
    app.include_router(ocr.router)
    app.include_router(websocket.router)
    app.include_router(jobs.router)
    app.include_router(artifacts.router)
    app.include_router(translation.router)
    app.include_router(extraction.router)
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
        "preprocess_pages": "false",
        "orientation_detection": "false",
        "deskew": "false",
        "denoise": "false",
        "normalize_contrast": "false",
        "crop_cleanup": "false",
        "quality_routing": "false",
    }


def _pdf_upload() -> tuple[str, bytes, str]:
    return ("input.pdf", b"%PDF-1.4\n%%EOF\n", "application/pdf")


def test_ssrf_fails_closed_and_requires_explicit_local_allowance():
    with patch.dict(os.environ, {}, clear=True):
        with patch("local_deepl.utils.security.socket.getaddrinfo") as getaddrinfo:
            getaddrinfo.side_effect = _public_dns
            assert is_ssrf_target("http://api.openai.com/v1") is False
            assert is_ssrf_target("localhost:1234/v1") is True
            assert is_ssrf_target("ftp://api.openai.com/v1") is True
            assert is_ssrf_target(None) is True

    with patch.dict(os.environ, {}, clear=True):
        with patch("local_deepl.utils.security.socket.getaddrinfo") as getaddrinfo:
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
            self.last_failed_pages: list[int] = []

        async def run(self, input_path, output_path, **kwargs):
            Path(output_path).write_bytes(b"%PDF-1.4\n%%EOF\n")
            return {0: ["safe text"]}

    client = _api_client()

    with (
        patch("local_deepl.utils.security.socket.getaddrinfo", side_effect=_public_dns),
        patch("local_deepl.api.routers.ocr.OCRPipeline", DummyPipeline),
        patch(
            "local_deepl.api.routers.ocr.OCRProcessor",
            lambda *args, **kwargs: SimpleNamespace(),
        ),
        patch(
            "local_deepl.api.routers.ocr.HybridAligner",
            lambda *args, **kwargs: SimpleNamespace(),
        ),
        patch(
            "local_deepl.api.routers.ocr.PDFHandler",
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
    first_token = first.headers["X-Text-Artifact-Token"]
    second_token = second.headers["X-Text-Artifact-Token"]
    assert first_id != second_id
    assert first_token != second_token
    assert len(first_id) == 32

    assert (
        client.get(
            "/text/same-client",
            headers={"Authorization": f"Bearer {first_token}"},
        ).status_code
        == 404
    )
    assert client.get(f"/text/{first_id}").status_code == 403
    assert (
        client.get(
            f"/text/{first_id}",
            headers={"Authorization": f"Bearer {second_token}"},
        ).status_code
        == 403
    )
    text_response = client.get(
        f"/text/{first_id}",
        headers={"Authorization": f"Bearer {first_token}"},
    )
    assert text_response.status_code == 200
    assert text_response.json() == {"0": ["safe text"]}


def test_text_artifact_retrieval_expires_router_store(tmp_path):
    clock = SimpleNamespace(value=0.0)

    def now() -> float:
        return clock.value

    original_store = state.text_artifacts
    try:
        store = TextArtifactStore(ttl_seconds=5, clock=now, artifact_dir=tmp_path)
        state.text_artifacts = store
        handle = store.create({0: ["expiring text"]})
        client = _api_client()

        response = client.get(
            f"/text/{handle.artifact_id}",
            headers={"Authorization": f"Bearer {handle.token}"},
        )
        assert response.status_code == 200

        clock.value = 6.0
        response = client.get(
            f"/text/{handle.artifact_id}",
            headers={"Authorization": f"Bearer {handle.token}"},
        )
        assert response.status_code == 404
        assert not Path(handle.path).exists()
    finally:
        state.text_artifacts = original_store


def test_process_omits_document_metadata_artifact_when_no_report(tmp_path: Path):
    class DummyPipeline:
        def __init__(self, *args, **kwargs):
            self.last_document_result = None
            self.last_failed_pages: list[int] = []

        async def run(self, input_path, output_path, **kwargs):
            Path(output_path).write_bytes(b"%PDF-1.4\n%%EOF\n")
            return {0: ["safe text"]}

    original_text_store = state.text_artifacts
    original_metadata_store = state.metadata_artifacts
    state.text_artifacts = TextArtifactStore(artifact_dir=tmp_path / "text")
    state.metadata_artifacts = TextArtifactStore(artifact_dir=tmp_path / "metadata")

    try:
        client = _api_client()
        with (
            patch(
                "local_deepl.utils.security.socket.getaddrinfo",
                side_effect=_public_dns,
            ),
            patch("local_deepl.api.routers.ocr.OCRPipeline", DummyPipeline),
            patch("local_deepl.api.routers.ocr.HybridAligner"),
            patch("local_deepl.api.routers.ocr.PDFHandler"),
        ):
            response = client.post(
                "/process", data=_process_form(), files={"file": _pdf_upload()}
            )

        assert response.status_code == 200
        assert "X-Text-Artifact-Id" in response.headers
        assert "X-Document-Metadata-Artifact-Id" not in response.headers
        assert "X-Document-Metadata-Artifact-Token" not in response.headers
    finally:
        state.text_artifacts = original_text_store
        state.metadata_artifacts = original_metadata_store


def test_process_exposes_token_bound_document_metadata_artifact(tmp_path: Path):
    class DummyPipeline:
        def __init__(self, *args, **kwargs):
            self.last_document_result = None
            self.last_failed_pages: list[int] = []

        async def run(self, input_path, output_path, **kwargs):
            Path(output_path).write_bytes(b"%PDF-1.4\n%%EOF\n")
            document = DocumentResult.from_pages_data(
                {0: [([0.1, 0.1, 0.4, 0.2], "Invoice")]}
            )
            page = document.pages[0]
            block = page.blocks[0]
            block.reading_order = 0
            block.kind = "heading"
            block.metadata["structure"] = {
                "kind": "heading",
                "confidence": 0.9,
                "signals": ["test_heading"],
            }
            block.metadata["section"] = {
                "section_index": 0,
                "title": "Invoice",
                "heading_page_index": 0,
                "heading_block_index": 0,
            }
            page.metadata["quality"] = {
                "block_count": 1,
                "text_char_count": 7,
                "text_density": 70.0,
                "findings": [],
            }
            page.metadata["structure"] = {
                "block_kinds": {"heading": 1},
                "has_key_values": False,
                "has_tables": False,
            }
            page.metadata["sections"] = {
                "section_count": 1,
                "active_section": "Invoice",
                "headings": [block.metadata["section"]],
            }
            self.last_document_result = document
            return {0: ["safe text"]}

    original_text_store = state.text_artifacts
    original_metadata_store = state.metadata_artifacts
    state.text_artifacts = TextArtifactStore(artifact_dir=tmp_path / "text")
    state.metadata_artifacts = TextArtifactStore(artifact_dir=tmp_path / "metadata")

    try:
        client = _api_client()
        with (
            patch(
                "local_deepl.utils.security.socket.getaddrinfo",
                side_effect=_public_dns,
            ),
            patch("local_deepl.api.routers.ocr.OCRPipeline", DummyPipeline),
            patch("local_deepl.api.routers.ocr.HybridAligner"),
            patch("local_deepl.api.routers.ocr.PDFHandler"),
        ):
            response = client.post(
                "/process", data=_process_form(), files={"file": _pdf_upload()}
            )

        assert response.status_code == 200
        artifact_id = response.headers["X-Document-Metadata-Artifact-Id"]
        token = response.headers["X-Document-Metadata-Artifact-Token"]

        denied = client.get(
            f"/metadata/{artifact_id}",
            headers={"Authorization": f"Bearer {'A' * 43}"},
        )
        assert denied.status_code == 403

        metadata_response = client.get(
            f"/metadata/{artifact_id}", headers={"Authorization": f"Bearer {token}"}
        )
        assert metadata_response.status_code == 200
        payload = metadata_response.json()

        assert payload["summary"]["processors"] == [
            "quality_analysis",
            "reading_order",
            "section_analysis",
            "structure_analysis",
        ]
        assert payload["pages"][0]["metadata"]["quality"]["block_count"] == 1
        block_report = payload["pages"][0]["blocks"][0]
        assert block_report["reading_order"] == 0
        assert block_report["metadata"]["structure"]["kind"] == "heading"
        assert "text" not in block_report
    finally:
        state.text_artifacts = original_text_store
        state.metadata_artifacts = original_metadata_store


def test_document_export_artifact_is_token_bound(tmp_path: Path):
    original_text_store = state.text_artifacts
    original_export_store = state.export_artifacts
    state.text_artifacts = TextArtifactStore(artifact_dir=tmp_path / "text")
    state.export_artifacts = TextArtifactStore(artifact_dir=tmp_path / "export")

    try:
        handle = state.text_artifacts.create({0: ["alpha", "beta"]})
        client = _api_client()
        response = client.post(
            "/api/export/document",
            json={
                "text_artifact_id": handle.artifact_id,
                "text_artifact_token": handle.token,
                "export_format": "markdown",
            },
        )
        assert response.status_code == 200
        body = response.json()

        denied = client.get(
            f"/export/{body['artifact_id']}",
            headers={"Authorization": f"Bearer {'A' * 43}"},
        )
        assert denied.status_code == 403

        exported = client.get(
            f"/export/{body['artifact_id']}",
            headers={"Authorization": f"Bearer {body['token']}"},
        )
        assert exported.status_code == 200
        assert exported.text.startswith("## Page 1")
    finally:
        state.text_artifacts = original_text_store
        state.export_artifacts = original_export_store


def test_progress_session_uses_token_bound_websocket_channels():
    client = _api_client()

    session_response = client.post(
        "/api/progress/session", json={"client_id": "visible-client"}
    )
    assert session_response.status_code == 200
    session = session_response.json()
    assert session["channel_id"] != "visible-client"
    assert session["session_token"] != "visible-client"

    with client.websocket_connect(
        f"/ws/{session['channel_id']}?token={session['session_token']}"
    ):
        assert websocket.manager.is_authorized(
            session["channel_id"], session["session_token"]
        )
        assert not websocket.manager.is_authorized(session["channel_id"], "A" * 32)


def test_process_surfaces_partial_page_failures_in_headers_and_history(tmp_path: Path):
    """A page whose OCR call raises must be reported in the response
    ``X-Failed-Pages`` header and the job-history record. The job
    status stays ``"complete"`` — the pipeline degrades gracefully and
    writes a PDF even with bad pages.

    The WebSocket frame shape is covered separately by
    ``test_websocket_manager_emits_warning_flag``; here we just confirm
    the router wires the partial-failure signal through.
    """

    class _FailingDummyPipeline:
        def __init__(self, *args, **kwargs):
            self.last_document_result = None
            self.last_failed_pages: list[int] = [1]  # 0-indexed page 1 fails

        async def run(self, input_path, output_path, **kwargs):
            on_warning = kwargs.get("on_warning")
            if on_warning is not None:
                await on_warning(1, RuntimeError("simulated page 1 failure"))
            Path(output_path).write_bytes(b"%PDF-1.4\n%%EOF\n")
            return {0: ["page0"], 1: [], 2: ["page2"]}

    client = _api_client()

    with (
        patch("local_deepl.utils.security.socket.getaddrinfo", side_effect=_public_dns),
        patch("local_deepl.api.routers.ocr.OCRPipeline", _FailingDummyPipeline),
        patch("local_deepl.api.routers.ocr.HybridAligner"),
        patch("local_deepl.api.routers.ocr.PDFHandler"),
    ):
        response = client.post(
            "/process",
            data=_process_form(),
            files={"file": _pdf_upload()},
        )

    assert response.status_code == 200
    assert response.headers.get("X-Failed-Pages") == "1"

    # The job record reflects the partial failure.
    jobs = client.get("/api/jobs").json()
    assert jobs, "no job history recorded"
    latest = jobs[0]
    assert latest["status"] == "complete"
    assert latest["failed_pages"] == [1]


def test_websocket_manager_emits_warning_flag():
    """The ConnectionManager.send_progress path must serialize the
    ``warning`` flag in the WebSocket frame so the UI can render a
    partial-failure indicator without parsing the message text."""
    from local_deepl.api.routers.websocket import ConnectionManager

    sent_frames: list[dict] = []

    class _StubWS:
        async def accept(self):
            pass

        async def send_json(self, payload):
            sent_frames.append(payload)

    async def _drive():
        manager = ConnectionManager()
        await manager.connect(_StubWS(), "abcd" * 8, "efgh" * 8)  # 32-char tokens
        await manager.send_progress("abcd" * 8, "all good", 50, stage="ocr")
        await manager.send_progress(
            "abcd" * 8,
            "OCR failed for page 7: TimeoutError",
            0,
            stage="ocr",
            warning=True,
        )

    asyncio.run(_drive())

    assert sent_frames[0] == {
        "status": "all good",
        "percent": 50,
        "stage": "ocr",
    }
    assert sent_frames[1] == {
        "status": "OCR failed for page 7: TimeoutError",
        "percent": 0,
        "stage": "ocr",
        "warning": True,
    }


def test_translate_error_response_does_not_expose_internal_exception():
    async def fail_completion(*args, **kwargs):
        raise RuntimeError("secret-api-key leaked by provider")

    client = _api_client()
    with (
        patch("local_deepl.utils.security.socket.getaddrinfo", side_effect=_public_dns),
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
    static_js = Path("src/local_deepl/static/js")
    for path in static_js.glob("*.js"):
        source = path.read_text(encoding="utf-8")
        assert "innerHTML" not in source
        assert "insertAdjacentHTML" not in source
        assert "outerHTML" not in source
