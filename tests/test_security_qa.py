"""Unit tests verifying security patches, QA bug fixes, and translation chunking."""

from __future__ import annotations

import asyncio
import json
import os
import socket
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pdf_ocr.api.tasks import process_translation_task
from pdf_ocr.core.translation import chunk_text, evaluate_node
from pdf_ocr.utils.security import is_ssrf_target


def test_is_ssrf_target_defaults():
    # By default, ALLOW_SSRF_LOCAL is "true" in config for local development
    # But when ALLOW_SSRF_LOCAL is "false", let's verify SSRF catches local addresses
    with patch.dict(os.environ, {"ALLOW_SSRF_LOCAL": "false"}):
        with patch("socket.getaddrinfo") as mock_getaddrinfo:

            def side_effect(host, port, *args, **kwargs):
                if host in (
                    "localhost",
                    "127.0.0.1",
                    "192.168.1.1",
                    "10.0.0.1",
                    "127.0.0.1.nip.io",
                ):
                    return [
                        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
                    ]
                elif "openai.com" in host:
                    return [
                        (
                            socket.AF_INET,
                            socket.SOCK_STREAM,
                            6,
                            "",
                            ("104.18.3.161", 80),
                        )
                    ]
                else:
                    raise socket.gaierror(-2, "Name or service not known")

            mock_getaddrinfo.side_effect = side_effect

            assert is_ssrf_target("http://localhost:1234/v1") is True
            assert is_ssrf_target("http://127.0.0.1/v1") is True
            assert is_ssrf_target("http://192.168.1.1/v1") is True
            assert is_ssrf_target("http://10.0.0.1/v1") is True
            assert is_ssrf_target("http://127.0.0.1.nip.io/v1") is True
            # Public resources should pass cleanly
            assert is_ssrf_target("http://api.openai.com/v1") is False


def test_is_ssrf_target_allowed():
    # If ALLOW_SSRF_LOCAL is explicitly set to true
    with patch.dict(os.environ, {"ALLOW_SSRF_LOCAL": "true"}):
        assert is_ssrf_target("http://localhost:1234/v1") is False
        assert is_ssrf_target("http://127.0.0.1/v1") is False


def test_translation_chunking_preserves_size():
    # Generate text larger than 4000 characters
    long_text = "Paragraph one.\n\n" * 400
    assert len(long_text) > 4000

    chunks = chunk_text(long_text, max_chunk_size=4000)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 4000
        assert chunk.strip() != ""


def test_evaluate_node_minimal_input_skips_loop():
    # Minimal punctuation-only input should not trigger loops
    punctuation_state = {
        "source_chunk": ".",
        "target_language": "English",
        "rag_context": [],
        "translated_chunk": "",
        "evaluation_score": 1.0,
        "feedback": "",
        "attempts": 1,
    }

    result = evaluate_node(punctuation_state)
    assert result["evaluation_score"] == 1.0
    assert result["feedback"] == "Looks good"


def test_evaluate_node_normal_input_fails_correctly():
    # Normal input that is translated too shortly should fail evaluation
    bad_state = {
        "source_chunk": "This is a much longer sentence that deserves translation.",
        "target_language": "Spanish",
        "rag_context": [],
        "translated_chunk": "",  # Empty translation
        "evaluation_score": 1.0,
        "feedback": "",
        "attempts": 1,
    }

    result = evaluate_node(bad_state)
    assert result["evaluation_score"] == 0.0
    assert "too short" in result["feedback"]


def test_celery_task_raises_value_error_on_translation_error():
    # Task should raise ValueError on translation errors
    # With bind=True, Celery's .run() method automatically binds the task instance to 'self'.
    # We patch 'update_state' to prevent it from complaining about missing task context during test run.
    with patch.object(process_translation_task, "update_state"):
        with patch(
            "pdf_ocr.core.translation.run_translation",
            return_value="[Translation Error: Connection refused]",
        ):
            with pytest.raises(ValueError) as exc_info:
                process_translation_task.run("doc_123", "Hello World")
            assert "Translation failed" in str(exc_info.value)


def test_extract_data_robust_json_parsing():
    pytest.importorskip("fastapi")
    from pdf_ocr.api.routers import ocr

    # Verify our custom regex fallback in ocr.py doesn't crash when JSON matches are missing or fail
    async def mock_acompletion(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Bad model output with no JSON")
                )
            ]
        )

    with (
        patch.object(ocr.json, "loads") as mock_loads,
        patch.object(ocr.re, "search") as mock_search,
        patch("litellm.acompletion", mock_acompletion),
    ):
        mock_loads.side_effect = json.JSONDecodeError("JSON Decode Error", "", 0)
        mock_search.return_value = None  # No matching bracket/braces found

        # We call the FastAPI handler synchronously via standard coroutine run
        with patch("pdf_ocr.utils.security.socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.18.3.161", 443))
            ]
            response = asyncio.run(
                ocr.extract_data(
                    {
                        "text": "Hello World",
                        "template": "invoice",
                        "api_base": "http://api.openai.com/v1",
                    }
                )
            )

        # Verify it handled the error gracefully and returned empty extracted_data rather than raising exception
        assert isinstance(response, dict)
        assert response["extracted_data"] == {}
