from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from pdf_ocr.api.schemas.requests import (
    ExtractionRequest,
    ExtractionTemplate,
    TranslationRequest,
)
from pdf_ocr.api.services import ai


def _config() -> dict[str, object]:
    return {
        "api_base": "http://config.example/v1",
        "api_key": "config-key",
        "model": "config-model",
    }


def _completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


async def test_translate_uses_request_settings_and_builds_prompt():
    calls: list[dict[str, Any]] = []

    async def capture_completion(**kwargs):
        calls.append(kwargs)
        return _completion("  Traduccion directa  ")

    request = TranslationRequest(
        text="# Hello\n\nA table follows.",
        target_language="Spanish",
        api_base="http://request.example/v1",
        api_key="request-key",
        model="custom-local-model",
    )

    with (
        patch("pdf_ocr.api.services.ai.is_ssrf_target", return_value=False),
        patch("litellm.acompletion", capture_completion),
    ):
        translated = await ai.translate_text(request, config=_config())

    assert translated == "Traduccion directa"
    assert len(calls) == 1
    assert calls[0]["api_base"] == "http://request.example/v1"
    assert calls[0]["api_key"] == "request-key"
    assert calls[0]["model"] == "custom-local-model"
    assert calls[0]["custom_llm_provider"] == "openai"
    assert calls[0]["temperature"] == 0.3
    prompt = calls[0]["messages"][0]["content"]
    assert "Translate the following document text into Spanish" in prompt
    assert "Maintain all markdown formatting" in prompt
    assert "TEXT:\n# Hello" in prompt


async def test_extract_uses_template_prompt_and_config_defaults():
    calls: list[dict[str, Any]] = []

    async def capture_completion(**kwargs):
        calls.append(kwargs)
        return _completion(
            '```json\n{"vendor_name": "Acme", "total_amount": "10.00"}\n```'
        )

    request = ExtractionRequest(
        text="Invoice from Acme", template=ExtractionTemplate.INVOICE
    )

    with (
        patch("pdf_ocr.api.services.ai.is_ssrf_target", return_value=False),
        patch("litellm.acompletion", capture_completion),
    ):
        extracted = await ai.extract_structured_data(request, config=_config())

    assert extracted == {"vendor_name": "Acme", "total_amount": "10.00"}
    assert calls[0]["api_base"] == "http://config.example/v1"
    assert calls[0]["api_key"] == "config-key"
    assert calls[0]["model"] == "config-model"
    assert calls[0]["temperature"] == 0.1
    prompt = calls[0]["messages"][0]["content"]
    assert "Extract standard invoice fields" in prompt
    assert "Output the results STRICTLY as a single valid JSON object" in prompt
    assert "DOCUMENT TEXT:\nInvoice from Acme" in prompt


async def test_extract_invalid_json_returns_empty_object():
    async def bad_completion(**kwargs):
        return _completion("Bad model output with no JSON")

    request = ExtractionRequest(
        text="Invoice text",
        template=ExtractionTemplate.INVOICE,
        api_base="http://request.example/v1",
    )

    with (
        patch("pdf_ocr.api.services.ai.is_ssrf_target", return_value=False),
        patch("litellm.acompletion", bad_completion),
    ):
        assert await ai.extract_structured_data(request, config=_config()) == {}


def test_parse_extraction_json_accepts_fenced_and_embedded_objects():
    assert ai.parse_extraction_json('```json\n{"invoice_number": "A-1"}\n```') == {
        "invoice_number": "A-1"
    }
    assert ai.parse_extraction_json('prefix text {"total": 12} suffix') == {"total": 12}


async def test_ssrf_blocking_is_distinct_and_skips_provider_call():
    async def unexpected_completion(**kwargs):
        raise AssertionError("provider should not be called")

    request = TranslationRequest(text="hello", api_base="http://127.0.0.1:1234/v1")

    with (
        patch("pdf_ocr.api.services.ai.is_ssrf_target", return_value=True),
        patch("litellm.acompletion", unexpected_completion),
        pytest.raises(ai.BlockedAPIBaseError) as exc_info,
    ):
        await ai.translate_text(request, config=_config())

    assert exc_info.value.status_code == 403
    assert exc_info.value.public_message == ai.SAFE_API_BASE_ERROR


async def test_provider_failure_wraps_without_public_detail_leak():
    async def fail_completion(**kwargs):
        raise RuntimeError("secret-api-key leaked by provider")

    request = TranslationRequest(
        text="hello",
        api_base="http://request.example/v1",
        api_key="secret-api-key",
        model="custom-local-model",
    )

    with (
        patch("pdf_ocr.api.services.ai.is_ssrf_target", return_value=False),
        patch("litellm.acompletion", fail_completion),
        pytest.raises(ai.AIProviderError) as exc_info,
    ):
        await ai.translate_text(request, config=_config())

    public_payload = f"{exc_info.value.public_message} {exc_info.value}"
    assert exc_info.value.status_code == 500
    assert ai.SERVER_ERROR_MESSAGE in public_payload
    assert "secret-api-key" not in public_payload
    assert "provider" not in public_payload
