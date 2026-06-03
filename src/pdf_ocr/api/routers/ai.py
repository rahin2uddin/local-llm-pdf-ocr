from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from pdf_ocr.api.schemas import ExtractionRequest, TranslationRequest
from pdf_ocr.api.services.ai import (
    AIServiceError,
    extract_structured_data,
)
from pdf_ocr.api.services.ai import (
    translate_text as translate_document_text,
)
from pdf_ocr.api.services.security import SERVER_ERROR_MESSAGE

from .config import _config

router = APIRouter()
logger = logging.getLogger(__name__)


def _stable_server_error(status_code: int = 500) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"error": SERVER_ERROR_MESSAGE}
    )


def _ai_error_response(exc: AIServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.public_message},
    )


@router.post("/api/translate")
async def translate_text(body: TranslationRequest):
    """Translate OCR text into the requested target language."""
    try:
        translated = await translate_document_text(body, config=_config)
    except AIServiceError as exc:
        return _ai_error_response(exc)
    except Exception:
        logger.exception("Translation request failed")
        return _stable_server_error()
    return {"translated_text": translated}


@router.post("/api/extract")
async def extract_data(body: ExtractionRequest):
    """Extract structured JSON data from OCR text."""
    try:
        extracted = await extract_structured_data(body, config=_config)
    except AIServiceError as exc:
        return _ai_error_response(exc)
    except Exception:
        logger.exception("Extraction request failed")
        return _stable_server_error()
    return {"extracted_data": extracted}


@router.post("/api/translate/async")
async def translate_text_async(body: dict[str, Any]):
    """Trigger a background translation job via Celery."""
    from pdf_ocr.api.tasks import process_translation_task

    raw_document_id = body.get("document_id") or uuid.uuid4().hex
    document_id = str(raw_document_id).strip() or uuid.uuid4().hex
    raw_text = body.get("text", "")
    text = raw_text if isinstance(raw_text, str) else ""

    try:
        task = process_translation_task.delay(document_id, text)
    except Exception:
        logger.exception("Async translation dispatch failed")
        return _stable_server_error()
    return {"job_id": task.id, "status": "Processing"}


@router.get("/api/translate/status/{job_id}")
async def get_translation_status(job_id: str):
    """Poll the status of a Celery background translation job."""
    from pdf_ocr.api.celery_app import celery_app

    try:
        task = celery_app.AsyncResult(job_id)
        response: dict[str, Any] = {
            "job_id": job_id,
            "state": task.state,
        }

        if task.state == "PENDING":
            response["status"] = "Pending..."
        elif task.state != "FAILURE":
            response["info"] = task.info
            if task.state == "SUCCESS":
                response["result"] = task.get()
        else:
            logger.error("Async translation task failed: %s", task.info)
            response["error"] = SERVER_ERROR_MESSAGE

        return response
    except Exception:
        logger.exception("Async translation status lookup failed")
        return _stable_server_error()
