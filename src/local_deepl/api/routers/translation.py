import logging
import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from local_deepl.api.routers.common import _stable_server_error
from local_deepl.api.routers.config import _config
from local_deepl.api.schemas import TranslationRequest
from local_deepl.api.services.security import SAFE_API_BASE_ERROR, SERVER_ERROR_MESSAGE
from local_deepl.core.translation_config import AsyncTranslationUnavailable
from local_deepl.utils import is_ssrf_target

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/translate")
async def translate_text(body: TranslationRequest):
    """
    Translate OCR'ed document text into the specified target language.
    """
    text = body.text
    target_lang = body.target_language

    if not text.strip():
        return {"translated_text": ""}

    active_api_base = body.api_base or _config["api_base"]
    if is_ssrf_target(active_api_base):
        return JSONResponse(status_code=403, content={"error": SAFE_API_BASE_ERROR})
    active_api_key = body.api_key or _config["api_key"]
    active_model = body.model or _config["model"]

    prompt = (
        f"Translate the following document text into {target_lang}. "
        f"Maintain all markdown formatting, headings, lists, tables, and mathematical formulas exactly. "
        f"Do not add any introductory or concluding comments, explanations, or meta-commentary. "
        f"Only output the direct translation.\n\n"
        f"TEXT:\n{text}"
    )

    try:
        import litellm

        from local_deepl.utils.litellm_provider import resolve_custom_provider

        custom_provider = resolve_custom_provider(active_model)

        response = await litellm.acompletion(
            model=active_model,
            custom_llm_provider=custom_provider,
            api_base=active_api_base,
            api_key=active_api_key,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        translated = (response.choices[0].message.content or "").strip()
        return {"translated_text": translated}
    except Exception:
        logger.exception("Translation request failed")
        return _stable_server_error()


@router.post("/api/translate/async")
async def translate_text_async(body: dict):
    """
    Trigger a background translation job via Celery.
    """
    from local_deepl.api.tasks import process_translation_task

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=422,
            content={"error": "Invalid request parameters."},
        )

    document_id = body.get("document_id", str(uuid.uuid4()))
    text = body.get("text", "")
    if not isinstance(document_id, str) or not document_id.strip():
        return JSONResponse(
            status_code=422,
            content={"error": "document_id must be a non-empty string."},
        )
    if not isinstance(text, str):
        return JSONResponse(
            status_code=422,
            content={"error": "text must be a string."},
        )

    # Dispatch Celery task
    try:
        task = process_translation_task.delay(document_id, text)
    except AsyncTranslationUnavailable as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})

    return {"job_id": task.id, "status": "Processing"}


@router.get("/api/translate/status/{job_id}")
async def get_translation_status(job_id: str):
    """
    Poll the status of a Celery background translation job.
    """
    from local_deepl.api.celery_app import celery_app

    try:
        task = celery_app.AsyncResult(job_id)
    except AsyncTranslationUnavailable as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})

    response = {
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
        # something went wrong in the background job
        logger.error("Async translation task failed: %s", task.info)
        response["error"] = SERVER_ERROR_MESSAGE

    return response
