import logging

from pdf_ocr.api.celery_app import celery_app
from pdf_ocr.core.translation_config import TranslationSettings

logger = logging.getLogger(__name__)


def _current_translation_settings() -> TranslationSettings:
    """Use mutable web settings when available, otherwise environment settings."""
    try:
        from pdf_ocr.api.routers.config import get_translation_settings
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".", maxsplit=1)[0] == "fastapi":
            return TranslationSettings.from_env()
        raise

    return get_translation_settings()


@celery_app.task(bind=True, name="process_translation")
def process_translation_task(self, document_id: str, document_text: str):
    """
    Background task to run the LangGraph translation workflow on the extracted text.
    """
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")
    if not isinstance(document_text, str):
        raise ValueError("document_text must be a string")

    logger.info(f"Starting translation task for document_id={document_id}")

    # Update state to started
    self.update_state(
        state="PROGRESS",
        meta={"progress": 0, "status": "Started LangGraph Translation"},
    )

    from pdf_ocr.core.translation import run_translation

    # Run translation
    translated_text = run_translation(
        document_text,
        settings=_current_translation_settings(),
    )

    if translated_text.startswith("[Translation Error:"):
        raise ValueError(f"Translation failed: {translated_text}")

    return {"document_id": document_id, "translated_text": translated_text}
