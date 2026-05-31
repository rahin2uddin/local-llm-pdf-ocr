import logging

from pdf_ocr.api.celery_app import celery_app

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="process_translation")
def process_translation_task(self, document_id: str, document_text: str):
    """
    Background task to run the LangGraph translation workflow on the extracted text.
    """
    logger.info(f"Starting translation task for document_id={document_id}")

    # Update state to started
    self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Started LangGraph Translation'})

    from pdf_ocr.core.translation import run_translation

    # Run translation
    translated_text = run_translation(document_text)

    return {"document_id": document_id, "translated_text": translated_text}
