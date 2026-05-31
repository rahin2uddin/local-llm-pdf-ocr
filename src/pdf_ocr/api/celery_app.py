import os

from celery import Celery

# Allow configuration via environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Initialize Celery app
celery_app = Celery(
    "pdf_ocr_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["pdf_ocr.api.tasks"]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # To prevent OOM errors on a single GPU setup (12GB VRAM), we force a single worker process
    worker_concurrency=1,
)
