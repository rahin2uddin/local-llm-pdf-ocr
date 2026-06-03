import os
from collections.abc import Callable
from typing import Any

from pdf_ocr.core.translation_config import AsyncTranslationUnavailable

try:
    from celery import Celery as CeleryClass
except ImportError:
    CeleryClass = None

# Allow configuration via environment variables
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class _MissingCeleryTask:
    def __init__(self, func: Callable[..., Any]) -> None:
        self._func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(self, *args, **kwargs)

    def delay(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AsyncTranslationUnavailable(
            "Async translation requires optional dependency 'celery'. "
            "Install the async translation extras to enable background jobs."
        )

    def update_state(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _MissingCeleryApp:
    def task(self, *_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
        def decorator(func: Callable[..., Any]) -> _MissingCeleryTask:
            return _MissingCeleryTask(func)

        return decorator

    def AsyncResult(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AsyncTranslationUnavailable(
            "Async translation status requires optional dependency 'celery'. "
            "Install the async translation extras to enable background jobs."
        )


if CeleryClass is None:
    celery_app = _MissingCeleryApp()
else:
    # Initialize Celery app
    celery_app = CeleryClass(
        "pdf_ocr_tasks",
        broker=REDIS_URL,
        backend=REDIS_URL,
        include=["pdf_ocr.api.tasks"],
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
