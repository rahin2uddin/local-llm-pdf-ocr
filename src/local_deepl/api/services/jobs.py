from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, cast

JobStatus = Literal["complete", "error", "rejected"]
_MAX_TEXT_LENGTH = 512
_MAX_FAILED_PAGES = 10_000


@dataclass(frozen=True)
class JobRecord:
    """Deterministic API shape for a completed or rejected OCR job.

    ``failed_pages`` lists 0-indexed page numbers whose OCR pipeline raised
    an exception and was caught at the per-page isolation boundary. A job
    with a non-empty ``failed_pages`` still has ``status="complete"`` —
    the pipeline degrades gracefully and writes a PDF with empty
    searchable text on the failed pages, so the rest of the document
    is still useful. The list is omitted from :meth:`to_dict` when empty
    to preserve the wire format for the common no-failure case.
    """

    id: str
    filename: str
    model: str
    pipeline_mode: str
    pages: str | None
    duration_s: float
    timestamp: str
    status: JobStatus
    failed_pages: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "filename": self.filename,
            "model": self.model,
            "pipeline_mode": self.pipeline_mode,
            "pages": self.pages,
            "duration_s": self.duration_s,
            "timestamp": self.timestamp,
            "status": self.status,
        }
        if self.failed_pages:
            payload["failed_pages"] = list(self.failed_pages)
        return payload


class JobHistory:
    """Capped in-memory FIFO history with newest-first reads."""

    def __init__(
        self,
        *,
        max_jobs: int = 50,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(max_jobs, int) or max_jobs < 1:
            raise ValueError("max_jobs must be a positive integer.")
        self._records: deque[JobRecord] = deque(maxlen=max_jobs)
        self._now = now or (lambda: datetime.now(timezone.utc))

    @property
    def max_jobs(self) -> int:
        return self._records.maxlen or 0

    def record(
        self,
        *,
        job_id: str,
        filename: str,
        model: str,
        pipeline_mode: str,
        pages: str | None,
        duration_s: float,
        status: JobStatus,
        failed_pages: Sequence[int] = (),
    ) -> JobRecord:
        record = JobRecord(
            id=_clean_required_text(job_id, "job_id"),
            filename=_clean_required_text(filename, "filename"),
            model=_clean_required_text(model, "model"),
            pipeline_mode=_clean_required_text(pipeline_mode, "pipeline_mode"),
            pages=_clean_optional_text(pages, "pages"),
            duration_s=_clean_duration(duration_s),
            timestamp=_current_timestamp(self._now),
            status=_clean_status(status),
            failed_pages=_clean_failed_pages(failed_pages),
        )
        self._records.append(record)
        return record

    def list(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in reversed(self._records)]

    def clear(self) -> None:
        self._records.clear()


def _current_timestamp(now: Callable[[], datetime]) -> str:
    timestamp = now()
    if not isinstance(timestamp, datetime):
        raise TypeError("now must return a datetime.")
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat()


def _clean_required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty.")
    if len(cleaned) > _MAX_TEXT_LENGTH:
        raise ValueError(f"{field_name} exceeds {_MAX_TEXT_LENGTH} characters.")
    return cleaned


def _clean_optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_required_text(value, field_name)


def _clean_duration(value: float) -> float:
    if not isinstance(value, int | float):
        raise TypeError("duration_s must be numeric.")
    duration = float(value)
    if duration < 0:
        raise ValueError("duration_s must not be negative.")
    return round(duration, 2)


def _clean_status(value: str) -> JobStatus:
    if value not in {"complete", "error", "rejected"}:
        raise ValueError("status must be one of: complete, error, rejected.")
    return cast(JobStatus, value)


def _clean_failed_pages(value: Sequence[int]) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("failed_pages must be a sequence of integers.")
    cleaned: list[int] = []
    seen: set[int] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError("failed_pages entries must be integers.")
        if item < 0:
            raise ValueError("failed_pages entries must be non-negative.")
        if item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
        if len(cleaned) > _MAX_FAILED_PAGES:
            raise ValueError(f"failed_pages exceeds {_MAX_FAILED_PAGES} entries.")
    return tuple(cleaned)
