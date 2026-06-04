from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal, cast

JobStatus = Literal["complete", "error", "rejected"]
_MAX_TEXT_LENGTH = 512


@dataclass(frozen=True)
class JobRecord:
    """Deterministic API shape for a completed or rejected OCR job."""

    id: str
    filename: str
    model: str
    pipeline_mode: str
    pages: str | None
    duration_s: float
    timestamp: str
    status: JobStatus

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
