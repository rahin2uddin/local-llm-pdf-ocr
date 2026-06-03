from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pdf_ocr.api.services.jobs import JobHistory
from pdf_ocr.api.services.progress import (
    ProgressService,
    sanitize_display_client_id,
    stage_to_percent,
    validate_channel_id,
    validate_stage,
)


class Clock:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> datetime:
        self.value += 1
        return datetime(2026, 1, 1, 0, 0, self.value, tzinfo=timezone.utc)


def test_job_history_caps_fifo_and_lists_newest_first() -> None:
    history = JobHistory(max_jobs=3, now=Clock())

    for index in range(5):
        history.record(
            job_id=f"job-{index}",
            filename=f"file-{index}.pdf",
            model="vlm",
            pipeline_mode="hybrid",
            pages=None,
            duration_s=1.234,
            status="complete",
        )

    records = history.list()
    assert [record["id"] for record in records] == ["job-4", "job-3", "job-2"]
    assert records[0] == {
        "id": "job-4",
        "filename": "file-4.pdf",
        "model": "vlm",
        "pipeline_mode": "hybrid",
        "pages": None,
        "duration_s": 1.23,
        "timestamp": "2026-01-01T00:00:05+00:00",
        "status": "complete",
    }


def test_job_history_clear_is_idempotent() -> None:
    history = JobHistory(max_jobs=2, now=Clock())
    history.record(
        job_id="job-1",
        filename="file.pdf",
        model="vlm",
        pipeline_mode="grounded",
        pages="1-2",
        duration_s=0,
        status="error",
    )

    history.clear()
    history.clear()

    assert history.list() == []


def test_job_history_rejects_invalid_boundary_values() -> None:
    history = JobHistory()

    with pytest.raises(ValueError, match="filename"):
        history.record(
            job_id="job-1",
            filename=" ",
            model="vlm",
            pipeline_mode="hybrid",
            pages=None,
            duration_s=1,
            status="complete",
        )

    with pytest.raises(ValueError, match="duration_s"):
        history.record(
            job_id="job-1",
            filename="file.pdf",
            model="vlm",
            pipeline_mode="hybrid",
            pages=None,
            duration_s=-1,
            status="complete",
        )

    with pytest.raises(ValueError, match="status"):
        history.record(
            job_id="job-1",
            filename="file.pdf",
            model="vlm",
            pipeline_mode="hybrid",
            pages=None,
            duration_s=1,
            status="pending",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("stage", "current", "total", "expected"),
    [
        ("convert", 1, 2, 7),
        ("detect", 1, 2, 20),
        ("ocr", 1, 2, 50),
        ("refine", 1, 2, 82),
        ("embed", 1, 2, 95),
        ("unknown", 1, 2, 50),
        ("ocr", 20, 10, 75),
        ("ocr", 1, 0, 25),
        ("ocr", -5, 10, 25),
    ],
)
def test_stage_to_percent_bounds_and_defaults(
    stage: str, current: int, total: int, expected: int
) -> None:
    assert stage_to_percent(stage, current, total) == expected


def test_validate_stage_is_strict() -> None:
    assert validate_stage(" OCR ") == "ocr"

    with pytest.raises(ValueError, match="stage"):
        validate_stage("unknown")


def test_progress_channel_values_are_opaque_and_validated() -> None:
    service = ProgressService()

    channel = service.create_channel(display_client_id="visible-client_1")

    assert channel.display_client_id == "visible-client_1"
    assert service.validate_channel_id(channel.channel_id) == channel.channel_id
    assert (
        service.validate_session_token(channel.session_token) == channel.session_token
    )
    assert channel.channel_id != channel.display_client_id
    assert channel.session_token != channel.display_client_id


@pytest.mark.parametrize(
    "value",
    [
        "",
        "short",
        "client id with spaces",
        "../client",
        "client:123",
        "a" * 129,
    ],
)
def test_invalid_channel_values_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="channel_id"):
        validate_channel_id(value)


def test_display_client_id_is_sanitized_but_not_used_as_channel() -> None:
    assert sanitize_display_client_id(" client-1_2 ") == "client-1_2"
    assert sanitize_display_client_id(" ") is None

    with pytest.raises(ValueError, match="display client ID"):
        sanitize_display_client_id("client id")


def test_token_binding_requires_matching_channel_and_session() -> None:
    service = ProgressService()
    channel = service.create_channel()
    other = service.create_channel()

    assert service.is_bound(
        channel_id=channel.channel_id,
        session_token=channel.session_token,
        expected_channel_id=channel.channel_id,
        expected_session_token=channel.session_token,
    )
    assert not service.is_bound(
        channel_id=channel.channel_id,
        session_token=other.session_token,
        expected_channel_id=channel.channel_id,
        expected_session_token=channel.session_token,
    )
    assert not service.is_bound(
        channel_id=other.channel_id,
        session_token=channel.session_token,
        expected_channel_id=channel.channel_id,
        expected_session_token=channel.session_token,
    )
