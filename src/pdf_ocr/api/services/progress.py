from __future__ import annotations

import hmac
import re
import secrets
from dataclasses import dataclass
from typing import Final, Literal

Stage = Literal["convert", "detect", "ocr", "refine", "embed"]
CHANNEL_TOKEN_BYTES: Final = 24
SESSION_TOKEN_BYTES: Final = 32
_TOKEN_RE: Final = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
_DISPLAY_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_STAGE_WEIGHTS: Final[dict[Stage, tuple[int, int]]] = {
    "convert": (0, 15),
    "detect": (15, 25),
    "ocr": (25, 75),
    "refine": (75, 90),
    "embed": (90, 100),
}


@dataclass(frozen=True)
class ProgressChannel:
    """Opaque channel and session tokens for websocket/process binding."""

    channel_id: str
    session_token: str
    display_client_id: str | None = None


class ProgressService:
    """Progress math and opaque channel validation."""

    def stage_to_percent(self, stage: str, current: int, total: int) -> int:
        return stage_to_percent(stage, current, total)

    def create_channel(self, display_client_id: str | None = None) -> ProgressChannel:
        return ProgressChannel(
            channel_id=secrets.token_urlsafe(CHANNEL_TOKEN_BYTES),
            session_token=secrets.token_urlsafe(SESSION_TOKEN_BYTES),
            display_client_id=sanitize_display_client_id(display_client_id),
        )

    def validate_channel_id(self, channel_id: str) -> str:
        return validate_channel_id(channel_id)

    def validate_session_token(self, session_token: str) -> str:
        return validate_session_token(session_token)

    def is_bound(
        self,
        *,
        channel_id: str,
        session_token: str,
        expected_channel_id: str,
        expected_session_token: str,
    ) -> bool:
        channel = validate_channel_id(channel_id)
        token = validate_session_token(session_token)
        expected_channel = validate_channel_id(expected_channel_id)
        expected_token = validate_session_token(expected_session_token)
        return hmac.compare_digest(channel, expected_channel) and hmac.compare_digest(
            token, expected_token
        )


def stage_to_percent(stage: str, current: int, total: int) -> int:
    """Map a pipeline stage + sub-progress into a 0-100 overall percent."""
    clean_stage = _clean_stage(stage)
    if clean_stage in _STAGE_WEIGHTS:
        lo, hi = _STAGE_WEIGHTS[clean_stage]
    else:
        lo, hi = (0, 100)
    clean_total = _clean_progress_count(total, "total")
    if clean_total <= 0:
        return lo
    clean_current = min(_clean_progress_count(current, "current"), clean_total)
    return lo + int((clean_current / clean_total) * (hi - lo))


def validate_stage(stage: str) -> Stage:
    clean_stage = _clean_stage(stage)
    if clean_stage not in _STAGE_WEIGHTS:
        raise ValueError("stage must be one of: convert, detect, ocr, refine, embed.")
    return clean_stage


def sanitize_display_client_id(client_id: str | None) -> str | None:
    if client_id is None:
        return None
    if not isinstance(client_id, str):
        raise TypeError("display client ID must be a string.")
    cleaned = client_id.strip()
    if not cleaned:
        return None
    if not _DISPLAY_ID_RE.fullmatch(cleaned):
        raise ValueError("display client ID contains invalid characters.")
    return cleaned


def validate_channel_id(channel_id: str) -> str:
    return _validate_token(channel_id, "channel_id")


def validate_session_token(session_token: str) -> str:
    return _validate_token(session_token, "session_token")


def _clean_stage(stage: str) -> str:
    if not isinstance(stage, str):
        raise TypeError("stage must be a string.")
    return stage.strip().lower()


def _clean_progress_count(value: int, field_name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    return max(value, 0)


def _validate_token(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    cleaned = value.strip()
    if not _TOKEN_RE.fullmatch(cleaned):
        raise ValueError(f"{field_name} is not a valid opaque token.")
    return cleaned
