"""Typed configuration boundary for the optional async translation workflow."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_TRANSLATION_API_BASE = "http://localhost:1234/v1"
DEFAULT_TRANSLATION_API_KEY = "lm-studio"
DEFAULT_TRANSLATION_MODEL = "allenai/olmocr-2-7b"


class AsyncTranslationUnavailable(RuntimeError):
    """Raised when the optional async translation runtime is unavailable."""


@dataclass(frozen=True, slots=True)
class TranslationSettings:
    """OpenAI-compatible endpoint settings used by async translation."""

    api_base: str = DEFAULT_TRANSLATION_API_BASE
    api_key: str = DEFAULT_TRANSLATION_API_KEY
    model: str = DEFAULT_TRANSLATION_MODEL

    def __post_init__(self) -> None:
        for field_name, value in (
            ("api_base", self.api_base),
            ("api_key", self.api_key),
            ("model", self.model),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

    @classmethod
    def from_env(cls) -> TranslationSettings:
        """Build settings from the same environment variables as the web config."""
        return cls(
            api_base=os.getenv("LLM_API_BASE", DEFAULT_TRANSLATION_API_BASE),
            api_key=os.getenv("LLM_API_KEY", DEFAULT_TRANSLATION_API_KEY),
            model=os.getenv("LLM_MODEL", DEFAULT_TRANSLATION_MODEL),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> TranslationSettings:
        """Build settings from a broader runtime config mapping."""
        return cls(
            api_base=_string_value(values, "api_base", DEFAULT_TRANSLATION_API_BASE),
            api_key=_string_value(values, "api_key", DEFAULT_TRANSLATION_API_KEY),
            model=_string_value(values, "model", DEFAULT_TRANSLATION_MODEL),
        )


def _string_value(values: Mapping[str, object], key: str, default: str) -> str:
    value = values.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value
