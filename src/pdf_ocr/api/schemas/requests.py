from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PipelineMode(StrEnum):
    HYBRID = "hybrid"
    GROUNDED = "grounded"


class DenseMode(StrEnum):
    AUTO = "auto"
    ALWAYS = "always"
    NEVER = "never"


class SpellcheckMode(StrEnum):
    NONE = "none"
    AR = "ar"
    EN_US = "en-US"
    DE = "de"
    ES = "es"
    FR = "fr"


class ExtractionTemplate(StrEnum):
    INVOICE = "invoice"
    RESUME = "resume"
    ACADEMIC = "academic"
    CUSTOM = "custom"


_PAGE_RANGE_RE = re.compile(
    r"^\s*\d+\s*(?:-\s*\d+\s*)?(?:,\s*\d+\s*(?:-\s*\d+\s*)?)*\s*$"
)


def _reject_bool_for_int(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("must be an integer")
    return value


def _reject_string_for_config_number(value: Any) -> Any:
    if isinstance(value, str):
        raise ValueError("must be a JSON number")
    return _reject_bool_for_int(value)


def _reject_string_for_config_bool(value: Any) -> Any:
    if not isinstance(value, bool):
        raise ValueError("must be a JSON boolean")
    return value


def _non_empty_string(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("must be a non-empty string")
    return value.strip()


class ConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_base: str | None = None
    api_key: str | None = None
    model: str | None = None
    concurrency: int | None = Field(default=None, ge=1, le=64)
    dpi: int | None = Field(default=None, ge=10, le=600)
    dense_mode: DenseMode | None = None
    dense_threshold: int | None = Field(default=None, ge=0, le=10_000)
    max_image_dim: int | None = Field(default=None, ge=100, le=4096)
    refine: bool | None = None
    verify_model: bool | None = None
    pipeline_mode: PipelineMode | None = None
    self_correction: bool | None = None
    binarize: bool | None = None
    dual_engine: bool | None = None
    spellcheck: SpellcheckMode | None = None
    cross_page: bool | None = None

    @field_validator("api_base", "api_key", "model", mode="before")
    @classmethod
    def validate_strings(cls, value: Any) -> Any:
        if value is None:
            return value
        return _non_empty_string(value)

    @field_validator(
        "concurrency",
        "dpi",
        "dense_threshold",
        "max_image_dim",
        mode="before",
    )
    @classmethod
    def validate_config_numbers(cls, value: Any) -> Any:
        if value is None:
            return value
        return _reject_string_for_config_number(value)

    @field_validator(
        "refine",
        "verify_model",
        "self_correction",
        "binarize",
        "dual_engine",
        "cross_page",
        mode="before",
    )
    @classmethod
    def validate_config_booleans(cls, value: Any) -> Any:
        if value is None:
            return value
        return _reject_string_for_config_bool(value)


class ProcessSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_base: str
    api_key: str
    model: str
    pipeline_mode: PipelineMode
    dpi: int = Field(ge=10, le=600)
    concurrency: int = Field(ge=1, le=64)
    dense_mode: DenseMode
    dense_threshold: int = Field(ge=0, le=10_000)
    pages: str | None = None
    refine: bool
    max_image_dim: int = Field(ge=100, le=4096)
    self_correction: bool
    binarize: bool
    dual_engine: bool
    spellcheck: SpellcheckMode
    cross_page: bool

    @field_validator("api_base", "api_key", "model", mode="before")
    @classmethod
    def validate_strings(cls, value: Any) -> Any:
        return _non_empty_string(value)

    @field_validator(
        "dpi", "concurrency", "dense_threshold", "max_image_dim", mode="before"
    )
    @classmethod
    def validate_form_numbers(cls, value: Any) -> Any:
        return _reject_bool_for_int(value)

    @field_validator("pages", mode="before")
    @classmethod
    def validate_pages(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        if not isinstance(value, str) or not _PAGE_RANGE_RE.match(value):
            raise ValueError("must be a comma-separated page range such as 1-3,5")
        return value.strip()


class TranslationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = ""
    target_language: str = Field(default="Spanish", min_length=1, max_length=80)
    api_base: str | None = None
    api_key: str | None = None
    model: str | None = None

    @field_validator(
        "text", "target_language", "api_base", "api_key", "model", mode="before"
    )
    @classmethod
    def validate_optional_strings(cls, value: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, str):
            raise ValueError("must be a string")
        return value.strip()


class ExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = ""
    template: ExtractionTemplate = ExtractionTemplate.INVOICE
    custom_prompt: str = Field(default="", max_length=4000)
    api_base: str | None = None
    api_key: str | None = None
    model: str | None = None

    @field_validator(
        "text", "custom_prompt", "api_base", "api_key", "model", mode="before"
    )
    @classmethod
    def validate_optional_strings(cls, value: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, str):
            raise ValueError("must be a string")
        return value.strip()
