import logging
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from pdf_ocr.api.schemas import ConfigUpdate
from pdf_ocr.api.services.security import SAFE_API_BASE_ERROR, SERVER_ERROR_MESSAGE
from pdf_ocr.core.translation_config import (
    DEFAULT_TRANSLATION_API_BASE,
    DEFAULT_TRANSLATION_API_KEY,
    DEFAULT_TRANSLATION_MODEL,
    TranslationSettings,
)
from pdf_ocr.utils.security import is_ssrf_target

router = APIRouter()
logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid integer environment value for %s", name)
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# In-memory configuration store – initialised from environment variables
# ---------------------------------------------------------------------------
_config: dict = {
    "api_base": os.getenv("LLM_API_BASE", DEFAULT_TRANSLATION_API_BASE),
    "api_key": os.getenv("LLM_API_KEY", DEFAULT_TRANSLATION_API_KEY),
    "model": os.getenv("LLM_MODEL", DEFAULT_TRANSLATION_MODEL),
    "concurrency": _env_int("OCR_CONCURRENCY", 3),
    "dpi": _env_int("OCR_DPI", 200),
    "dense_mode": os.getenv("OCR_DENSE_MODE", "auto"),
    "dense_threshold": _env_int("OCR_DENSE_THRESHOLD", 60),
    "max_image_dim": _env_int("OCR_MAX_IMAGE_DIM", 1024),
    "refine": _env_bool("OCR_REFINE", True),
    "verify_model": _env_bool("OCR_VERIFY_MODEL", True),
    "pipeline_mode": os.getenv("OCR_PIPELINE_MODE", "hybrid"),
    "self_correction": _env_bool("OCR_SELF_CORRECTION", False),
    "binarize": _env_bool("OCR_BINARIZE", False),
    "dual_engine": _env_bool("OCR_DUAL_ENGINE", False),
    "spellcheck": os.getenv("OCR_SPELLCHECK", "none"),
    "cross_page": _env_bool("OCR_CROSS_PAGE", False),
}


def get_translation_settings() -> TranslationSettings:
    """Return core-owned settings for the optional async translation workflow."""
    return TranslationSettings.from_mapping(_config)


# ---- Configuration --------------------------------------------------------


@router.get("/api/config")
async def get_config():
    """Return the current runtime configuration, masking the API key."""
    safe_config = _config.copy()
    if safe_config.get("api_key") and safe_config["api_key"] != "lm-studio":
        key = safe_config["api_key"]
        safe_config["api_key"] = (
            f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "********"
        )
    return JSONResponse(content=safe_config)


@router.post("/api/config")
async def update_config(body: ConfigUpdate):
    """
    Update runtime configuration.

    Unknown keys and invalid value types fail validation instead of being
    silently ignored.
    """
    values = body.model_dump(exclude_unset=True)
    for key, val in values.items():
        if (
            key == "api_key"
            and isinstance(val, str)
            and ("..." in val or val == "********")
        ):
            continue
        if key == "api_base" and is_ssrf_target(val):
            return JSONResponse(status_code=403, content={"error": SAFE_API_BASE_ERROR})
        _config[key] = val.value if hasattr(val, "value") else val
    return await get_config()


# ---- Model discovery ------------------------------------------------------


@router.get("/api/models")
async def list_models():
    """
    Query the OpenAI-compatible endpoint for available models.

    Uses the current ``api_base`` from the config store.
    """
    if is_ssrf_target(_config["api_base"]):
        return JSONResponse(status_code=403, content={"error": SAFE_API_BASE_ERROR})
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url=_config["api_base"],
            api_key=_config["api_key"],
        )
        response = await client.models.list()
        model_ids = [m.id for m in response.data] if response.data else []
        return JSONResponse(content={"models": model_ids})
    except Exception:
        logger.exception("Model discovery failed")
        return JSONResponse(content={"models": [], "error": SERVER_ERROR_MESSAGE})
