import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory configuration store – initialised from environment variables
# ---------------------------------------------------------------------------
_config: dict = {
    "api_base": os.getenv("LLM_API_BASE", "http://localhost:1234/v1"),
    "api_key": os.getenv("LLM_API_KEY", "lm-studio"),
    "model": os.getenv("LLM_MODEL", "allenai/olmocr-2-7b"),
    "concurrency": int(os.getenv("OCR_CONCURRENCY", "3")),
    "dpi": int(os.getenv("OCR_DPI", "200")),
    "dense_mode": os.getenv("OCR_DENSE_MODE", "auto"),
    "dense_threshold": int(os.getenv("OCR_DENSE_THRESHOLD", "60")),
    "max_image_dim": int(os.getenv("OCR_MAX_IMAGE_DIM", "1024")),
    "refine": os.getenv("OCR_REFINE", "1") != "0",
    "verify_model": os.getenv("OCR_VERIFY_MODEL", "1") != "0",
    "pipeline_mode": os.getenv("OCR_PIPELINE_MODE", "hybrid"),
    "self_correction": os.getenv("OCR_SELF_CORRECTION", "0") != "0",
    "binarize": os.getenv("OCR_BINARIZE", "0") != "0",
    "dual_engine": os.getenv("OCR_DUAL_ENGINE", "0") != "0",
    "spellcheck": os.getenv("OCR_SPELLCHECK", "none"),
    "cross_page": os.getenv("OCR_CROSS_PAGE", "0") != "0",
}


# ---- Configuration --------------------------------------------------------

@router.get("/api/config")
async def get_config():
    """Return the current runtime configuration, masking the API key."""
    safe_config = _config.copy()
    if safe_config.get("api_key") and safe_config["api_key"] != "lm-studio":
        key = safe_config["api_key"]
        safe_config["api_key"] = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "********"
    return JSONResponse(content=safe_config)


@router.post("/api/config")
async def update_config(body: dict):
    """
    Update runtime configuration.

    Only keys already present in the config store are accepted; unknown
    keys are silently ignored.
    """
    for key in _config:
        if key in body:
            expected = type(_config[key])
            try:
                # Don't overwrite api_key with the masked version if the user didn't change it
                if key == "api_key" and ("..." in body[key] or body[key] == "********"):
                    continue
                _config[key] = expected(body[key])
            except (ValueError, TypeError):
                pass  # skip values that cannot be coerced
    return await get_config()



# ---- Model discovery ------------------------------------------------------

@router.get("/api/models")
async def list_models():
    """
    Query the OpenAI-compatible endpoint for available models.

    Uses the current ``api_base`` from the config store.
    """
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url=_config["api_base"],
            api_key=_config["api_key"],
        )
        response = await client.models.list()
        model_ids = [m.id for m in response.data]
        return JSONResponse(content={"models": model_ids})
    except Exception as exc:
        return JSONResponse(content={"models": [], "error": str(exc)})


