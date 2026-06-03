# Architecture Ledger

## System Shape

`local-llm-pdf-ocr` is a Python 3.11+ OCR application with a shared pipeline
used by the CLI and FastAPI web server. Inputs are PDFs or images. Outputs are
searchable sandwich PDFs with normalized OCR bounding boxes embedded as an
invisible text layer.

## Pipeline

```text
PDF/image -> raster pages -> Surya detection -> sparse: full-page VLM OCR -> DP alignment --+
                                      \-> dense: per-box VLM OCR ---------------------------+-> optional refine -> optional post-process -> searchable PDF

PDF/image -> grounded bbox-native VLM OCR -> optional post-process -> searchable PDF
```

## Directory Responsibilities

| Path | Single Responsibility |
| --- | --- |
| `src/pdf_ocr/__init__.py` | Lazy package-level public exports that avoid loading OCR or web dependencies during unrelated submodule imports |
| `src/pdf_ocr/cli.py` | CLI arguments, runtime wiring, and Rich progress output |
| `src/pdf_ocr/server.py` | Lazy optional-web dependency loading, FastAPI application setup, and server entry point |
| `src/pdf_ocr/pipeline.py` | Shared hybrid and grounded OCR orchestration |
| `src/pdf_ocr/core/aligner.py` | Surya detection and DP text-to-box alignment |
| `src/pdf_ocr/core/ocr.py` | OpenAI-compatible VLM calls, prompts, limits, and OCR response filters |
| `src/pdf_ocr/core/pdf.py` | PDF/image conversion and searchable PDF embedding |
| `src/pdf_ocr/core/grounded.py` | Grounded OCR backends and bbox-native response parsing |
| `src/pdf_ocr/core/postprocess.py` | Dictionary-based spellcheck post-processing |
| `src/pdf_ocr/core/translation_config.py` | Core-owned typed settings and optional-feature errors for async translation |
| `src/pdf_ocr/core/translation.py` | Optional LangGraph translation workflow |
| `src/pdf_ocr/resources/dictionaries/` | Packaged compiled spellcheck dictionaries loaded before legacy repository-root dictionaries |
| `src/pdf_ocr/api/routers/ocr.py` | OCR, translation, extraction, and asynchronous job routes |
| `src/pdf_ocr/api/routers/config.py` | Runtime configuration and model discovery |
| `src/pdf_ocr/api/routers/websocket.py` | WebSocket progress transport |
| `src/pdf_ocr/api/schemas/` | Typed FastAPI boundary schemas for runtime configuration, OCR form settings, translation, and extraction requests |
| `src/pdf_ocr/api/services/security.py` | API upload validation, stable error constants, temporary-file cleanup, and opaque text artifact IDs |
| `src/pdf_ocr/api/tasks.py` | Optional Celery translation task execution |
| `src/pdf_ocr/utils/image.py` | Image crop, blank-region detection, and crop encoding helpers |
| `src/pdf_ocr/utils/security.py` | SSRF target validation |
| `src/pdf_ocr/utils/litellm_provider.py` | LiteLLM provider selection |
| `src/pdf_ocr/utils/tqdm_patch.py` | Surya progress-bar suppression |
| `src/pdf_ocr/static/` | Browser workstation assets |
| `tests/` | Unit, integration, security, and slow-path validation |

## Extension Points

`OCRPipeline` accepts injected `aligner`, `ocr_processor`, `pdf_handler`,
`output_writer`, and `grounded_backend` components. Keep PDF and image inputs
on the same output-writer path, and keep normalized bboxes in `[x0, y0, x1, y1]`
form until embedding.

## Performance Notes

- Dense-mode and refine crop paths decode a page image once and reuse the PIL
  image across boxes.
- Grounded PDF rasterization converts PyMuPDF pixmaps directly into Pillow
  images before producing the final thumbnail JPEG.

## Change Blueprint

### 2026-06-02: Direct grounded PDF pixmap conversion

| File | Responsibility |
| --- | --- |
| `src/pdf_ocr/core/grounded.py` | Convert PDF pixmaps directly into Pillow images before emitting the final grounded OCR thumbnail JPEG |
| `tests/test_grounded.py` | Guard against restoring the redundant intermediate JPEG decode |
| `ARCHITECTURE.md` | Record the existing module layout and the direct pixmap conversion invariant |

### 2026-06-02: Stage 1 API and browser safety hardening

| File | Responsibility |
| --- | --- |
| `src/pdf_ocr/api/schemas/requests.py` | Validate config JSON, OCR multipart settings, translation requests, and extraction requests with explicit enums, booleans, and numeric ranges |
| `src/pdf_ocr/api/services/security.py` | Enforce streaming upload byte limits, content-signature upload type detection, stable API error messages, and server-issued text artifact IDs |
| `src/pdf_ocr/api/routers/config.py` | Apply typed config validation, SSRF checks, safe environment parsing, and non-leaking model discovery errors |
| `src/pdf_ocr/api/routers/ocr.py` | Apply typed OCR/AI boundary validation, hardened upload dispatch, opaque text artifact retrieval, SSRF checks, and stable client-facing errors |
| `src/pdf_ocr/utils/security.py` | Fail closed for malformed, unsupported, or unresolvable URLs and only allow local/private endpoints when `ALLOW_SSRF_LOCAL=true` is explicitly set |
| `src/pdf_ocr/static/js/app.js` | Use server-issued text artifact IDs and render extraction status/errors/cards without HTML injection |
| `src/pdf_ocr/static/js/state_and_api.js` | Build model select placeholder with DOM APIs before appending model-controlled option text |
| `src/pdf_ocr/static/js/workspace_ui.js` | Provide safe DOM helpers for clearing elements and rendering extraction status cards |
| `tests/test_api_safety.py` | Cover config validation, SSRF fail-closed behavior, streaming upload validation, opaque text artifacts, stable API errors, and static JS sink removal |
| `tests/test_security_qa.py` | Keep extraction JSON parsing deterministic under fail-closed SSRF validation |

### 2026-06-03: Optional async translation boundary

| File | Responsibility |
| --- | --- |
| `src/pdf_ocr/core/translation_config.py` | Own typed translation settings and the deterministic optional-feature error used by core and API boundaries |
| `src/pdf_ocr/core/translation.py` | Keep chunking and evaluation helpers importable without async extras, lazily build the LangGraph workflow, and accept injected translation settings |
| `src/pdf_ocr/api/routers/config.py` | Adapt the mutable web runtime config into core-owned translation settings without exposing `_config` to core modules |
| `src/pdf_ocr/api/celery_app.py` | Guard Celery imports and provide an import-safe fallback task facade when async extras are not installed |
| `src/pdf_ocr/api/tasks.py` | Validate async translation task inputs and pass explicit translation settings into the core workflow |
| `src/pdf_ocr/api/routers/ocr.py` | Validate async translation route inputs and return deterministic 503 responses when optional async extras are unavailable |
| `pyproject.toml` | Move Celery, Redis, LangGraph, ChromaDB, and sentence-transformers into the `async-translation` extra with `translation` as an alias extra |
| `tests/test_translation_boundary.py` | Cover guarded imports without async extras and explicit translation settings injection |

### 2026-06-03: Spellcheck resource package cleanup

| File | Responsibility |
| --- | --- |
| `src/pdf_ocr/resources/dictionaries/ara.json.gz` | Packaged Arabic compiled spellcheck dictionary for installed distributions |
| `src/pdf_ocr/resources/dictionaries/eng.json.gz` | Packaged English compiled spellcheck dictionary for installed distributions |
| `src/pdf_ocr/core/postprocess.py` | Load packaged dictionaries first while retaining legacy repository-root and user-cache fallbacks |
| `pyproject.toml` | Exclude bytecode cache artifacts from Hatch package builds |
| `tests/test_dictionary_postprocess.py` | Cover packaged dictionary lookup and legacy repository-root fallback |

### 2026-06-03: Lazy web server imports

| File | Responsibility |
| --- | --- |
| `src/pdf_ocr/__init__.py` | Preserve package-level OCR exports through lazy lookups so `import pdf_ocr.server` does not load OCR core dependencies first |
| `src/pdf_ocr/server.py` | Preserve `pdf_ocr.server:app` and `pdf_ocr.server:main` while deferring FastAPI, router, static-file, and uvicorn imports until the web app is created or run |
| `tests/test_server_lazy_imports.py` | Verify base-install-safe `pdf_ocr.server` imports and deterministic missing-web-extra errors without uninstalling FastAPI |
| `ARCHITECTURE.md` | Record the optional-web lazy import boundary for the server module |
