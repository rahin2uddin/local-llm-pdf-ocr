# Architecture Ledger

## System Shape

`local-deepl` is a Python 3.11+ OCR application with a shared pipeline
used by the CLI and FastAPI web server. Inputs are PDFs or images. Outputs are
searchable sandwich PDFs with normalized OCR bounding boxes embedded as an
invisible text layer.

## Pipeline

```text
PDF/image -> raster pages -> Surya detection -> sparse: full-page VLM OCR -> DP alignment --+
                                      \-> dense: per-box VLM OCR ---------------------------+-> optional refine -> optional post-process -> DocumentResult -> optional document processors -> searchable PDF

PDF/image -> grounded bbox-native VLM OCR -> optional post-process -> DocumentResult -> optional document processors -> searchable PDF
```

## Directory Responsibilities

| Path | Single Responsibility |
| --- | --- |
| `src/local_deepl/__init__.py` | Lazy package-level public exports that avoid loading OCR or web dependencies during unrelated submodule imports |
| `src/local_deepl/cli.py` | CLI arguments, runtime wiring, and Rich progress output |
| `src/local_deepl/server.py` | Lazy optional-web dependency loading, FastAPI application setup, and server entry point |
| `src/local_deepl/pipeline.py` | Shared hybrid and grounded OCR orchestration |
| `src/local_deepl/core/document.py` | Normalized `DocumentResult` IR, pages, blocks, spans, text aggregation, and legacy pages-data adapter |
| `src/local_deepl/core/processors.py` | Local deterministic document processor protocol, registry, reading-order processor, quality-analysis processor, and user-facing processor builder |
| `src/local_deepl/core/aligner.py` | Surya detection and DP text-to-box alignment |
| `src/local_deepl/core/ocr.py` | OpenAI-compatible VLM calls, prompts, limits, and OCR response filters |
| `src/local_deepl/core/pdf.py` | PDF/image conversion and searchable PDF embedding |
| `src/local_deepl/core/grounded.py` | Grounded OCR backends and bbox-native response parsing |
| `src/local_deepl/core/postprocess.py` | Dictionary-based spellcheck post-processing |
| `src/local_deepl/core/translation_config.py` | Core-owned typed settings and optional-feature errors for async translation |
| `src/local_deepl/core/translation.py` | Optional LangGraph translation workflow |
| `src/local_deepl/resources/dictionaries/` | Packaged compiled spellcheck dictionaries loaded before legacy repository-root dictionaries |
| `src/local_deepl/api/routers/ocr.py` | OCR, translation, extraction, and asynchronous job routes |
| `src/local_deepl/api/routers/config.py` | Runtime configuration and model discovery |
| `src/local_deepl/api/routers/websocket.py` | Token-bound WebSocket progress transport and progress session issuance |
| `src/local_deepl/api/schemas/` | Typed FastAPI boundary schemas for runtime configuration, OCR form settings, translation, and extraction requests |
| `src/local_deepl/api/services/document_metadata.py` | Compact JSON report builder and atomic writer for token-bound `DocumentResult` metadata artifacts |
| `src/local_deepl/api/services/security.py` | API upload validation, stable error constants, temporary-file cleanup, and opaque text artifact IDs |
| `src/local_deepl/api/tasks.py` | Optional Celery translation task execution |
| `src/local_deepl/utils/image.py` | Image crop, blank-region detection, and crop encoding helpers |
| `src/local_deepl/utils/security.py` | SSRF target validation |
| `src/local_deepl/utils/litellm_provider.py` | LiteLLM provider selection |
| `src/local_deepl/utils/tqdm_patch.py` | Surya progress-bar suppression |
| `src/local_deepl/static/` | Browser workstation assets |
| `tests/` | Unit, integration, security, and slow-path validation |

## Extension Points

`OCRPipeline` accepts injected `aligner`, `ocr_processor`, `pdf_handler`,
`output_writer`, `grounded_backend`, and `document_processors` components. Keep
PDF and image inputs on the same output-writer path, and keep normalized bboxes
in `[x0, y0, x1, y1]` form until embedding.

Document processors receive a mutable `DocumentResult` after OCR cleanup,
spellcheck, and cross-page merge but before PDF embedding. The web/API surface
can select built-in local processors by name through `document_processors`.
Current built-ins are `reading_order`, `quality_analysis`, and
`structure_analysis`, and `section_analysis`.

## Performance Notes

- Dense-mode and refine crop paths decode a page image once and reuse the PIL
  image across boxes.
- Grounded PDF rasterization converts PyMuPDF pixmaps directly into Pillow
  images before producing the final thumbnail JPEG.

## Change Blueprint

### 2026-06-09: Local document processors exposed to web/API

| File | Responsibility |
| --- | --- |
| `src/local_deepl/core/document.py` | Provide the normalized `DocumentResult` handoff used by post-OCR document processors |
| `src/local_deepl/core/processors.py` | Define built-in local processors and map user-facing names to deterministic processor instances |
| `src/local_deepl/api/schemas/requests.py` | Validate `document_processors` for config JSON and multipart OCR requests |
| `src/local_deepl/api/routers/ocr.py` | Instantiate selected processors, pass them into `OCRPipeline`, and expose quality metadata through `X-Document-Quality` when available |
| `src/local_deepl/static/js/state_and_api.js` | Persist and submit web-selected document processors |
| `src/local_deepl/static/index.html` | Expose Reading Order, Quality Analysis, Structure Analysis, and Section Analysis toggles in Advanced Configuration |
| `tests/test_document_processor_selection.py` | Cover processor selection parsing, validation, and factory mapping |

### 2026-06-09: Stage 2 local structure analysis processor

| File | Responsibility |
| --- | --- |
| `src/local_deepl/core/processors.py` | Add `structure_analysis`, a deterministic local processor that classifies blocks as headings, paragraphs, list items, key-values, table candidates, or empty blocks |
| `src/local_deepl/api/routers/ocr.py` | Expose page-level structure summaries through `X-Document-Structure` when structure metadata is present |
| `src/local_deepl/static/index.html` | Add the Structure Analysis opt-in control |
| `tests/test_document.py` | Cover block classification without rewriting output text |

### 2026-06-09: Stage 3 local section analysis processor

| File | Responsibility |
| --- | --- |
| `src/local_deepl/core/processors.py` | Add `section_analysis`, a deterministic local processor that assigns blocks to detected heading sections across page boundaries |
| `src/local_deepl/api/routers/ocr.py` | Expose page-level section summaries through `X-Document-Sections` when section metadata is present |
| `src/local_deepl/static/index.html` | Add the Section Analysis opt-in control |
| `tests/test_document.py` | Cover section grouping while preserving original block text |

### 2026-06-09: Stage 4 document metadata artifact surface

| File | Responsibility |
| --- | --- |
| `src/local_deepl/api/services/document_metadata.py` | Build compact JSON-safe metadata reports from `DocumentResult` page/block processor annotations and write them atomically as temporary artifacts |
| `src/local_deepl/api/routers/ocr.py` | Issue `X-Document-Metadata-Artifact-Id` and `X-Document-Metadata-Artifact-Token` only when report content exists, and serve protected `GET /metadata/{artifact_id}` |
| `tests/test_api_safety.py` | Cover token-bound metadata artifact access and payload shape without changing text artifact behavior |

### 2026-06-02: Direct grounded PDF pixmap conversion

| File | Responsibility |
| --- | --- |
| `src/local_deepl/core/grounded.py` | Convert PDF pixmaps directly into Pillow images before emitting the final grounded OCR thumbnail JPEG |
| `tests/test_grounded.py` | Guard against restoring the redundant intermediate JPEG decode |
| `ARCHITECTURE.md` | Record the existing module layout and the direct pixmap conversion invariant |

### 2026-06-02: Stage 1 API and browser safety hardening

| File | Responsibility |
| --- | --- |
| `src/local_deepl/api/schemas/requests.py` | Validate config JSON, OCR multipart settings, translation requests, and extraction requests with explicit enums, booleans, and numeric ranges |
| `src/local_deepl/api/services/security.py` | Enforce streaming upload byte limits, content-signature upload type detection, stable API error messages, and server-issued text artifact IDs |
| `src/local_deepl/api/routers/config.py` | Apply typed config validation, SSRF checks, safe environment parsing, and non-leaking model discovery errors |
| `src/local_deepl/api/routers/ocr.py` | Apply typed OCR/AI boundary validation, hardened upload dispatch, opaque text artifact retrieval, SSRF checks, and stable client-facing errors |
| `src/local_deepl/utils/security.py` | Fail closed for malformed, unsupported, or unresolvable URLs and only allow local/private endpoints when `ALLOW_SSRF_LOCAL=true` is explicitly set |
| `src/local_deepl/static/js/app.js` | Use server-issued text artifact IDs and render extraction status/errors/cards without HTML injection |
| `src/local_deepl/static/js/state_and_api.js` | Build model select placeholder with DOM APIs before appending model-controlled option text |
| `src/local_deepl/static/js/workspace_ui.js` | Provide safe DOM helpers for clearing elements and rendering extraction status cards |
| `tests/test_api_safety.py` | Cover config validation, SSRF fail-closed behavior, streaming upload validation, opaque text artifacts, stable API errors, and static JS sink removal |
| `tests/test_security_qa.py` | Keep extraction JSON parsing deterministic under fail-closed SSRF validation |

### 2026-06-03: Optional async translation boundary

| File | Responsibility |
| --- | --- |
| `src/local_deepl/core/translation_config.py` | Own typed translation settings and the deterministic optional-feature error used by core and API boundaries |
| `src/local_deepl/core/translation.py` | Keep chunking and evaluation helpers importable without async extras, lazily build the LangGraph workflow, and accept injected translation settings |
| `src/local_deepl/api/routers/config.py` | Adapt the mutable web runtime config into core-owned translation settings without exposing `_config` to core modules |
| `src/local_deepl/api/celery_app.py` | Guard Celery imports and provide an import-safe fallback task facade when async extras are not installed |
| `src/local_deepl/api/tasks.py` | Validate async translation task inputs and pass explicit translation settings into the core workflow |
| `src/local_deepl/api/routers/ocr.py` | Validate async translation route inputs and return deterministic 503 responses when optional async extras are unavailable |
| `pyproject.toml` | Move Celery, Redis, LangGraph, ChromaDB, and sentence-transformers into the `async-translation` extra with `translation` as an alias extra |
| `tests/test_translation_boundary.py` | Cover guarded imports without async extras and explicit translation settings injection |

### 2026-06-03: Spellcheck resource package cleanup

| File | Responsibility |
| --- | --- |
| `src/local_deepl/resources/dictionaries/ara.json.gz` | Packaged Arabic compiled spellcheck dictionary for installed distributions |
| `src/local_deepl/resources/dictionaries/eng.json.gz` | Packaged English compiled spellcheck dictionary for installed distributions |
| `src/local_deepl/core/postprocess.py` | Load packaged dictionaries first while retaining legacy repository-root and user-cache fallbacks |
| `pyproject.toml` | Exclude bytecode cache artifacts from Hatch package builds |
| `tests/test_dictionary_postprocess.py` | Cover packaged dictionary lookup and legacy repository-root fallback |

### 2026-06-03: Lazy web server imports

| File | Responsibility |
| --- | --- |
| `src/local_deepl/__init__.py` | Preserve package-level OCR exports through lazy lookups so `import local_deepl.server` does not load OCR core dependencies first |
| `src/local_deepl/server.py` | Preserve `local_deepl.server:app` and `local_deepl.server:main` while deferring FastAPI, router, static-file, and uvicorn imports until the web app is created or run |
| `tests/test_server_lazy_imports.py` | Verify base-install-safe `local_deepl.server` imports and deterministic missing-web-extra errors without uninstalling FastAPI |
| `ARCHITECTURE.md` | Record the optional-web lazy import boundary for the server module |
