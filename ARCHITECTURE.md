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
| `src/pdf_ocr/cli.py` | CLI arguments, runtime wiring, and Rich progress output |
| `src/pdf_ocr/server.py` | FastAPI application setup and server entry point |
| `src/pdf_ocr/pipeline.py` | Shared hybrid and grounded OCR orchestration |
| `src/pdf_ocr/core/aligner.py` | Surya detection and DP text-to-box alignment |
| `src/pdf_ocr/core/ocr.py` | OpenAI-compatible VLM calls, prompts, limits, and OCR response filters |
| `src/pdf_ocr/core/pdf.py` | PDF/image conversion and searchable PDF embedding |
| `src/pdf_ocr/core/grounded.py` | Grounded OCR backends and bbox-native response parsing |
| `src/pdf_ocr/core/postprocess.py` | Dictionary-based spellcheck post-processing |
| `src/pdf_ocr/core/translation.py` | Optional LangGraph translation workflow |
| `src/pdf_ocr/api/routers/ocr.py` | OCR, translation, extraction, and asynchronous job routes |
| `src/pdf_ocr/api/routers/config.py` | Runtime configuration and model discovery |
| `src/pdf_ocr/api/routers/websocket.py` | WebSocket progress transport |
| `src/pdf_ocr/api/schemas/` | Typed FastAPI boundary schemas for runtime configuration, OCR form settings, translation, and extraction requests |
| `src/pdf_ocr/api/services/security.py` | API upload validation, stable error constants, temporary-file cleanup, and opaque text artifact IDs |
| `src/pdf_ocr/api/tasks.py` | Celery OCR task execution |
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
