# AGENTS.md

This file tells coding agents and contributors how to work with this repository.

## Quick Start

```bash
uv sync
uv sync --extra web
uv sync --extra web --extra async-translation
uv run local-llm-pdf-ocr input.pdf
uv run local-llm-pdf-ocr-server --port 8000
```

Real OCR requires an OpenAI-compatible VLM endpoint. The default is LM Studio at `http://localhost:1234/v1`.

## Validation

```bash
uv run pytest
uv run pytest -m "not slow"
uv run pytest -m slow
uv run pytest tests/test_aligner.py -v
uv run ruff check src tests
uv run ruff format src tests --check
uv run mypy src
```

- `pytest-asyncio` uses auto mode. Write `async def test_...` without decorators.
- Slow tests load Surya and may download its model on the first run.
- Markers are `slow` and `live_llm`.

## Conventions

- Python 3.11 or newer. Use `uv`; do not install dependencies with `pip`.
- Prefer self-documenting code and docstrings. Add comments only when they clarify non-obvious behavior.
- Preserve lazy imports for heavy modules in CLI entry points so `--help` remains fast.
- Keep `tqdm_patch.apply()` before `from surya.detection import DetectionPredictor` in `core/aligner.py`.
- Keep bboxes normalized as `[x0, y0, x1, y1]` in `0..1` until `PDFHandler.embed_structured_text`.
- Treat image inputs as first-class inputs. PDF and image paths share the output writer.
- Keep CLI and web capabilities distinct: advanced enhancement settings are wired through the web router and `OCRPipeline`, but not exposed as CLI flags.

## Pipeline Paths

```text
PDF/image -> pages -> Surya detection -> sparse: full-page OCR -> DP alignment -> refine --+
                                    \-> dense: per-box OCR -------------------------------+-> post-process -> searchable PDF

PDF/image -> grounded bbox-native VLM -> post-process -> searchable PDF
```

- Hybrid is the default: Surya detection, VLM OCR, DP alignment, optional refine, optional post-processing, embed.
- Dense hybrid pages use per-box OCR. `dense_mode="auto"` switches when box count exceeds `dense_threshold`.
- Grounded OCR uses `grounded_backend=` and skips Surya, DP alignment, and refine.

## Key Files

| File | Role |
| --- | --- |
| `src/pdf_ocr/cli.py` | CLI parser and Rich progress output |
| `src/pdf_ocr/server.py` | FastAPI application and server entry point |
| `src/pdf_ocr/pipeline.py` | Shared hybrid and grounded orchestration |
| `src/pdf_ocr/core/aligner.py` | Surya detection and DP alignment |
| `src/pdf_ocr/core/ocr.py` | LiteLLM OCR calls, prompts, limits, and filters |
| `src/pdf_ocr/core/pdf.py` | PDF/image conversion and sandwich-PDF embedding |
| `src/pdf_ocr/core/grounded.py` | Grounded backends and bbox JSON parsers |
| `src/pdf_ocr/core/postprocess.py` | Dictionary spellcheck |
| `src/pdf_ocr/core/translation_config.py` | Core-owned async translation settings |
| `src/pdf_ocr/core/translation.py` | Optional LangGraph translation workflow |
| `src/pdf_ocr/resources/dictionaries/` | Packaged spellcheck dictionaries |
| `src/pdf_ocr/api/routers/ocr.py` | OCR, translation, extraction, and job routes |
| `src/pdf_ocr/api/routers/config.py` | Runtime configuration and model discovery |
| `src/pdf_ocr/utils/security.py` | SSRF target validation |
| `src/pdf_ocr/utils/litellm_provider.py` | LiteLLM provider selection |

## Extension Points

`OCRPipeline` accepts injected components:

- `aligner=`: layout detection and text alignment
- `ocr_processor=`: page and crop OCR backend
- `pdf_handler=`: input conversion and default PDF writer
- `output_writer=`: alternate output generation
- `grounded_backend=`: bbox-native OCR path

## Web Notes

- Browser translation and structured extraction use synchronous endpoints and do not require Redis.
- `/api/translate/async` uses Celery, Redis, and LangGraph from the `async-translation` extra.
- `ALLOW_SSRF_LOCAL=true` is the local-development default. Set it to `false` when exposing the server to untrusted users.
- Web runtime settings are initialized in `api/routers/config.py`.

## Known Tech Debt

- `api/routers/ocr.py` mixes OCR, translation, extraction, and asynchronous task routes.
- The grounded web route instantiates hybrid components even though `OCRPipeline` skips them in grounded mode.
- `ZAIHostedOCR` remains an experimental backend.
