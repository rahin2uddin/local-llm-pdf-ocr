# AGENTS.md

This file tells AI coding agents and new contributors how to work with this codebase.

## Quick Start

```bash
uv sync                                # install deps
uv sync --extra web                    # + FastAPI server deps
uv run local-llm-pdf-ocr input.pdf     # CLI OCR
uv run local-llm-pdf-ocr-server --port 8000  # web UI
```

Requires a local OpenAI-compatible VLM endpoint (LM Studio at `http://localhost:1234/v1` by default).

## Testing

```bash
uv run pytest                          # full suite (179 fast + 23 slow)
uv run pytest -m "not slow"            # fast tier only (~11s, no Surya model load)
uv run pytest -m slow                  # integration: real Surya + example PDFs
uv run pytest tests/test_aligner.py -v # single file
```

- `pytest-asyncio` is in **auto mode** -- write `async def test_...` without decorators.
- Slow tests (`-m slow`) download Surya models (~500MB) on first run.
- Tests live under `tests/`. Markers: `slow`, `live_llm`.

## Lint / Typecheck

Ruff (linter/formatter) and Mypy (type checker) are configured in `pyproject.toml`. Run them locally via:
```bash
uv run ruff check src tests            # linting
uv run ruff format src tests --check   # format checking
uv run ruff format src tests           # auto-format code
uv run mypy src                        # static type checking
```


## Code Conventions

- **Python >= 3.11**. Deps managed with `uv`; do not use `pip install`.
- **No comments unless asked** -- code style prefers self-documenting names and docstrings.
- **Lazy imports** for heavy modules (Surya, PyMuPDF) in CLI entry points so `--help` stays fast.
- **`tqdm_patch.apply()`** must run before `from surya.detection import DetectionPredictor` in `aligner.py`.
- **Normalized bboxes**: `[nx0, ny0, nx1, ny1]` in `0..1` everywhere except `embed_structured_text` which converts to PDF points.

## Architecture (Two Pipeline Paths)

```
                +-- (sparse) -- LLM full-page OCR -- DP line->box alignment -- crop re-OCR --+
PDF -> images --|                                                                               +-- output_writer -> searchable PDF
                +-- (dense)  -- per-box OCR (each Surya box -> one LLM crop call) ------------+
```

- **Hybrid** (default): Surya detect -> LLM OCR -> DP align -> refine -> embed
- **Grounded** (`--grounded`): bbox-native VLM returns text+bbox in one call, skips Surya/DP/refine

## Key Files

| File | Role |
|------|------|
| `src/pdf_ocr/cli.py` | CLI entry point |
| `src/pdf_ocr/server.py` | FastAPI web server |
| `src/pdf_ocr/pipeline.py` | `OCRPipeline` orchestration seam |
| `src/pdf_ocr/core/aligner.py` | Surya detect + Needleman-Wunsch DP |
| `src/pdf_ocr/core/ocr.py` | LLM client (OpenAI-compat) |
| `src/pdf_ocr/core/pdf.py` | PDF/image I/O + sandwich-PDF embedding |
| `src/pdf_ocr/core/grounded.py` | Grounded backends + JSON parsers |
| `src/pdf_ocr/core/postprocess.py` | Dictionary spellcheck |
| `src/pdf_ocr/core/translation.py` | LangGraph translation workflow |
| `src/pdf_ocr/api/routers/` | FastAPI route handlers |
| `src/pdf_ocr/utils/litellm_provider.py` | Shared litellm provider detection |
| `src/pdf_ocr/utils/image.py` | Crop + blank-detection utility |

## Extension Points

All pipeline phases are injected into `OCRPipeline`:
- `aligner=` -- swap layout detection (e.g. DETR)
- `ocr_processor=` -- swap LLM backend
- `output_writer=` -- swap output format (e.g. EPUB)
- `grounded_backend=` -- use bbox-native VLM

## Known Tech Debt

- The translation feature (`core/translation.py`, `api/tasks.py`, `api/celery_app.py`) depends on Celery + Redis + LangGraph + ChromaDB but these are listed as core deps, not optional.
- The API routers (`api/routers/ocr.py`) have the translate/extract endpoints mixed with the core OCR endpoint -- consider splitting into separate routers.
- `ZAIHostedOCR` in `core/grounded.py` is a skeleton with untested endpoint paths.
