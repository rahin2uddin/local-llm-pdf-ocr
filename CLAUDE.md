# CLAUDE.md

Repository guidance for Claude Code. Read `AGENTS.md` first; it contains the shared contributor rules. `README.md` is the user-facing source of truth for installation, CLI usage, web behavior, environment variables, and HTTP routes.

## Common Commands

```bash
uv sync
uv sync --extra web
uv run local-deepl input.pdf
uv run local-deepl input.pdf output.pdf --pages 1-3,5 --dpi 300
uv run local-deepl notes.pdf --dense-mode always --concurrency 5
uv run local-deepl scan.pdf --grounded --model qwen/qwen3-vl-8b
uv run local-deepl-server --port 8000
uv run pytest -m "not slow"
uv run ruff check src tests
uv run mypy src
```

Debugging and evaluation tools live in `scripts/`.

## Important Behavior

- The default hybrid path is `convert -> detect -> OCR -> align -> refine -> post-process -> embed`.
- Dense hybrid pages skip full-page OCR and run per-box crop OCR.
- The grounded path is `grounded OCR -> post-process -> embed`; it does not load Surya.
- Bounding boxes stay normalized as `[x0, y0, x1, y1]` until PDF embedding.
- Local model pre-flight checks call `GET /v1/models`. CLI users can bypass this with `--no-verify-model`; the web server uses `OCR_VERIFY_MODEL=0`.
- Cloud-style LiteLLM model prefixes and `api.openai.com` skip local model verification automatically.
- Full-page OCR and crop OCR have separate timeout and token limits in `core/ocr.py`.
- Crop OCR drops blank-region and known placeholder responses before embedding.
- Refine deduplication clears crop text that duplicates nearby aligned text.

## Web-Only Enhancements

The web workspace and `OCRPipeline.run()` expose settings that are not CLI flags:

- `self_correction`
- `binarize`
- `dual_engine`
- `spellcheck`
- `cross_page`
- `document_processors` (`reading_order`, `quality_analysis`, `structure_analysis`, `section_analysis`)

Do not document them as CLI options unless the CLI parser is updated.

## Translation Paths

There are two translation implementations:

- `POST /api/translate`: synchronous LiteLLM request used by the browser workspace.
- `POST /api/translate/async`: optional Celery task using Redis and `core/translation.py`.

Redis and Celery are not required for ordinary OCR or browser translation.

## Testing Notes

- Fast tests avoid loading Surya: `uv run pytest -m "not slow"`.
- Slow tests exercise Surya and example PDFs: `uv run pytest -m slow`.
- Live endpoint tests use the `live_llm` marker when added.
- CI runs Ruff, Mypy, and fast tests on Python 3.11 and 3.13.

## Gotchas

- Do not move `tqdm_patch.apply()` below the Surya import.
- Preserve the Pillow override in `pyproject.toml`; Pillow 11.3 or newer provides AVIF decoding.
- Preserve lazy heavy imports in the CLI.
- Keep SSRF behavior explicit when editing web endpoints. `ALLOW_SSRF_LOCAL=true` intentionally permits local model servers during development.
- The Windows launcher starts Redis, Celery, and Uvicorn. Ordinary manual web-server startup only needs the `web` extra.
