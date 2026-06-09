# LocalDeepL

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Web_UI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

Turn scanned PDFs and images into searchable, selectable PDFs using local vision language models (VLMs). All processing runs locally, ensuring your document content remains secure on your machine.

---

## Features

- **Format Support**: Accepts PDFs and images (JPEG, PNG, BMP, WebP, TIFF, AVIF). Multi-frame TIFFs are expanded into multi-page PDFs.
- **Searchable Output**: Generates sandwich PDFs containing the original page images with a hidden, searchable text layer.
- **Dual Processing Paths**:
  - *Hybrid (Default)*: Surya layout detection -> VLM OCR -> Dynamic Programming (DP) text-to-box alignment. Automatically switches to per-box OCR on dense pages.
  - *Grounded*: Directly transcribes with layout positions using bbox-native VLMs (e.g. Qwen2.5-VL / Qwen3-VL), bypassing layout models.
- **Local Document Processors**: Optional local post-OCR processors can normalize reading order, attach page-level quality analysis, classify document structure, and group content under section headings without changing the searchable PDF output by default.
- **Web Workspace**: Premium FastAPI-based web interface featuring page selection, live WebSocket progress tracking, text preview, translation, structured JSON data extraction (invoices, resumes, academic papers), and job history.

---

## Requirements

- Python 3.11 or newer
- An OpenAI-compatible VLM endpoint (defaults to LM Studio at `http://localhost:1234/v1` running `allenai/olmocr-2-7b`)

---

## Installation

### Recommended (with `uv`)

1. Clone the repository and sync packages:
   ```bash
   git clone https://github.com/Sifr-r/LocalDeepL.git
   cd LocalDeepL
   uv sync
   ```
2. Sync optional dependencies as needed:
   ```bash
   uv sync --extra web                       # For Web UI & API
   uv sync --extra web --extra async-translation  # For Celery/Redis background translation
   ```

### Manual Installation (without `uv`)

If you prefer using standard Python tooling:

1. Clone the repository:
   ```bash
   git clone https://github.com/Sifr-r/LocalDeepL.git
   cd LocalDeepL
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On macOS/Linux:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate
   ```
3. Install package and requirements:
   ```bash
   pip install -e .[web,async-translation]
   ```

### Windows (Quick Start)

Double-click `install.bat`. It runs `install.ps1` to install `uv` (if missing), sync the environment, verify Docker, and create Desktop / Start Menu shortcuts.

---

## Configuration

Create a `.env` file in the repository root to configure your backend:

```env
LLM_API_BASE=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL=allenai/olmocr-2-7b
```

Additional optional settings:
- `OCR_CONCURRENCY=3` (Parallel LLM calls)
- `ALLOW_SSRF_LOCAL=true` (Allows local host loopbacks for LM Studio/Ollama)
- `REDIS_URL=redis://localhost:6379/0` (For background Celery translation)

---

## CLI Usage

```bash
uv run local-deepl input.pdf [output_ocr.pdf]
uv run local-deepl scan.png
```
*Note: If output path is omitted, it defaults to `<input_stem>_ocr.pdf`.*

### Common Commands

```bash
# Run specific pages and increase rendering DPI
uv run local-deepl input.pdf output.pdf --pages 1-3,5 --dpi 300

# Grounded bbox-native OCR (skips Surya/DP alignment)
uv run local-deepl scan.pdf --grounded --model qwen/qwen3-vl-8b

# Ollama with image dimension constraints
uv run local-deepl scan.pdf --api-base http://localhost:11434/v1 --model glm-ocr:latest --max-image-dim 640 --no-verify-model
```

---

## Web Workspace

Start the FastAPI server:
```bash
uv run local-deepl-server --port 8000
```
Open `http://localhost:8000`. The browser interface offers advanced features like adaptive binarization, spelling auto-correction, and Celery background translation.

The Advanced Configuration panel also exposes local document processors:

- **Reading Order** enables deterministic top-to-bottom, left-to-right block ordering before embedding.
- **Quality Analysis** records page-level block counts, text density, and advisory findings in the pipeline document metadata. API responses include this as an `X-Document-Quality` header when enabled.
- **Structure Analysis** classifies blocks with local heuristics for headings, paragraphs, list items, key-value lines, table candidates, and empty blocks. API responses include page-level summaries as an `X-Document-Structure` header when enabled.
- **Section Analysis** groups blocks under detected headings and carries the active section across page boundaries. API responses include page-level summaries as an `X-Document-Sections` header when enabled.

When processor metadata exists, OCR responses also include token-bound `X-Document-Metadata-Artifact-Id` and `X-Document-Metadata-Artifact-Token` headers. Fetch `GET /metadata/{artifact_id}` with that token to retrieve the compact page/block metadata report; the existing text artifact and searchable PDF outputs are unchanged.

To run Celery/Redis translation worker:
```bash
docker run -d --name redis-local-ocr -p 6379:6379 redis
uv run celery -A local_deepl.api.celery_app worker --loglevel=info -P solo
```

---

## Developer Guides

- Refer to [ARCHITECTURE.md](file:///c:/Users/rahin/LocalDeepL/ARCHITECTURE.md) for details on the pipeline orchestration, DP matching algorithm, and codebase layout.
- Refer to [AGENTS.md](file:///c:/Users/rahin/LocalDeepL/AGENTS.md) for rules on running tests, linting, formatting, and contributing.

## License

MIT
