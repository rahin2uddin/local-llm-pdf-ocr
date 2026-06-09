# Scribe Handover Audit

Date: 2026-06-09

## Executive Summary

This pass focused on the OCR orchestration boundary where hybrid and grounded
pipelines now converge into a normalized `DocumentResult`. That handoff is the
main architectural seam for future reading-order, quality, extraction, and
export work because it sits after OCR cleanup but before PDF embedding.

## Triage Signals

- Complexity analyzer: a read-only AST pass flagged `OCRPipeline.run`,
  `HybridAligner._dp_align`, `api/routers/ocr.py::process_pdf`,
  `translation.chunk_text`, and the new document processors as high-context
  functions.
- Coverage proxy: direct test references are strongest around aligner and API
  safety modules; the newer `core/document.py` and `core/processors.py` rely on
  a smaller, focused test surface in `tests/test_document.py`.
- Git historian: recent uncommitted work introduces the `DocumentResult`
  processor chain and a Surya detection retry. The older committed history is
  shallow, so handover notes should live close to these seams.

## Findings

- `DocumentResult` is a canonical handoff, not a replacement for the legacy
  pages-data writer API. Processors may mutate block text, order, and metadata;
  `to_pages_data()` converts the result back for PDF embedding.
- Bounding boxes must remain normalized `[x0, y0, x1, y1]` in `0..1` until the
  output writer. Pixel-space geometry at the document-processor layer will break
  reading-order bucketing and downstream embedding assumptions.
- Document processors run after cross-page merge and spellcheck. Their changes
  are therefore the last text/order mutations before the searchable layer is
  written.
- Grounded OCR bypasses Surya, DP alignment, and refine, then rejoins the same
  document processor chain as hybrid OCR. New processor features should avoid
  backend-specific branches unless they depend on unavailable hybrid metadata.
- The Surya detection retry only fires when every page in a batch has no boxes.
  A mixed batch with real blank pages should preserve those blank pages instead
  of retrying the whole batch.
- `api/routers/ocr.py` remains the broadest handover risk: one router handles
  upload validation, pipeline construction, progress, translation, extraction,
  and artifact responses. Prefer moving new behavior into services or pipeline
  collaborators before adding more route-local branching.

## Documentation Added

- `core/document.py`: `DocumentResult` and `from_pages_data` notes for the
  normalized handoff model, mutability, zero-indexed pages, and bbox validation.
- `core/processors.py`: processor contract notes covering sequential mutation,
  reading-order assumptions, quality metadata behavior, and registry factory
  expectations.
- `pipeline.py`: comments marking the exact point where document processors
  mutate the payload that feeds PDF embedding.
- `core/aligner.py`: comment explaining the all-empty Surya detection retry and
  why mixed batches are not retried.

## Stage 1 Status

- Web/API OCR requests can now select local deterministic document processors
  through `document_processors`.
- Built-in selectable processors are `reading_order` and `quality_analysis`.
- The web Advanced Configuration panel exposes Reading Order and Quality
  Analysis toggles.
- `quality_analysis` metadata is additive. Searchable PDF output and text
  artifacts remain unchanged; API responses include `X-Document-Quality` when
  quality metadata exists.
- Focused selection tests live in `tests/test_document_processor_selection.py`.

## Stage 2 Status

- Added `structure_analysis`, a local deterministic document processor that
  classifies blocks as `heading`, `paragraph`, `list_item`, `key_value`,
  `table_candidate`, or `empty`.
- The processor stores block-level structure metadata on each
  `DocumentBlock.metadata["structure"]` and page-level summaries on
  `DocumentPage.metadata["structure"]`.
- Web/API selection uses the existing `document_processors` field. The web
  Advanced Configuration panel now includes a Structure Analysis toggle.
- API responses include compact page summaries in `X-Document-Structure` when
  structure metadata exists.
- The processor does not rewrite block text or change searchable PDF output.

## Next Stage Prep

- Keep LangGraph orchestration outside processor internals. Future stages should use
  plain Python processors and deterministic `DocumentResult` transforms.
- Natural next targets are section/header grouping, richer table detection,
  key-value extraction scaffolding, or export/report surfaces for
  `DocumentResult` metadata.
- Prefer extending `core/processors.py` and the existing
  `document_processors` API field before adding new route-specific flags.
- If metadata grows beyond compact headers, add a token-bound metadata artifact
  endpoint rather than changing the existing text artifact shape.
- Preserve defaults: no document processors run unless explicitly selected.

## Follow-On Audit Targets

- Add a focused handover pass for `api/routers/ocr.py` before expanding web OCR
  features; it is the highest coupling point.
- Document `translation.chunk_text` and async translation optional dependency
  behavior when translation work resumes.
- Consider a real coverage report once `pytest-cov` or `coverage` is available
  in the dev dependency set.
