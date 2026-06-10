"""DocumentResult processor interfaces and registry.

Processors run after OCR/refinement/spellcheck and before output embedding.
They receive the mutable normalized document graph, so changes to block text,
order, and metadata are visible to later processors and to export surfaces that
read ``OCRPipeline.last_document_result``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Protocol

from local_deepl.core.document import DocumentBlock, DocumentPage, DocumentResult

LOCAL_DOCUMENT_PROCESSOR_NAMES = (
    "reading_order",
    "quality_analysis",
    "structure_analysis",
    "section_analysis",
    "layout_enrichment",
    "table_extraction",
)

_KEY_VALUE_RE = re.compile(r"^\s*([^:\n]{1,50}):\s*(\S.+)$")
_LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-*\u2022\u25e6\u2013\u2014]|\(?\d+[\).]|\(?[A-Za-z][\).])\s+"
)
_TABLE_SPLIT_RE = re.compile(r"\t+|\|+|\s{2,}")


class DocumentProcessor(Protocol):
    """Async transform contract for in-memory document handoff stages."""

    name: str

    async def process(self, document: DocumentResult) -> DocumentResult: ...


DocumentProcessorFactory = Callable[[], DocumentProcessor]


class DocumentProcessorRegistry:
    """Name-to-factory registry used by callers that expose processor choices.

    Factories should return fresh processor instances; processors may keep
    per-run state and the pipeline executes them sequentially.
    """

    def __init__(self) -> None:
        self._factories: dict[str, DocumentProcessorFactory] = {}

    @property
    def names(self) -> list[str]:
        return sorted(self._factories)

    def register(self, name: str, factory: DocumentProcessorFactory) -> None:
        key = name.strip()
        if not key:
            raise ValueError("Document processor name cannot be empty")
        if key in self._factories:
            raise ValueError(f"Document processor already registered: {key}")
        self._factories[key] = factory

    def create(self, name: str) -> DocumentProcessor:
        try:
            return self._factories[name]()
        except KeyError:
            raise KeyError(f"Unknown document processor: {name}") from None

    def create_many(self, names: Sequence[str]) -> list[DocumentProcessor]:
        return [self.create(name) for name in names]


class ReadingOrderProcessor:
    """Assign deterministic row-major order using normalized bbox positions.

    Warning:
        This mutates each page's block list in place. The row bucketing assumes
        bboxes remain normalized in ``0..1``; pixel coordinates would collapse
        unrelated rows into unstable groups.
    """

    name = "reading_order"

    def __init__(self, row_tolerance: float = 0.02) -> None:
        if row_tolerance <= 0:
            raise ValueError("row_tolerance must be positive")
        self.row_tolerance = row_tolerance

    async def process(self, document: DocumentResult) -> DocumentResult:
        for page in document.pages:
            page.blocks.sort(key=self._sort_key)
            for index, block in enumerate(page.blocks):
                block.reading_order = index
        return document

    def _sort_key(self, block: DocumentBlock) -> tuple[int, float, float]:
        x0, y0, _, _ = block.bbox
        return (round(y0 / self.row_tolerance), x0, y0)


class QualityAnalysisProcessor:
    """Attach lightweight page quality findings without rejecting the document.

    Findings are advisory metadata for UI/export decisions. They deliberately do
    not raise on sparse text or blank boxes because OCR can still produce a
    useful searchable PDF for partially readable scans.
    """

    name = "quality_analysis"

    def __init__(
        self,
        empty_block_area_threshold: float = 0.05,
        sparse_block_threshold: int = 20,
        min_chars_per_block: float = 2.0,
    ) -> None:
        if not 0 < empty_block_area_threshold <= 1:
            raise ValueError("empty_block_area_threshold must be in (0, 1]")
        if sparse_block_threshold < 1:
            raise ValueError("sparse_block_threshold must be positive")
        if min_chars_per_block < 0:
            raise ValueError("min_chars_per_block must be non-negative")
        self.empty_block_area_threshold = empty_block_area_threshold
        self.sparse_block_threshold = sparse_block_threshold
        self.min_chars_per_block = min_chars_per_block

    async def process(self, document: DocumentResult) -> DocumentResult:
        for page in document.pages:
            block_count = len(page.blocks)
            text_char_count = sum(len(block.text.strip()) for block in page.blocks)
            bbox_area = sum(_bbox_area(block.bbox) for block in page.blocks)
            chars_per_block = text_char_count / block_count if block_count else 0.0
            findings: list[dict[str, object]] = []
            if text_char_count == 0:
                findings.append({"code": "empty_page", "severity": "warning"})
            if (
                block_count >= self.sparse_block_threshold
                and chars_per_block < self.min_chars_per_block
            ):
                findings.append({"code": "sparse_text", "severity": "warning"})
            for index, block in enumerate(page.blocks):
                area = _bbox_area(block.bbox)
                if not block.text.strip() and area >= self.empty_block_area_threshold:
                    findings.append(
                        {
                            "code": "empty_large_block",
                            "severity": "warning",
                            "block_index": index,
                        }
                    )
            page.metadata["quality"] = {
                "block_count": block_count,
                "text_char_count": text_char_count,
                "text_density": text_char_count / bbox_area if bbox_area else 0.0,
                "findings": findings,
            }
        return document


class StructureAnalysisProcessor:
    """Attach deterministic block structure hints for local document intelligence."""

    name = "structure_analysis"

    def __init__(
        self,
        heading_max_chars: int = 90,
        heading_max_words: int = 12,
        table_min_columns: int = 3,
    ) -> None:
        if heading_max_chars < 1:
            raise ValueError("heading_max_chars must be positive")
        if heading_max_words < 1:
            raise ValueError("heading_max_words must be positive")
        if table_min_columns < 2:
            raise ValueError("table_min_columns must be at least 2")
        self.heading_max_chars = heading_max_chars
        self.heading_max_words = heading_max_words
        self.table_min_columns = table_min_columns

    async def process(self, document: DocumentResult) -> DocumentResult:
        for page in document.pages:
            counts: Counter[str] = Counter()
            for block in page.blocks:
                kind, confidence, signals = self._classify(block)
                block.kind = kind
                block.metadata["structure"] = {
                    "kind": kind,
                    "confidence": confidence,
                    "signals": signals,
                }
                counts[kind] += 1

            page.metadata["structure"] = {
                "block_kinds": dict(sorted(counts.items())),
                "has_key_values": counts["key_value"] > 0,
                "has_tables": counts["table_candidate"] > 0,
            }
        return document

    def _classify(self, block: DocumentBlock) -> tuple[str, float, list[str]]:
        text = block.text.strip()
        if not text:
            return "empty", 1.0, ["blank_text"]

        if _LIST_ITEM_RE.match(text):
            return "list_item", 0.86, ["list_marker"]

        key_value = _KEY_VALUE_RE.match(text)
        if key_value and len(key_value.group(1).strip().split()) <= 6:
            return "key_value", 0.84, ["colon_key_value"]

        columns = [part.strip() for part in _TABLE_SPLIT_RE.split(text) if part.strip()]
        if len(columns) >= self.table_min_columns:
            return "table_candidate", 0.76, ["column_separators"]

        words = text.split()
        if self._looks_like_heading(text, words):
            return "heading", 0.68, ["short_prominent_text"]

        return "paragraph", 0.55, ["default_text"]

    def _looks_like_heading(self, text: str, words: list[str]) -> bool:
        if "\n" in text or len(text) > self.heading_max_chars:
            return False
        if len(words) > self.heading_max_words:
            return False
        if text.endswith((".", ",", ";", ":")):
            return False

        letters = [char for char in text if char.isalpha()]
        if not letters:
            return False
        uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
        title_words = sum(word[:1].isupper() for word in words if word[:1].isalpha())
        return uppercase_ratio >= 0.65 or title_words >= max(1, len(words) // 2)


class SectionAnalysisProcessor:
    """Group blocks under locally detected section headings."""

    name = "section_analysis"

    def __init__(
        self, heading_max_chars: int = 120, heading_max_words: int = 14
    ) -> None:
        if heading_max_chars < 1:
            raise ValueError("heading_max_chars must be positive")
        if heading_max_words < 1:
            raise ValueError("heading_max_words must be positive")
        self.heading_max_chars = heading_max_chars
        self.heading_max_words = heading_max_words

    async def process(self, document: DocumentResult) -> DocumentResult:
        current_section: dict[str, object] | None = None
        section_index = -1

        for page in document.pages:
            page_headings: list[dict[str, object]] = []
            for block_index, block in enumerate(page.blocks):
                if self._is_heading(block):
                    section_index += 1
                    title = _normalize_space(block.text)
                    current_section = {
                        "section_index": section_index,
                        "title": title,
                        "heading_page_index": page.page_index,
                        "heading_block_index": block_index,
                    }
                    page_headings.append(dict(current_section))
                    block.metadata["section"] = {
                        **current_section,
                        "role": "heading",
                    }
                    continue

                if current_section is None:
                    block.metadata["section"] = {
                        "section_index": None,
                        "title": None,
                        "heading_page_index": None,
                        "heading_block_index": None,
                        "role": "unsectioned",
                    }
                else:
                    block.metadata["section"] = {
                        **current_section,
                        "role": "body",
                    }

            page.metadata["sections"] = {
                "headings": page_headings,
                "section_count": len(page_headings),
                "active_section": current_section["title"]
                if current_section is not None
                else None,
            }

        return document

    def _is_heading(self, block: DocumentBlock) -> bool:
        kind = _structure_kind(block)
        if kind == "heading":
            return True
        if kind not in {"text", "paragraph"}:
            return False

        text = block.text.strip()
        if not text or "\n" in text:
            return False
        if len(text) > self.heading_max_chars:
            return False
        words = text.split()
        if len(words) > self.heading_max_words:
            return False
        if text.endswith((".", ",", ";", ":")):
            return False
        if _LIST_ITEM_RE.match(text) or _KEY_VALUE_RE.match(text):
            return False
        columns = [part.strip() for part in _TABLE_SPLIT_RE.split(text) if part.strip()]
        if len(columns) >= 3:
            return False

        letters = [char for char in text if char.isalpha()]
        if not letters:
            return False
        uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
        title_words = sum(word[:1].isupper() for word in words if word[:1].isalpha())
        return uppercase_ratio >= 0.65 or title_words >= max(1, len(words) // 2)


class LayoutEnrichmentProcessor:
    """Attach local page-region and document-layout labels to blocks."""

    name = "layout_enrichment"

    async def process(self, document: DocumentResult) -> DocumentResult:
        for page in document.pages:
            counts: Counter[str] = Counter()
            for block in page.blocks:
                role, region, confidence, signals = self._classify(block)
                block.metadata["layout"] = {
                    "role": role,
                    "region": region,
                    "confidence": confidence,
                    "signals": signals,
                }
                counts[role] += 1

            page.metadata["layout"] = {
                "roles": dict(sorted(counts.items())),
                "has_figures": counts["figure"] > 0,
                "has_captions": counts["caption"] > 0,
                "has_headers": counts["header"] > 0,
                "has_footers": counts["footer"] > 0,
            }
        return document

    def _classify(self, block: DocumentBlock) -> tuple[str, str, float, list[str]]:
        text = _normalize_space(block.text)
        lower = text.lower()
        x0, y0, x1, _y1 = block.bbox
        region = _page_region(block.bbox)
        kind = _structure_kind(block)
        signals: list[str] = [f"region:{region}"]

        if region == "header" and text and len(text) <= 120:
            return "header", region, 0.72, signals + ["top_short_text"]
        if region == "footer":
            if text.isdecimal() or lower.startswith(("page ", "p. ")):
                return "page_number", region, 0.82, signals + ["footer_page_number"]
            if text and len(text) <= 140:
                return "footer", region, 0.7, signals + ["bottom_short_text"]
        if lower.startswith(("figure ", "fig. ", "table ", "caption:")):
            return "caption", region, 0.82, signals + ["caption_prefix"]
        if kind == "heading" and y0 < 0.28:
            return "title_block", region, 0.76, signals + ["early_heading"]
        if not text and _bbox_area(block.bbox) >= 0.08:
            return "figure", region, 0.58, signals + ["large_empty_region"]

        width = x1 - x0
        if width < 0.32:
            region = f"{region}_side"
            signals.append("narrow_column")
        return "body", region, 0.5, signals


class TableExtractionProcessor:
    """Extract simple local table structures from aligned OCR boxes."""

    name = "table_extraction"

    def __init__(self, row_tolerance: float = 0.018, min_columns: int = 2):
        if row_tolerance <= 0:
            raise ValueError("row_tolerance must be positive")
        if min_columns < 2:
            raise ValueError("min_columns must be at least 2")
        self.row_tolerance = row_tolerance
        self.min_columns = min_columns

    async def process(self, document: DocumentResult) -> DocumentResult:
        for page in document.pages:
            candidate_indices = [
                index
                for index, block in enumerate(page.blocks)
                if self._is_candidate(block)
            ]
            page.metadata["tables"] = self._extract_page_tables(page, candidate_indices)
        return document

    def _is_candidate(self, block: DocumentBlock) -> bool:
        """Decide whether a block is a possible table cell.

        The pre-filter has two complementary signals so the row-bucketing
        pass in :meth:`_extract_page_tables` gets a fair chance to find a
        grid:

        1. **Column-separator heuristic** — block text contains
           ``min_columns`` or more tab/pipe/multi-space columns. Catches
           aligned, whitespace-padded cell content.
        2. **Cell-shape heuristic** — block is short text in a narrow box
           (width < 35% of page, area < 8% of page). Catches grid layouts
           where each cell is a single short token, which the separator
           heuristic misses.

        We intentionally do NOT consult :func:`_structure_kind`: requiring a
        pre-classified ``table_candidate`` label from
        :class:`StructureAnalysisProcessor` made the table extraction
        silently produce empty results when a user enabled ``table_extraction``
        *without* also enabling ``structure_analysis`` in the right order
        (or, equivalently, when only ``layout_enrichment`` ran first). The
        row-tolerance and ``min_columns`` filters in
        :meth:`_extract_page_tables` protect against false positives — a
        page full of narrow short blocks won't form a 2-row × 2-column grid
        unless they're actually arranged as one.
        """
        text = _normalize_space(block.text)
        if not text:
            return False
        if len(_TABLE_SPLIT_RE.split(text)) >= self.min_columns:
            return True
        # Cell-shape: a single-token block in a narrow, small box is more
        # likely a grid cell than a paragraph. Tuned for typical Letter
        # pages: a cell occupies < 35% of page width and < 8% of page area,
        # and the token is short (≤ 24 chars after whitespace collapse).
        if len(text) > 24:
            return False
        x0, y0, x1, y1 = block.bbox
        width = max(0.0, x1 - x0)
        height = max(0.0, y1 - y0)
        area = width * height
        return width < 0.35 and area < 0.08

    def _extract_page_tables(
        self, page: DocumentPage, candidate_indices: list[int]
    ) -> list[dict[str, object]]:
        if not candidate_indices:
            return []

        rows: list[list[int]] = []
        for block_index in candidate_indices:
            block = page.blocks[block_index]
            center_y = (block.bbox[1] + block.bbox[3]) / 2
            for row in rows:
                row_center = sum(
                    (page.blocks[i].bbox[1] + page.blocks[i].bbox[3]) / 2 for i in row
                ) / len(row)
                if abs(center_y - row_center) <= self.row_tolerance:
                    row.append(block_index)
                    break
            else:
                rows.append([block_index])

        rows = [sorted(row, key=lambda i: page.blocks[i].bbox[0]) for row in rows]
        rows.sort(key=lambda row: page.blocks[row[0]].bbox[1])
        if len(rows) < 2 or max(len(row) for row in rows) < self.min_columns:
            return []

        cells: list[dict[str, object]] = []
        for row_index, row in enumerate(rows):
            for column_index, block_index in enumerate(row):
                block = page.blocks[block_index]
                block.metadata["table"] = {
                    "table_index": 0,
                    "row_index": row_index,
                    "column_index": column_index,
                }
                cells.append(
                    {
                        "row_index": row_index,
                        "column_index": column_index,
                        "block_index": block_index,
                        "text": block.text,
                        "bbox": block.bbox,
                    }
                )

        return [
            {
                "table_index": 0,
                "row_count": len(rows),
                "column_count": max(len(row) for row in rows),
                "cells": cells,
            }
        ]


def build_document_processors(names: Iterable[str]) -> tuple[DocumentProcessor, ...]:
    """Instantiate known local document processors by user-facing name."""

    registry = DocumentProcessorRegistry()
    registry.register("reading_order", ReadingOrderProcessor)
    registry.register("quality_analysis", QualityAnalysisProcessor)
    registry.register("structure_analysis", StructureAnalysisProcessor)
    registry.register("section_analysis", SectionAnalysisProcessor)
    registry.register("layout_enrichment", LayoutEnrichmentProcessor)
    registry.register("table_extraction", TableExtractionProcessor)
    return tuple(registry.create(name) for name in names)


def _structure_kind(block: DocumentBlock) -> str:
    structure = block.metadata.get("structure")
    if isinstance(structure, dict):
        kind = structure.get("kind")
        if isinstance(kind, str):
            return kind
    return block.kind


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def _page_region(bbox: Sequence[float]) -> str:
    _x0, y0, _x1, y1 = bbox
    if y1 <= 0.16:
        return "header"
    if y0 >= 0.84:
        return "footer"
    return "body"


def _bbox_area(bbox: Sequence[float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


async def run_document_processors(
    document: DocumentResult, processors: Sequence[DocumentProcessor]
) -> DocumentResult:
    """Run processors in order, passing each mutation to the next stage."""

    result = document
    for processor in processors:
        result = await processor.process(result)
    return result
