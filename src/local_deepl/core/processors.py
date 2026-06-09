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

from local_deepl.core.document import DocumentBlock, DocumentResult

LOCAL_DOCUMENT_PROCESSOR_NAMES = (
    "reading_order",
    "quality_analysis",
    "structure_analysis",
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


def build_document_processors(names: Iterable[str]) -> tuple[DocumentProcessor, ...]:
    """Instantiate known local document processors by user-facing name."""

    registry = DocumentProcessorRegistry()
    registry.register("reading_order", ReadingOrderProcessor)
    registry.register("quality_analysis", QualityAnalysisProcessor)
    registry.register("structure_analysis", StructureAnalysisProcessor)
    return tuple(registry.create(name) for name in names)


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
