"""Local document intelligence intermediate representation.

The public pipeline still writes legacy ``{page: [(bbox, text)]}`` structures,
but document processors need a richer handoff object for ordering, quality
metadata, and future extraction/export features. Keep bboxes normalized in
``0..1`` here; PDF coordinate conversion belongs at the output writer boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

BBox = list[float]


@dataclass(slots=True)
class DocumentSpan:
    text: str
    bbox: BBox | None = None
    confidence: float | None = None
    source_processor: str = "unknown"


@dataclass(slots=True)
class DocumentBlock:
    bbox: BBox
    text: str
    kind: str = "text"
    confidence: float | None = None
    source_processor: str = "ocr"
    reading_order: int | None = None
    spans: list[DocumentSpan] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentPage:
    page_index: int
    blocks: list[DocumentBlock] = field(default_factory=list)
    width: int | None = None
    height: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks if block.text.strip())


@dataclass(slots=True)
class DocumentResult:
    """Canonical in-memory handoff for optional document processors.

    Pages are zero-indexed to match the existing OCR pipeline dictionaries.
    Blocks are intentionally mutable: processors can reorder blocks, annotate
    metadata, or rewrite text before ``to_pages_data`` feeds the PDF writer.
    """

    pages: list[DocumentPage]
    source_path: str | None = None

    @classmethod
    def from_pages_data(
        cls,
        pages_data: Mapping[int, Sequence[tuple[Sequence[float], str]]],
        *,
        source_path: str | None = None,
        source_processor: str = "ocr",
    ) -> DocumentResult:
        """Build a result from legacy ``{page: [(bbox, text)]}`` payloads.

        The conversion validates every bbox up front so downstream processors can
        assume normalized ``[x0, y0, x1, y1]`` geometry. Invalid or pixel-space
        boxes raise ``ValueError`` instead of being embedded silently.
        """

        pages: list[DocumentPage] = []
        for page_index in sorted(pages_data):
            blocks = [
                DocumentBlock(
                    bbox=_normalize_bbox(bbox),
                    text=text,
                    source_processor=source_processor,
                    reading_order=reading_order,
                )
                for reading_order, (bbox, text) in enumerate(pages_data[page_index])
            ]
            pages.append(DocumentPage(page_index=page_index, blocks=blocks))
        return cls(pages=pages, source_path=source_path)

    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())

    def to_pages_data(self) -> dict[int, list[tuple[BBox, str]]]:
        return {
            page.page_index: [(block.bbox, block.text) for block in page.blocks]
            for page in self.pages
        }


def _normalize_bbox(bbox: Sequence[float]) -> BBox:
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {len(bbox)}")
    normalized = [float(value) for value in bbox]
    x0, y0, x1, y1 = normalized
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"Expected normalized bbox in 0..1, got {normalized!r}")
    return normalized
