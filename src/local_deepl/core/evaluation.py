from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from local_deepl.core.document import DocumentResult


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    text_similarity: float
    block_count: int
    invalid_bbox_count: int
    reading_order_coverage: float
    table_count: int

    def to_report(self) -> dict[str, Any]:
        return {
            "text_similarity": self.text_similarity,
            "block_count": self.block_count,
            "invalid_bbox_count": self.invalid_bbox_count,
            "reading_order_coverage": self.reading_order_coverage,
            "table_count": self.table_count,
        }


def evaluate_document(
    document: DocumentResult,
    *,
    expected_text: str = "",
) -> EvaluationMetrics:
    actual_text = document.text()
    text_similarity = (
        SequenceMatcher(None, expected_text, actual_text).ratio()
        if expected_text
        else 0.0
    )
    blocks = [block for page in document.pages for block in page.blocks]
    ordered = sum(1 for block in blocks if block.reading_order is not None)
    tables = sum(
        len(tables)
        for page in document.pages
        if isinstance(tables := page.metadata.get("tables"), list)
    )
    return EvaluationMetrics(
        text_similarity=text_similarity,
        block_count=len(blocks),
        invalid_bbox_count=sum(1 for block in blocks if not _valid_bbox(block.bbox)),
        reading_order_coverage=ordered / len(blocks) if blocks else 0.0,
        table_count=tables,
    )


def _valid_bbox(bbox: list[float]) -> bool:
    if len(bbox) != 4:
        return False
    x0, y0, x1, y1 = bbox
    return 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
