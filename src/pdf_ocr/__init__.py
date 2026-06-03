"""
Local LLM PDF OCR - Package for OCR processing using local vision models.

Converts scanned PDFs into searchable documents using local vision LLMs
for text extraction and Surya for layout detection.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_EXPORT_MODULES = {
    "PDFHandler": "pdf_ocr.core.pdf",
    "OCRProcessor": "pdf_ocr.core.ocr",
    "HybridAligner": "pdf_ocr.core.aligner",
    "OCRPipeline": "pdf_ocr.pipeline",
    "GroundedBlock": "pdf_ocr.core.grounded",
    "GroundedResponse": "pdf_ocr.core.grounded",
    "GroundedOCRBackend": "pdf_ocr.core.grounded",
    "PromptedGroundedOCR": "pdf_ocr.core.grounded",
    "ZAIHostedOCR": "pdf_ocr.core.grounded",
    "DEFAULT_GROUNDING_PROMPT": "pdf_ocr.core.grounded",
    "parse_zai_response": "pdf_ocr.core.grounded",
    "parse_glm_layout_details": "pdf_ocr.core.grounded",
    "parse_page_range": "pdf_ocr.pipeline",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'pdf_ocr' has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORT_MODULES])


__all__ = [
    "PDFHandler",
    "OCRProcessor",
    "HybridAligner",
    "OCRPipeline",
    "GroundedBlock",
    "GroundedResponse",
    "GroundedOCRBackend",
    "PromptedGroundedOCR",
    "ZAIHostedOCR",
    "DEFAULT_GROUNDING_PROMPT",
    "parse_zai_response",
    "parse_glm_layout_details",
    "parse_page_range",
    "__version__",
]
