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
    "PDFHandler": "local_deepl.core.pdf",
    "OCRProcessor": "local_deepl.core.ocr",
    "HybridAligner": "local_deepl.core.aligner",
    "DocumentBlock": "local_deepl.core.document",
    "DocumentPage": "local_deepl.core.document",
    "DocumentResult": "local_deepl.core.document",
    "DocumentProcessor": "local_deepl.core.processors",
    "DocumentProcessorRegistry": "local_deepl.core.processors",
    "LOCAL_DOCUMENT_PROCESSOR_NAMES": "local_deepl.core.processors",
    "QualityAnalysisProcessor": "local_deepl.core.processors",
    "ReadingOrderProcessor": "local_deepl.core.processors",
    "SectionAnalysisProcessor": "local_deepl.core.processors",
    "StructureAnalysisProcessor": "local_deepl.core.processors",
    "build_document_processors": "local_deepl.core.processors",
    "run_document_processors": "local_deepl.core.processors",
    "OCRPipeline": "local_deepl.pipeline",
    "GroundedBlock": "local_deepl.core.grounded",
    "GroundedResponse": "local_deepl.core.grounded",
    "GroundedOCRBackend": "local_deepl.core.grounded",
    "PromptedGroundedOCR": "local_deepl.core.grounded",
    "ZAIHostedOCR": "local_deepl.core.grounded",
    "DEFAULT_GROUNDING_PROMPT": "local_deepl.core.grounded",
    "parse_zai_response": "local_deepl.core.grounded",
    "parse_glm_layout_details": "local_deepl.core.grounded",
    "parse_page_range": "local_deepl.pipeline",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'local_deepl' has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORT_MODULES])


__all__ = [
    "PDFHandler",
    "OCRProcessor",
    "HybridAligner",
    "DocumentBlock",
    "DocumentPage",
    "DocumentResult",
    "DocumentProcessor",
    "DocumentProcessorRegistry",
    "QualityAnalysisProcessor",
    "ReadingOrderProcessor",
    "run_document_processors",
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
