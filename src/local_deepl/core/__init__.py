"""Core OCR processing modules."""

from local_deepl.core.aligner import HybridAligner
from local_deepl.core.document import DocumentBlock, DocumentPage, DocumentResult
from local_deepl.core.ocr import OCRProcessor
from local_deepl.core.pdf import PDFHandler
from local_deepl.core.processors import (
    LOCAL_DOCUMENT_PROCESSOR_NAMES,
    DocumentProcessor,
    DocumentProcessorRegistry,
    QualityAnalysisProcessor,
    ReadingOrderProcessor,
    SectionAnalysisProcessor,
    StructureAnalysisProcessor,
    build_document_processors,
    run_document_processors,
)

__all__ = [
    "PDFHandler",
    "OCRProcessor",
    "HybridAligner",
    "DocumentBlock",
    "DocumentPage",
    "DocumentResult",
    "DocumentProcessor",
    "DocumentProcessorRegistry",
    "LOCAL_DOCUMENT_PROCESSOR_NAMES",
    "QualityAnalysisProcessor",
    "ReadingOrderProcessor",
    "SectionAnalysisProcessor",
    "StructureAnalysisProcessor",
    "build_document_processors",
    "run_document_processors",
]
