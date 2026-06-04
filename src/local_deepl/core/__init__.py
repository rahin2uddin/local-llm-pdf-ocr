"""Core OCR processing modules."""

from local_deepl.core.aligner import HybridAligner
from local_deepl.core.ocr import OCRProcessor
from local_deepl.core.pdf import PDFHandler

__all__ = ["PDFHandler", "OCRProcessor", "HybridAligner"]
