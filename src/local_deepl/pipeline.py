"""
OCRPipeline - Shared orchestration for CLI and web entry points.

This module exposes the `OCRPipeline` class which acts as an orchestration
facade. Internally, it delegates execution to either `GroundedEngine` or
`HybridEngine` based on the configured components.
"""

from __future__ import annotations

from collections.abc import Sequence

from local_deepl.core.grounded import GroundedOCRBackend
from local_deepl.core.preprocessing import PagePreprocessingOptions, PagePreprocessor
from local_deepl.core.processors import DocumentProcessor
from local_deepl.core.routing import QualityRoutingOptions
from typing import cast
from local_deepl.core.workflows import (
    EngineBase,
    GroundedEngine,
    HybridEngine,
    OutputWriter,
    ProgressCallback,
    WarningCallback,
)


class OCRPipeline:
    def __init__(
        self,
        aligner=None,
        ocr_processor=None,
        pdf_handler=None,
        output_writer: OutputWriter | None = None,
        grounded_backend: GroundedOCRBackend | None = None,
        document_processors: Sequence[DocumentProcessor] | None = None,
        page_preprocessor: PagePreprocessor | None = None,
    ):
        self.grounded_backend = grounded_backend
        if pdf_handler is None:
            raise ValueError("pdf_handler is required (used for output writing)")
        output_writer = output_writer or pdf_handler.embed_structured_text

        self._engine: EngineBase
        if self.grounded_backend is not None:
            self._engine = GroundedEngine(
                grounded_backend=self.grounded_backend,
                output_writer=output_writer,
                document_processors=document_processors,
            )
        else:
            if aligner is None or ocr_processor is None:
                raise ValueError(
                    "Hybrid pipeline requires both `aligner` and `ocr_processor`. "
                    "Pass a `grounded_backend=...` instead to use the grounded path."
                )
            self._engine = HybridEngine(
                aligner=aligner,
                ocr_processor=ocr_processor,
                pdf_handler=pdf_handler,
                output_writer=output_writer,
                document_processors=document_processors,
                page_preprocessor=page_preprocessor,
            )

    @property
    def last_document_result(self):
        return self._engine.last_document_result

    @property
    def last_failed_pages(self):
        return self._engine.last_failed_pages

    async def run(
        self,
        input_path: str,
        output_path: str,
        *,
        dpi: int = 200,
        pages: str | None = None,
        concurrency: int = 1,
        refine: bool = True,
        max_image_dim: int = 1024,
        dense_threshold: int = 60,
        dense_mode: str = "auto",
        self_correction: bool = False,
        binarize: bool = False,
        dual_engine: bool = False,
        spellcheck: str = "none",
        cross_page: bool = False,
        preprocessing_options: PagePreprocessingOptions | None = None,
        quality_routing_options: QualityRoutingOptions | None = None,
        progress: ProgressCallback | None = None,
        on_warning: WarningCallback | None = None,
    ) -> dict[int, list[str]]:
        if self.grounded_backend is not None:
            grounded_engine = cast(GroundedEngine, self._engine)
            return await grounded_engine.execute(
                input_path=input_path,
                output_path=output_path,
                dpi=dpi,
                spellcheck=spellcheck,
                cross_page=cross_page,
                progress=progress,
                on_warning=on_warning,
            )
        else:
            hybrid_engine = cast(HybridEngine, self._engine)
            return await hybrid_engine.execute(
                input_path=input_path,
                output_path=output_path,
                dpi=dpi,
                pages=pages,
                concurrency=concurrency,
                refine=refine,
                max_image_dim=max_image_dim,
                dense_threshold=dense_threshold,
                dense_mode=dense_mode,
                self_correction=self_correction,
                binarize=binarize,
                dual_engine=dual_engine,
                spellcheck=spellcheck,
                cross_page=cross_page,
                preprocessing_options=preprocessing_options,
                quality_routing_options=quality_routing_options,
                progress=progress,
                on_warning=on_warning,
            )
