from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Sequence

from local_deepl.core.document import DocumentResult
from local_deepl.core.grounded import GroundedOCRBackend
from local_deepl.core.processors import DocumentProcessor, run_document_processors
from local_deepl.core.workflows.base import (
    EngineBase,
    OutputWriter,
    ProgressCallback,
    WarningCallback,
    _notify,
)


class GroundedEngine(EngineBase):
    def __init__(
        self,
        grounded_backend: GroundedOCRBackend,
        output_writer: OutputWriter,
        document_processors: Sequence[DocumentProcessor] | None = None,
    ):
        self.grounded_backend = grounded_backend
        self.output_writer = output_writer
        self.document_processors = tuple(document_processors or ())

        # State populated after a run
        self.last_document_result: DocumentResult | None = None
        self.last_failed_pages: list[int] = []

    async def execute(
        self,
        input_path: str,
        output_path: str,
        *,
        dpi: int,
        spellcheck: str = "none",
        cross_page: bool = False,
        progress: ProgressCallback | None = None,
        on_warning: WarningCallback | None = None,
    ) -> dict[int, list[str]]:
        """
        Grounded path: the backend returns (bbox, text) pairs directly.
        No Surya, no DP, no refine — the model already knows where the text is.
        """
        self.last_failed_pages = []
        response = await self.grounded_backend.ocr_document(
            input_path, progress=progress, on_warning=on_warning
        )
        if response.failed_pages:
            self.last_failed_pages.extend(response.failed_pages)

        pages_data: dict[int, list[tuple[list[float], str]]] = defaultdict(list)
        for block in response.blocks:
            pages_data[block.page_index].append((block.bbox, block.text))

        page_nums = sorted(pages_data.keys())

        if cross_page:
            self._cross_page_merge(pages_data, page_nums)

        if spellcheck and spellcheck != "none":
            await self._run_spellcheck(pages_data, page_nums, spellcheck)

        document_result = DocumentResult.from_pages_data(
            dict(pages_data), source_path=input_path, source_processor="grounded"
        )
        self.last_document_result = await run_document_processors(
            document_result, self.document_processors
        )
        pages_data = defaultdict(list, self.last_document_result.to_pages_data())
        page_nums = sorted(pages_data)

        pages_text: dict[int, list[str]] = defaultdict(list)
        for p in page_nums:
            pages_text[p] = [text for _, text in pages_data[p] if text.strip()]

        await _notify(progress, "embed", 0, 1, "Writing output...")
        await asyncio.to_thread(
            self.output_writer, input_path, output_path, dict(pages_data), dpi
        )
        await _notify(progress, "embed", 1, 1, "Done.")
        return dict(pages_text)
