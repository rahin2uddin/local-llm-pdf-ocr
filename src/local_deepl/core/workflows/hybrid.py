from __future__ import annotations

import asyncio
import base64
import io
from collections import defaultdict
from collections.abc import Sequence

from PIL import Image

from local_deepl.core.document import DocumentResult
from local_deepl.core.preprocessing import PagePreprocessingOptions, PagePreprocessor
from local_deepl.core.processors import DocumentProcessor, run_document_processors
from local_deepl.core.routing import QualityRoutingOptions, QualityRoutingPolicy
from local_deepl.core.workflows.base import (
    EngineBase,
    OutputWriter,
    ProgressCallback,
    WarningCallback,
    _notify,
)
from local_deepl.utils.image import crop_for_ocr_from_image


def parse_page_range(page_str: str, total_pages: int) -> list[int]:
    """Parse a 1-indexed range like '1-3,5,7-9' into sorted 0-indexed pages."""
    pages: set[int] = set()
    try:
        for part in page_str.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start, end = int(start_s), int(end_s)
                for p in range(start, end + 1):
                    if 1 <= p <= total_pages:
                        pages.add(p - 1)
            else:
                p = int(part)
                if 1 <= p <= total_pages:
                    pages.add(p - 1)
    except ValueError as e:
        raise ValueError(f"Invalid page range syntax: '{page_str}'") from e
    return sorted(pages)


def _decode_page_image(image_b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def _normalize_for_dedup(text: str) -> str:
    return " ".join(text.lower().split())


def _drop_refined_duplicates(
    page_boxes: list[tuple[list[float], str]],
    refined_indices: set[int],
    *,
    radius: int = 4,
) -> None:
    for r_idx in sorted(refined_indices):
        r_bbox, r_text = page_boxes[r_idx]
        if not r_text:
            continue
        r_norm = _normalize_for_dedup(r_text)
        if not r_norm:
            continue
        lo = max(0, r_idx - radius)
        hi = min(len(page_boxes), r_idx + radius + 1)
        for o_idx in range(lo, hi):
            if o_idx == r_idx or o_idx in refined_indices:
                continue
            _, o_text = page_boxes[o_idx]
            if not o_text:
                continue
            o_norm = _normalize_for_dedup(o_text)
            if r_norm in o_norm:
                page_boxes[r_idx] = (r_bbox, "")
                break


def _is_refinable(bbox: list[float]) -> bool:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width > 0.03 and height > 0.008


class HybridEngine(EngineBase):
    def __init__(
        self,
        aligner,
        ocr_processor,
        pdf_handler,
        output_writer: OutputWriter,
        document_processors: Sequence[DocumentProcessor] | None = None,
        page_preprocessor: PagePreprocessor | None = None,
    ):
        self.aligner = aligner
        self.ocr_processor = ocr_processor
        self.pdf_handler = pdf_handler
        self.output_writer = output_writer
        self.document_processors = tuple(document_processors or ())
        self.page_preprocessor = page_preprocessor

        # State populated after a run
        self.last_document_result: DocumentResult | None = None
        self.last_failed_pages: list[int] = []

    async def execute(
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
        if dense_mode not in ("auto", "always", "never"):
            raise ValueError(
                f"dense_mode must be one of 'auto', 'always', 'never'; got {dense_mode!r}"
            )
        
        self.last_failed_pages = []

        await _notify(progress, "convert", 0, 1, "Converting PDF to images...")
        images_dict = await asyncio.to_thread(
            self.pdf_handler.convert_to_images, input_path, dpi, max_image_dim
        )
        page_nums = sorted(images_dict.keys())
        total_pages = len(page_nums)

        if pages:
            selected = set(parse_page_range(pages, total_pages))
            page_nums = [p for p in page_nums if p in selected]
            images_dict = {
                p: image for p, image in images_dict.items() if p in selected
            }

        preprocessing_metadata: dict[int, dict[str, object]] = {}
        if (
            self.page_preprocessor is not None
            and preprocessing_options is not None
            and preprocessing_options.enabled
        ):
            await _notify(
                progress, "convert", 0, 1, f"Preprocessing {len(page_nums)} pages..."
            )
            preprocessing_result = await asyncio.to_thread(
                self.page_preprocessor.preprocess,
                images_dict,
                preprocessing_options,
            )
            images_dict = preprocessing_result.images
            preprocessing_metadata = preprocessing_result.metadata
        await _notify(progress, "convert", 1, 1, f"Converted {total_pages} pages.")

        # --- Phase 1: batch layout detection ---
        await _notify(
            progress, "detect", 0, 1, f"Detecting layout for {len(page_nums)} pages..."
        )

        batch_boxes = []
        chunk_size = 10
        for i in range(0, len(page_nums), chunk_size):
            chunk_pages = page_nums[i : i + chunk_size]
            chunk_bytes = [base64.b64decode(images_dict[p]) for p in chunk_pages]
            chunk_boxes = await asyncio.to_thread(
                self.aligner.get_detected_boxes_batch, chunk_bytes
            )
            batch_boxes.extend(chunk_boxes)
            await _notify(
                progress,
                "detect",
                min(i + chunk_size, len(page_nums)),
                len(page_nums),
                f"Detecting layout ({min(i + chunk_size, len(page_nums))}/{len(page_nums)})...",
            )

        pages_structured: dict[int, list] = {
            p: [(box, "") for box in batch_boxes[i]] for i, p in enumerate(page_nums)
        }
        await _notify(progress, "detect", 1, 1, "Layout detection complete.")

        per_box_pages: set[int] = set()
        for p_num in page_nums:
            n_boxes = len(pages_structured[p_num])
            if dense_mode == "always" or (
                dense_mode == "auto" and n_boxes > dense_threshold
            ):
                per_box_pages.add(p_num)

        # --- Phase 2: concurrent OCR ---
        pages_text: dict[int, list[str]] = {}
        semaphore = asyncio.Semaphore(max(1, concurrency))
        total = len(page_nums)

        async def process_page(p_num: int):
            try:
                if p_num in per_box_pages:
                    aligned = await self._ocr_per_box(
                        images_dict[p_num],
                        pages_structured[p_num],
                        semaphore,
                        self_correction,
                        binarize,
                        dual_engine,
                    )
                    llm_lines = [t for _, t in aligned if t]
                    return p_num, llm_lines, aligned, None
                async with semaphore:
                    llm_lines = await self.ocr_processor.perform_ocr(
                        images_dict[p_num],
                        self_correction=self_correction,
                        binarize=binarize,
                        dual_engine=dual_engine,
                    )
                    if llm_lines:
                        aligned = await asyncio.to_thread(
                            self.aligner.align_text, pages_structured[p_num], llm_lines
                        )
                    else:
                        aligned = pages_structured[p_num]
                    return p_num, llm_lines, aligned, None
            except Exception as e:
                import logging

                logging.warning(f"OCR failed for page {p_num}: {type(e).__name__}: {e}")
                return p_num, [], pages_structured[p_num], e

        completed = 0
        ocr_label = (
            "OCR"
            if not per_box_pages
            else f"OCR ({len(per_box_pages)} dense / {total - len(per_box_pages)} sparse)"
        )
        await _notify(progress, "ocr", 0, total, f"{ocr_label} (0/{total})...")
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(process_page(p)) for p in page_nums]
            for coro in asyncio.as_completed(tasks):
                p_num, llm_lines, aligned, page_error = await coro

                pages_text[p_num] = llm_lines
                pages_structured[p_num] = aligned
                completed += 1
                await _notify(
                    progress,
                    "ocr",
                    completed,
                    total,
                    f"{ocr_label} ({completed}/{total})",
                )
                if page_error is not None:
                    self.last_failed_pages.append(p_num)
                    if on_warning is not None:
                        await on_warning(p_num, page_error)

        # --- Phase 3: per-box crop re-OCR ---
        if refine:
            sparse_structured = {
                p: pages_structured[p] for p in page_nums if p not in per_box_pages
            }
            if sparse_structured:
                await self._refine_uncertain(
                    sparse_structured,
                    images_dict,
                    semaphore,
                    progress,
                    self_correction,
                    binarize,
                    dual_engine,
                )

        # --- Phase 4: post-processing ---
        if cross_page:
            self._cross_page_merge(pages_structured, page_nums)

        if spellcheck and spellcheck != "none":
            await self._run_spellcheck(pages_structured, page_nums, spellcheck)

        document_result = DocumentResult.from_pages_data(
            pages_structured, source_path=input_path, source_processor="hybrid"
        )
        for page in document_result.pages:
            metadata = preprocessing_metadata.get(page.page_index)
            if metadata:
                page.metadata["preprocessing"] = metadata
        self.last_document_result = await run_document_processors(
            document_result, self.document_processors
        )
        if quality_routing_options is not None and quality_routing_options.enabled:
            self.last_document_result = QualityRoutingPolicy().apply(
                self.last_document_result, quality_routing_options
            )
        pages_structured = self.last_document_result.to_pages_data()
        page_nums = sorted(pages_structured)

        for p in page_nums:
            pages_text[p] = [text for _, text in pages_structured[p] if text.strip()]

        # --- Phase 5: write output ---
        await _notify(progress, "embed", 0, 1, "Writing output...")
        await asyncio.to_thread(
            self.output_writer, input_path, output_path, pages_structured, dpi
        )
        await _notify(progress, "embed", 1, 1, "Done.")
        return pages_text

    async def _ocr_per_box(
        self,
        image_b64: str,
        structured: list[tuple[list[float], str]],
        semaphore: asyncio.Semaphore,
        self_correction: bool = False,
        binarize: bool = False,
        dual_engine: bool = False,
    ) -> list[tuple[list[float], str]]:
        page_image = await asyncio.to_thread(_decode_page_image, image_b64)

        async def ocr_one(idx: int, bbox: list[float]):
            try:
                async with semaphore:
                    if not _is_refinable(bbox):
                        return idx, ""
                    crop_b64 = await asyncio.to_thread(
                        crop_for_ocr_from_image, page_image, bbox
                    )
                    if crop_b64 is None:
                        return idx, ""
                    text = await self.ocr_processor.perform_ocr_on_crop(
                        crop_b64,
                        self_correction=self_correction,
                        binarize=binarize,
                        dual_engine=dual_engine,
                    )
                    return idx, text
            except Exception as e:
                import logging

                logging.warning(
                    f"Dense OCR failed for box {idx}: {type(e).__name__}: {e}"
                )
                return idx, ""

        results: dict[int, str] = {}
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(ocr_one(i, bbox))
                for i, (bbox, _) in enumerate(structured)
            ]
            for fut in asyncio.as_completed(tasks):
                idx, text = await fut
                results[idx] = text.strip()
        return [(bbox, results.get(i, "")) for i, (bbox, _) in enumerate(structured)]

    async def _refine_uncertain(
        self,
        sparse_structured: dict[int, list[tuple[list[float], str]]],
        images_dict: dict[int, str],
        semaphore: asyncio.Semaphore,
        progress: ProgressCallback | None,
        self_correction: bool = False,
        binarize: bool = False,
        dual_engine: bool = False,
    ) -> None:
        targets: list[tuple[int, int, list[float]]] = []
        for p_num, aligned in sparse_structured.items():
            for idx, (bbox, text) in enumerate(aligned):
                if not text.strip() and _is_refinable(bbox):
                    targets.append((p_num, idx, bbox))

        if not targets:
            return

        total = len(targets)
        await _notify(
            progress, "refine", 0, total, f"Refining {total} uncertain boxes..."
        )

        page_images: dict[int, Image.Image] = {}
        pages_needed = {p_num for p_num, _, _ in targets}
        for p_num in pages_needed:
            page_images[p_num] = await asyncio.to_thread(
                _decode_page_image, images_dict[p_num]
            )

        async def refine_one(p_num: int, idx: int, bbox: list[float]):
            try:
                async with semaphore:
                    crop_b64 = await asyncio.to_thread(
                        crop_for_ocr_from_image, page_images[p_num], bbox
                    )
                    if crop_b64 is None:
                        return p_num, idx, ""
                    text = await self.ocr_processor.perform_ocr_on_crop(
                        crop_b64,
                        self_correction=self_correction,
                        binarize=binarize,
                        dual_engine=dual_engine,
                    )
                    return p_num, idx, text
            except Exception as e:
                import logging

                logging.warning(
                    f"Refine failed for page {p_num} box {idx}: {type(e).__name__}: {e}"
                )
                return p_num, idx, ""

        completed = 0
        refined_indices: dict[int, set[int]] = defaultdict(set)
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(refine_one(p, i, b)) for p, i, b in targets]
            for coro in asyncio.as_completed(tasks):
                p_num, idx, text = await coro
                bbox_cur, _ = sparse_structured[p_num][idx]
                sparse_structured[p_num][idx] = (bbox_cur, text.strip())
                refined_indices[p_num].add(idx)
                completed += 1
                await _notify(
                    progress,
                    "refine",
                    completed,
                    total,
                    f"Refining boxes ({completed}/{total})",
                )

        for p_num, idxs in refined_indices.items():
            _drop_refined_duplicates(sparse_structured[p_num], idxs)
