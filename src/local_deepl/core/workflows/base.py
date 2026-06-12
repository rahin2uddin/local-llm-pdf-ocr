from __future__ import annotations

from collections.abc import Awaitable, Callable

ProgressCallback = Callable[[str, int, int, str], Awaitable[None]]
WarningCallback = Callable[[int, BaseException], Awaitable[None]]
OutputWriter = Callable[[str, str, dict, int], None]


async def _notify(
    cb: ProgressCallback | None, stage: str, current: int, total: int, message: str
) -> None:
    if cb is not None:
        await cb(stage, current, total, message)


class EngineBase:
    """
    Base class for OCR workflows (Hybrid and Grounded).
    Provides shared post-processing orchestration such as cross-page merges,
    spellcheck, document-processor execution, and output writing.
    """

    def _cross_page_merge(
        self,
        pages_structured: dict[int, list[tuple[list[float], str]]],
        page_nums: list[int],
    ) -> None:
        """
        Post-processing step that inspects the end of each page and merges
        trailing sentences without terminal punctuation into the first line of the
        subsequent page.
        """
        for i in range(len(page_nums) - 1):
            p1 = page_nums[i]
            p2 = page_nums[i + 1]

            p1_boxes = pages_structured.get(p1, [])
            last_idx = -1
            for idx in range(len(p1_boxes) - 1, -1, -1):
                if p1_boxes[idx][1].strip():
                    last_idx = idx
                    break

            p2_boxes = pages_structured.get(p2, [])
            first_idx = -1
            for idx in range(len(p2_boxes)):
                if p2_boxes[idx][1].strip():
                    first_idx = idx
                    break

            if last_idx != -1 and first_idx != -1:
                last_bbox, last_text = p1_boxes[last_idx]
                first_bbox, first_text = p2_boxes[first_idx]

                last_text_stripped = last_text.strip()
                # If the last box's text does not end with sentence-ending punctuation, merge them.
                if last_text_stripped and last_text_stripped[-1] not in (".", "!", "?"):
                    merged_text = last_text_stripped + " " + first_text.strip()
                    p2_boxes[first_idx] = (first_bbox, merged_text)
                    p1_boxes[last_idx] = (last_bbox, "")

    async def _run_spellcheck(
        self,
        pages_structured: dict[int, list[tuple[list[float], str]]],
        page_nums: list[int],
        lang: str,
    ) -> None:
        """
        Post-processing step that runs spelling auto-correction on each page.
        """
        from local_deepl.core.postprocess import DictionaryPostProcessor

        processor = DictionaryPostProcessor(lang)
        await processor.ensure_loaded()
        for p in page_nums:
            corrected = []
            for bbox, text in pages_structured[p]:
                if text:
                    corrected.append((bbox, processor.correct_text(text)))
                else:
                    corrected.append((bbox, text))
            pages_structured[p] = corrected
