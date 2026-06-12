import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Sequence
from typing import Any, cast

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from starlette.background import BackgroundTask

from local_deepl import (
    HybridAligner,
    OCRPipeline,
    OCRProcessor,
    PDFHandler,
    PromptedGroundedOCR,
    build_document_processors,
)
from local_deepl.api.schemas import ProcessSettings
from local_deepl.api.services.artifacts import PageText, TextArtifactHandle
from local_deepl.api.services.document_metadata import (
    build_document_metadata_report,
    write_document_metadata_atomic,
)
from local_deepl.api.services.jobs import JobStatus
from local_deepl.api.services.security import (
    SAFE_API_BASE_ERROR,
    UploadValidationError,
    save_validated_upload,
)
from local_deepl.api.services.workflow import build_workflow_summary
from local_deepl.core.preprocessing import (
    LocalPagePreprocessor,
    PagePreprocessingOptions,
)
from local_deepl.core.routing import QualityRoutingOptions
from local_deepl.utils import is_ssrf_target

from . import state
from .common import _cleanup, _stable_server_error
from .config import _config
from .websocket import manager

router = APIRouter()
logger = logging.getLogger(__name__)


def _validation_error_response(exc: ValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request parameters.",
            "detail": exc.errors(include_context=False),
        },
    )


def _document_quality_header(pipeline: OCRPipeline) -> str | None:
    document = getattr(pipeline, "last_document_result", None)
    if document is None:
        return None

    pages = []
    for page in document.pages:
        quality = page.metadata.get("quality")
        if isinstance(quality, dict):
            pages.append({"page_index": page.page_index, "quality": quality})

    if not pages:
        return None
    return json.dumps({"pages": pages}, separators=(",", ":"), sort_keys=True)


def _document_structure_header(pipeline: OCRPipeline) -> str | None:
    document = getattr(pipeline, "last_document_result", None)
    if document is None:
        return None

    pages = []
    for page in document.pages:
        structure = page.metadata.get("structure")
        if isinstance(structure, dict):
            pages.append({"page_index": page.page_index, "structure": structure})

    if not pages:
        return None
    return json.dumps({"pages": pages}, separators=(",", ":"), sort_keys=True)


def _document_sections_header(pipeline: OCRPipeline) -> str | None:
    document = getattr(pipeline, "last_document_result", None)
    if document is None:
        return None

    pages = []
    for page in document.pages:
        sections = page.metadata.get("sections")
        if isinstance(sections, dict):
            pages.append({"page_index": page.page_index, "sections": sections})

    if not pages:
        return None
    return json.dumps({"pages": pages}, separators=(",", ":"), sort_keys=True)


async def _create_document_metadata_artifact(
    pipeline: OCRPipeline,
) -> TextArtifactHandle | None:
    report = build_document_metadata_report(
        getattr(pipeline, "last_document_result", None)
    )
    if report is None:
        return None

    artifact_id = state.metadata_artifacts.issue_id()
    token = state.metadata_artifacts.issue_token()
    path = await asyncio.to_thread(
        write_document_metadata_atomic,
        report,
        directory=state.metadata_artifacts.artifact_dir,
        artifact_id=artifact_id,
    )
    return state.metadata_artifacts.put(artifact_id=artifact_id, token=token, path=path)


# ---------------------------------------------------------------------------
# In-memory job history – capped at 50 entries (FIFO)
# ---------------------------------------------------------------------------
def stage_to_percent(stage: str, current: int, total: int) -> int:
    """Map a pipeline stage + sub-progress into a 0-100 overall percent."""
    return state.progress_service.stage_to_percent(stage, current, total)


def _record_job(
    job_id: str,
    filename: str,
    model: str,
    pipeline_mode: str,
    pages: str | None,
    duration_s: float,
    status: JobStatus,
    failed_pages: Sequence[int] = (),
) -> None:
    """Append a validated job record to the capped in-memory history.

    ``failed_pages`` is the 0-indexed list of pages whose OCR call
    raised an exception that the pipeline caught at its per-page
    isolation boundary. Empty in the common case — the job history
    omits the field from the serialized record when it's empty so
    existing clients see the same shape as before.
    """
    state.job_history.record(
        job_id=job_id,
        filename=filename,
        model=model,
        pipeline_mode=pipeline_mode,
        pages=pages,
        duration_s=duration_s,
        status=status,
        failed_pages=failed_pages,
    )


# ---- PDF / image processing ----------------------------------------------


@router.post("/process")
async def process_pdf(
    file: UploadFile = File(...),
    client_id: str | None = Form(None),
    progress_channel: str | None = Form(None),
    progress_token: str | None = Form(None),
    api_base: str | None = Form(None),
    api_key: str | None = Form(None),
    model: str | None = Form(None),
    pipeline_mode: str | None = Form(None),
    dpi: str | None = Form(None),
    concurrency: str | None = Form(None),
    dense_mode: str | None = Form(None),
    dense_threshold: str | None = Form(None),
    pages: str | None = Form(None),
    refine: str | None = Form(None),
    max_image_dim: str | None = Form(None),
    self_correction: str | None = Form(None),
    binarize: str | None = Form(None),
    dual_engine: str | None = Form(None),
    spellcheck: str | None = Form(None),
    cross_page: str | None = Form(None),
    preprocess_pages: str | None = Form(None),
    orientation_detection: str | None = Form(None),
    deskew: str | None = Form(None),
    denoise: str | None = Form(None),
    normalize_contrast: str | None = Form(None),
    crop_cleanup: str | None = Form(None),
    quality_routing: str | None = Form(None),
    document_processors: str | None = Form(None),
):
    """
    Process a PDF or image file through the OCR pipeline.

    Every optional parameter falls back to the in-memory config store when
    not supplied by the caller.
    """
    try:
        settings = ProcessSettings.model_validate(
            {
                "api_base": api_base if api_base is not None else _config["api_base"],
                "api_key": api_key if api_key is not None else _config["api_key"],
                "model": model if model is not None else _config["model"],
                "pipeline_mode": pipeline_mode
                if pipeline_mode is not None
                else _config["pipeline_mode"],
                "dpi": dpi if dpi is not None else _config["dpi"],
                "concurrency": concurrency
                if concurrency is not None
                else _config["concurrency"],
                "dense_mode": dense_mode
                if dense_mode is not None
                else _config["dense_mode"],
                "dense_threshold": dense_threshold
                if dense_threshold is not None
                else _config["dense_threshold"],
                "pages": pages,
                "refine": refine if refine is not None else _config["refine"],
                "max_image_dim": max_image_dim
                if max_image_dim is not None
                else _config["max_image_dim"],
                "self_correction": self_correction
                if self_correction is not None
                else _config["self_correction"],
                "binarize": binarize if binarize is not None else _config["binarize"],
                "dual_engine": dual_engine
                if dual_engine is not None
                else _config["dual_engine"],
                "spellcheck": spellcheck
                if spellcheck is not None
                else _config["spellcheck"],
                "cross_page": cross_page
                if cross_page is not None
                else _config["cross_page"],
                "preprocess_pages": preprocess_pages
                if preprocess_pages is not None
                else _config["preprocess_pages"],
                "orientation_detection": orientation_detection
                if orientation_detection is not None
                else _config["orientation_detection"],
                "deskew": deskew if deskew is not None else _config["deskew"],
                "denoise": denoise if denoise is not None else _config["denoise"],
                "normalize_contrast": normalize_contrast
                if normalize_contrast is not None
                else _config["normalize_contrast"],
                "crop_cleanup": crop_cleanup
                if crop_cleanup is not None
                else _config["crop_cleanup"],
                "quality_routing": quality_routing
                if quality_routing is not None
                else _config["quality_routing"],
                "document_processors": document_processors
                if document_processors is not None
                else _config["document_processors"],
            }
        )
    except ValidationError as exc:
        return _validation_error_response(exc)

    if is_ssrf_target(settings.api_base):
        return JSONResponse(status_code=403, content={"error": SAFE_API_BASE_ERROR})

    try:
        upload = await save_validated_upload(file)
    except UploadValidationError as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})

    input_path = upload.path
    progress_target = (
        progress_channel
        if manager.is_authorized(progress_channel, progress_token)
        else None
    )
    output_path = os.path.join(tempfile.gettempdir(), f"output_{uuid.uuid4()}.pdf")
    text_path: str | None = None
    job_id = uuid.uuid4().hex
    t_start = time.monotonic()

    try:
        await manager.send_progress(progress_target, "Initializing...", 5, stage="init")

        # -- Build the pipeline based on the selected mode -------------------
        processors = build_document_processors(
            processor.value for processor in settings.document_processors
        )
        preprocessing_options = PagePreprocessingOptions(
            enabled=settings.preprocess_pages,
            orientation_detection=settings.orientation_detection,
            deskew=settings.deskew,
            denoise=settings.denoise,
            normalize_contrast=settings.normalize_contrast,
            crop_cleanup=settings.crop_cleanup,
        )
        page_preprocessor = (
            LocalPagePreprocessor() if preprocessing_options.enabled else None
        )
        quality_routing_options = QualityRoutingOptions(
            enabled=settings.quality_routing
        )
        backend: Any
        if settings.pipeline_mode == "grounded":
            # Grounded path: the backend returns (bbox, text) pairs directly,
            # so we do NOT need HybridAligner (which loads Surya's detection
            # model — ~hundreds of MB on first call) or a per-page OCRProcessor
            # (which the pipeline never invokes in grounded mode). Building
            # them here is pure waste and a known tech-debt wart; the grounded
            # branch now mirrors the hybrid branch's structure 1:1.
            backend = PromptedGroundedOCR(
                api_base=settings.api_base,
                api_key=settings.api_key,
                model=settings.model,
                max_image_dim=settings.max_image_dim,
                concurrency=settings.concurrency,
            )
            pipeline = OCRPipeline(
                pdf_handler=PDFHandler(),
                grounded_backend=backend,
                document_processors=processors,
                page_preprocessor=page_preprocessor,
            )
        else:
            # Default: hybrid mode — Surya detect + LLM OCR + DP alignment.
            backend = OCRProcessor(
                api_base=settings.api_base,
                api_key=settings.api_key,
                model=settings.model,
            )
            pipeline = OCRPipeline(
                aligner=HybridAligner(),
                ocr_processor=backend,
                pdf_handler=PDFHandler(),
                document_processors=processors,
                page_preprocessor=page_preprocessor,
            )

        # Verify model
        verify = _config.get("verify_model", True)

        # Automatically skip verification for cloud models since /v1/models
        # is an LM Studio/Ollama extension.
        # LiteLLM prefixes or known cloud hosts indicate it's not a local server.
        is_cloud = (
            any(
                settings.model.startswith(prefix)
                for prefix in (
                    "openai/",
                    "anthropic/",
                    "gemini/",
                    "deepseek/",
                    "groq/",
                    "vertex_ai/",
                )
            )
            or "api.openai.com" in settings.api_base
        )
        if is_cloud:
            verify = False

        if verify:
            await backend.ensure_model_loaded()

        # -- Progress callback -----------------------------------------------
        async def on_progress(stage, current, total, message):
            await manager.send_progress(
                progress_target,
                message,
                stage_to_percent(stage, current, total),
                stage=stage,
            )

        # -- Per-page warning callback --------------------------------------
        # The pipeline catches per-page exceptions and continues; we
        # surface the failure both as a WebSocket warning frame (so the
        # UI can flag the page in real time) and on the final response /
        # job record. Status stays "complete" because the pipeline
        # degrades gracefully — the PDF is still written, with empty
        # searchable text on the failed page.
        async def on_warning(page_index, exc):
            warning_message = (
                f"OCR failed for page {page_index + 1}: {type(exc).__name__}"
            )
            # We don't know the current OCR percent at the moment of
            # failure without plumbing it through the pipeline; pass 0
            # so the warning frame doesn't push the progress bar
            # backward. The frontend renders the warning text and is
            # free to ignore ``percent`` on warning frames.
            await manager.send_progress(
                progress_target,
                warning_message,
                0,
                stage="ocr",
                warning=True,
            )

        # -- Run the pipeline ------------------------------------------------
        pages_text = await pipeline.run(
            input_path,
            output_path,
            dpi=settings.dpi,
            pages=settings.pages,
            concurrency=settings.concurrency,
            refine=settings.refine,
            max_image_dim=settings.max_image_dim,
            dense_threshold=settings.dense_threshold,
            dense_mode=settings.dense_mode,
            self_correction=settings.self_correction,
            binarize=settings.binarize,
            dual_engine=settings.dual_engine,
            spellcheck=settings.spellcheck,
            cross_page=settings.cross_page,
            preprocessing_options=preprocessing_options,
            quality_routing_options=quality_routing_options,
            progress=on_progress,
            on_warning=on_warning,
        )

        failed_pages = list(pipeline.last_failed_pages)

        # -- Persist extracted text for token-bound later retrieval ----------
        artifact_handle = await asyncio.to_thread(
            state.text_artifacts.create, cast(PageText, pages_text)
        )
        text_path = artifact_handle.path
        metadata_handle = await _create_document_metadata_artifact(pipeline)
        job_id = artifact_handle.artifact_id

        duration_s = time.monotonic() - t_start
        _record_job(
            job_id=job_id,
            filename=file.filename or "unknown",
            model=settings.model,
            pipeline_mode=settings.pipeline_mode,
            pages=settings.pages,
            duration_s=duration_s,
            status="complete",
            failed_pages=failed_pages,
        )

        if failed_pages:
            await manager.send_progress(
                progress_target,
                f"Completed with {len(failed_pages)} page failure(s).",
                100,
                stage="complete",
            )
        else:
            await manager.send_progress(
                progress_target, "Done! Preparing download...", 100, stage="complete"
            )

        response = FileResponse(
            output_path,
            media_type="application/pdf",
            filename=f"ocr_{file.filename}",
            background=BackgroundTask(_cleanup, input_path, output_path),
        )
        response.headers["X-Text-Artifact-Id"] = artifact_handle.artifact_id
        if failed_pages:
            response.headers["X-Failed-Pages"] = ",".join(str(p) for p in failed_pages)
        response.headers["X-Text-Artifact-Token"] = artifact_handle.token
        response.headers["X-Document-Workflow"] = json.dumps(
            build_workflow_summary(settings), separators=(",", ":"), sort_keys=True
        )
        if metadata_handle is not None:
            response.headers["X-Document-Metadata-Artifact-Id"] = (
                metadata_handle.artifact_id
            )
            response.headers["X-Document-Metadata-Artifact-Token"] = (
                metadata_handle.token
            )
        quality_header = _document_quality_header(pipeline)
        if quality_header is not None:
            response.headers["X-Document-Quality"] = quality_header
        structure_header = _document_structure_header(pipeline)
        if structure_header is not None:
            response.headers["X-Document-Structure"] = structure_header
        sections_header = _document_sections_header(pipeline)
        if sections_header is not None:
            response.headers["X-Document-Sections"] = sections_header
        return response

    except ValueError as ve:
        duration_s = time.monotonic() - t_start
        _record_job(
            job_id=job_id,
            filename=file.filename or "unknown",
            model=settings.model,
            pipeline_mode=settings.pipeline_mode,
            pages=settings.pages,
            duration_s=duration_s,
            status="error",
        )
        logger.warning("OCR processing rejected invalid input: %s", ve)
        await manager.send_progress(progress_target, "Invalid input.", 0, stage="error")
        _cleanup(input_path, output_path, text_path)
        return JSONResponse(status_code=400, content={"error": "Invalid input."})

    except Exception:
        duration_s = time.monotonic() - t_start
        _record_job(
            job_id=job_id,
            filename=file.filename or "unknown",
            model=settings.model,
            pipeline_mode=settings.pipeline_mode,
            pages=settings.pages,
            duration_s=duration_s,
            status="error",
        )
        logger.exception("OCR processing failed")
        await manager.send_progress(
            progress_target, "Processing failed.", 0, stage="error"
        )
        _cleanup(input_path, output_path, text_path)
        return _stable_server_error()
