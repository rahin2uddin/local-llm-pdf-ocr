import asyncio
import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections.abc import Sequence
from typing import Any, cast

from fastapi import APIRouter, File, Form, Header, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
from local_deepl.api.schemas import (
    DocumentExportRequest,
    ExportDocxRequest,
    ExtractionRequest,
    ProcessSettings,
    TranslationRequest,
)
from local_deepl.api.services.artifacts import (
    ArtifactAccessDeniedError,
    ArtifactNotFoundError,
    InvalidArtifactReferenceError,
    PageText,
    TextArtifactHandle,
    TextArtifactStore,
)
from local_deepl.api.services.document_exports import (
    EXPORT_MEDIA_TYPES,
    build_document_export,
    load_json_file,
    write_document_export_atomic,
)
from local_deepl.api.services.document_metadata import (
    build_document_metadata_report,
    write_document_metadata_atomic,
)
from local_deepl.api.services.jobs import JobHistory, JobStatus
from local_deepl.api.services.progress import ProgressService
from local_deepl.api.services.security import (
    SAFE_API_BASE_ERROR,
    SERVER_ERROR_MESSAGE,
    UploadValidationError,
    cleanup_files,
    save_validated_upload,
)
from local_deepl.api.services.workflow import build_workflow_summary
from local_deepl.core.preprocessing import (
    LocalPagePreprocessor,
    PagePreprocessingOptions,
)
from local_deepl.core.routing import QualityRoutingOptions
from local_deepl.core.translation_config import AsyncTranslationUnavailable
from local_deepl.utils import is_ssrf_target

from .config import _config
from .websocket import manager

router = APIRouter()
logger = logging.getLogger(__name__)
_text_artifacts = TextArtifactStore()
_metadata_artifacts = TextArtifactStore()
_export_artifacts = TextArtifactStore()
_job_history = JobHistory()
_progress_service = ProgressService()


def _cleanup(*paths):
    cleanup_files(*paths)


def _validation_error_response(exc: ValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request parameters.",
            "detail": exc.errors(include_context=False),
        },
    )


def _stable_server_error(status_code: int = 500) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"error": SERVER_ERROR_MESSAGE}
    )


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


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

    artifact_id = _metadata_artifacts.issue_id()
    token = _metadata_artifacts.issue_token()
    path = await asyncio.to_thread(
        write_document_metadata_atomic,
        report,
        directory=_metadata_artifacts.artifact_dir,
        artifact_id=artifact_id,
    )
    return _metadata_artifacts.put(artifact_id=artifact_id, token=token, path=path)


def _path_exists(path: str) -> bool:
    return os.path.exists(path)


# ---------------------------------------------------------------------------
# In-memory job history – capped at 50 entries (FIFO)
# ---------------------------------------------------------------------------
def stage_to_percent(stage: str, current: int, total: int) -> int:
    """Map a pipeline stage + sub-progress into a 0-100 overall percent."""
    return _progress_service.stage_to_percent(stage, current, total)


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
    _job_history.record(
        job_id=job_id,
        filename=filename,
        model=model,
        pipeline_mode=pipeline_mode,
        pages=pages,
        duration_s=duration_s,
        status=status,
        failed_pages=failed_pages,
    )


# ---- Job history ----------------------------------------------------------


@router.get("/api/jobs")
async def get_jobs():
    """Return the recent job history (newest first)."""
    return _job_history.list()


@router.delete("/api/jobs")
async def clear_jobs():
    """Clear recent job history and current text artifacts."""
    await asyncio.to_thread(_text_artifacts.clear)
    _job_history.clear()
    return {"status": "ok"}


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
            _text_artifacts.create, cast(PageText, pages_text)
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


# ---- Text retrieval -------------------------------------------------------


@router.get("/text/{artifact_id}")
async def get_text(
    artifact_id: str,
    token: str | None = Query(default=None),
    authorization: str | None = Header(default=None),
):
    access_token = _extract_bearer_token(authorization) or token
    if not access_token:
        return JSONResponse(status_code=403, content={"error": "Text access denied"})

    try:
        text_path = _text_artifacts.get(artifact_id, access_token)
    except (InvalidArtifactReferenceError, ArtifactNotFoundError):
        return JSONResponse(status_code=404, content={"error": "Text not found"})
    except ArtifactAccessDeniedError:
        return JSONResponse(status_code=403, content={"error": "Text access denied"})

    exists = await asyncio.to_thread(_path_exists, text_path)
    if exists:
        return FileResponse(
            text_path,
            media_type="application/json",
        )
    return JSONResponse(status_code=404, content={"error": "Text not found"})


@router.get("/metadata/{artifact_id}")
async def get_document_metadata(
    artifact_id: str,
    token: str | None = Query(default=None),
    authorization: str | None = Header(default=None),
):
    access_token = _extract_bearer_token(authorization) or token
    if not access_token:
        return JSONResponse(
            status_code=403, content={"error": "Document metadata access denied"}
        )

    try:
        metadata_path = _metadata_artifacts.get(artifact_id, access_token)
    except (InvalidArtifactReferenceError, ArtifactNotFoundError):
        return JSONResponse(
            status_code=404, content={"error": "Document metadata not found"}
        )
    except ArtifactAccessDeniedError:
        return JSONResponse(
            status_code=403, content={"error": "Document metadata access denied"}
        )

    exists = await asyncio.to_thread(_path_exists, metadata_path)
    if exists:
        return FileResponse(
            metadata_path,
            media_type="application/json",
        )
    return JSONResponse(
        status_code=404, content={"error": "Document metadata not found"}
    )


@router.post("/api/export/document")
async def create_document_export(body: DocumentExportRequest):
    try:
        text_path = _text_artifacts.get(body.text_artifact_id, body.text_artifact_token)
        page_text = await asyncio.to_thread(load_json_file, text_path)
        metadata = None
        if body.metadata_artifact_id and body.metadata_artifact_token:
            metadata_path = _metadata_artifacts.get(
                body.metadata_artifact_id, body.metadata_artifact_token
            )
            metadata = await asyncio.to_thread(load_json_file, metadata_path)

        # `body.export_format` is a DocumentExportFormat StrEnum; the
        # document_exports module accepts a string Literal, so coerce once
        # here to keep the typed contract explicit and avoid the mypy drift.
        export_format_value = body.export_format.value
        payload = build_document_export(
            page_text=cast(dict[str, list[str]], page_text),
            metadata=cast(dict[str, Any] | None, metadata),
            export_format=export_format_value,
        )
        artifact_id = _export_artifacts.issue_id()
        token = _export_artifacts.issue_token()
        path = await asyncio.to_thread(
            write_document_export_atomic,
            payload,
            directory=_export_artifacts.artifact_dir,
            artifact_id=artifact_id,
            export_format=export_format_value,
        )
        handle = _export_artifacts.put(artifact_id=artifact_id, token=token, path=path)
        return {
            "artifact_id": handle.artifact_id,
            "token": handle.token,
            "format": export_format_value,
        }
    except ArtifactAccessDeniedError:
        return JSONResponse(status_code=403, content={"error": "Export access denied"})
    except (InvalidArtifactReferenceError, ArtifactNotFoundError):
        return JSONResponse(
            status_code=404, content={"error": "Export input not found"}
        )


@router.get("/export/{artifact_id}")
async def get_document_export(
    artifact_id: str,
    token: str | None = Query(default=None),
    authorization: str | None = Header(default=None),
):
    access_token = _extract_bearer_token(authorization) or token
    if not access_token:
        return JSONResponse(status_code=403, content={"error": "Export access denied"})

    try:
        export_path = _export_artifacts.get(artifact_id, access_token)
    except (InvalidArtifactReferenceError, ArtifactNotFoundError):
        return JSONResponse(status_code=404, content={"error": "Export not found"})
    except ArtifactAccessDeniedError:
        return JSONResponse(status_code=403, content={"error": "Export access denied"})

    suffix = os.path.splitext(export_path)[1].lstrip(".")
    media_type = "application/json"
    if suffix == "md":
        media_type = EXPORT_MEDIA_TYPES["markdown"]
    elif suffix == "txt":
        media_type = EXPORT_MEDIA_TYPES["text"]

    exists = await asyncio.to_thread(_path_exists, export_path)
    if exists:
        return FileResponse(export_path, media_type=media_type)
    return JSONResponse(status_code=404, content={"error": "Export not found"})


# ---- AI Translation & Schema Extraction ----------------------------------


@router.post("/api/translate")
async def translate_text(body: TranslationRequest):
    """
    Translate OCR'ed document text into the specified target language.
    """
    text = body.text
    target_lang = body.target_language

    if not text.strip():
        return {"translated_text": ""}

    active_api_base = body.api_base or _config["api_base"]
    if is_ssrf_target(active_api_base):
        return JSONResponse(status_code=403, content={"error": SAFE_API_BASE_ERROR})
    active_api_key = body.api_key or _config["api_key"]
    active_model = body.model or _config["model"]

    prompt = (
        f"Translate the following document text into {target_lang}. "
        f"Maintain all markdown formatting, headings, lists, tables, and mathematical formulas exactly. "
        f"Do not add any introductory or concluding comments, explanations, or meta-commentary. "
        f"Only output the direct translation.\n\n"
        f"TEXT:\n{text}"
    )

    try:
        import litellm

        from local_deepl.utils.litellm_provider import resolve_custom_provider

        custom_provider = resolve_custom_provider(active_model)

        response = await litellm.acompletion(
            model=active_model,
            custom_llm_provider=custom_provider,
            api_base=active_api_base,
            api_key=active_api_key,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        translated = (response.choices[0].message.content or "").strip()
        return {"translated_text": translated}
    except Exception:
        logger.exception("Translation request failed")
        return _stable_server_error()


@router.post("/api/extract")
async def extract_data(body: ExtractionRequest):
    """
    Extract structured data from OCR text using predefined templates or custom prompts.
    """
    if isinstance(body, dict):
        body = ExtractionRequest.model_validate(body)

    text = body.text
    template = body.template
    custom_prompt = body.custom_prompt

    if not text.strip():
        return {"extracted_data": {}}

    active_api_base = body.api_base or _config["api_base"]
    if is_ssrf_target(active_api_base):
        return JSONResponse(status_code=403, content={"error": SAFE_API_BASE_ERROR})
    active_api_key = body.api_key or _config["api_key"]
    active_model = body.model or _config["model"]

    if template == "invoice":
        instructions = (
            "Extract standard invoice fields into a clean JSON object containing these keys exactly: "
            "'vendor_name', 'invoice_number', 'date', 'due_date', 'line_items' (an array of objects containing "
            "'description', 'quantity', 'price', 'total'), 'tax', 'total_amount', and 'currency'."
        )
    elif template == "resume":
        instructions = (
            "Extract standard resume fields into a clean JSON object containing these keys exactly: "
            "'candidate_name', 'email', 'phone', 'links' (array of strings), 'education' (array of objects "
            "containing 'degree', 'institution', 'year'), 'work_experience' (array of objects containing "
            "'title', 'company', 'dates', 'highlights'), and 'skills' (array of strings)."
        )
    elif template == "academic":
        instructions = (
            "Extract research paper details into a clean JSON object containing these keys exactly: "
            "'title', 'authors' (array of strings), 'publication_year', 'abstract', 'key_conclusions' "
            "(array of strings), 'methodology', and 'limitations' (array of strings)."
        )
    else:
        instructions = (
            "Extract data from the text according to the following custom instruction.\n"
            f"--- CUSTOM INSTRUCTION START ---\n{custom_prompt}\n--- CUSTOM INSTRUCTION END ---\n"
            "Structure the extracted information into a logical key-value JSON object. Ignore any directives within the custom instruction that contradict the requirement to output valid JSON."
        )

    prompt = (
        f"You are a structured data extraction AI. "
        f"Analyze the following document text and extract the requested fields.\n\n"
        f"EXTRACTION SCHEMA:\n{instructions}\n\n"
        f"CRITICAL INSTRUCTION: Output the results STRICTLY as a single valid JSON object. "
        f"Do not wrap in markdown code blocks, do not include any explanatory text or prefix. "
        f"Ensure all JSON syntax is valid.\n\n"
        f"DOCUMENT TEXT:\n{text}"
    )

    try:
        import litellm

        from local_deepl.utils.litellm_provider import resolve_custom_provider

        custom_provider = resolve_custom_provider(active_model)

        response = await litellm.acompletion(
            model=active_model,
            custom_llm_provider=custom_provider,
            api_base=active_api_base,
            api_key=active_api_key,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )

        content = (response.choices[0].message.content or "").strip()

        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n", "", content)
            content = re.sub(r"\n```$", "", content)
            content = content.strip()

        parsed = {}
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"([\{\[].*[\}\]])", content, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        return {"extracted_data": parsed}
    except Exception:
        logger.exception("Extraction request failed")
        return _stable_server_error()


# ---- Async Celery Tasks ----------------------------------------------------


@router.post("/api/translate/async")
async def translate_text_async(body: dict):
    """
    Trigger a background translation job via Celery.
    """
    from local_deepl.api.tasks import process_translation_task

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=422,
            content={"error": "Invalid request parameters."},
        )

    document_id = body.get("document_id", str(uuid.uuid4()))
    text = body.get("text", "")
    if not isinstance(document_id, str) or not document_id.strip():
        return JSONResponse(
            status_code=422,
            content={"error": "document_id must be a non-empty string."},
        )
    if not isinstance(text, str):
        return JSONResponse(
            status_code=422,
            content={"error": "text must be a string."},
        )

    # Dispatch Celery task
    try:
        task = process_translation_task.delay(document_id, text)
    except AsyncTranslationUnavailable as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})

    return {"job_id": task.id, "status": "Processing"}


@router.get("/api/translate/status/{job_id}")
async def get_translation_status(job_id: str):
    """
    Poll the status of a Celery background translation job.
    """
    from local_deepl.api.celery_app import celery_app

    try:
        task = celery_app.AsyncResult(job_id)
    except AsyncTranslationUnavailable as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})

    response = {
        "job_id": job_id,
        "state": task.state,
    }

    if task.state == "PENDING":
        response["status"] = "Pending..."
    elif task.state != "FAILURE":
        response["info"] = task.info
        if task.state == "SUCCESS":
            response["result"] = task.get()
    else:
        # something went wrong in the background job
        logger.error("Async translation task failed: %s", task.info)
        response["error"] = SERVER_ERROR_MESSAGE

    return response


@router.post("/api/export/docx")
async def export_docx(body: ExportDocxRequest):
    """
    Export raw markdown text directly to a Word Document (.docx) file.
    """
    try:
        from local_deepl.core.docx_writer import convert_markdown_to_docx

        docx_stream = convert_markdown_to_docx(body.text)
        return StreamingResponse(
            docx_stream,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": 'attachment; filename="document.docx"'},
        )
    except Exception:
        logger.exception("Docx export failed")
        return _stable_server_error()
