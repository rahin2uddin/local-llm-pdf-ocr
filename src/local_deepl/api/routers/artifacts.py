import asyncio
import logging
import os
from typing import Any, cast

from fastapi import APIRouter, Header, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from local_deepl.api.routers import state
from local_deepl.api.routers.common import (
    _extract_bearer_token,
    _path_exists,
    _stable_server_error,
)
from local_deepl.api.schemas import DocumentExportRequest, ExportDocxRequest
from local_deepl.api.services.artifacts import (
    ArtifactAccessDeniedError,
    ArtifactNotFoundError,
    InvalidArtifactReferenceError,
)
from local_deepl.api.services.document_exports import (
    EXPORT_MEDIA_TYPES,
    build_document_export,
    load_json_file,
    write_document_export_atomic,
)

router = APIRouter()
logger = logging.getLogger(__name__)


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
        text_path = state.text_artifacts.get(artifact_id, access_token)
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
        metadata_path = state.metadata_artifacts.get(artifact_id, access_token)
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
        text_path = state.text_artifacts.get(
            body.text_artifact_id, body.text_artifact_token
        )
        page_text = await asyncio.to_thread(load_json_file, text_path)
        metadata = None
        if body.metadata_artifact_id and body.metadata_artifact_token:
            metadata_path = state.metadata_artifacts.get(
                body.metadata_artifact_id, body.metadata_artifact_token
            )
            metadata = await asyncio.to_thread(load_json_file, metadata_path)

        export_format_value = body.export_format.value
        payload = build_document_export(
            page_text=cast(dict[str, list[str]], page_text),
            metadata=cast(dict[str, Any] | None, metadata),
            export_format=export_format_value,
        )
        artifact_id = state.export_artifacts.issue_id()
        token = state.export_artifacts.issue_token()
        path = await asyncio.to_thread(
            write_document_export_atomic,
            payload,
            directory=state.export_artifacts.artifact_dir,
            artifact_id=artifact_id,
            export_format=export_format_value,
        )
        handle = state.export_artifacts.put(
            artifact_id=artifact_id, token=token, path=path
        )
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
        export_path = state.export_artifacts.get(artifact_id, access_token)
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
