from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024

SAFE_API_BASE_ERROR = (
    "Invalid api_base. Local, private, malformed, or unresolvable endpoints are "
    "blocked unless ALLOW_SSRF_LOCAL=true is explicitly configured."
)

SERVER_ERROR_MESSAGE = "The request could not be completed. Please try again later."


@dataclass(frozen=True)
class UploadResult:
    path: str
    suffix: str
    size_bytes: int


class UploadValidationError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def detect_upload_suffix(header: bytes) -> str:
    if header.startswith(b"%PDF-"):
        return ".pdf"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if header.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if header.startswith((b"II*\x00", b"MM\x00*")):
        return ".tiff"
    if header.startswith(b"BM"):
        return ".bmp"
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
        return ".webp"
    if (
        len(header) >= 12
        and header[4:8] == b"ftyp"
        and header[8:12]
        in {
            b"avif",
            b"avis",
        }
    ):
        return ".avif"
    raise UploadValidationError("Unsupported file type.", status_code=415)


async def save_validated_upload(
    file: UploadFile,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> UploadResult:
    first_chunk = await file.read(UPLOAD_CHUNK_BYTES)
    if not first_chunk:
        raise UploadValidationError("Uploaded file is empty.")

    suffix = detect_upload_suffix(first_chunk[:64])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    input_path = tmp.name
    size = 0

    try:
        while first_chunk:
            size += len(first_chunk)
            if size > max_bytes:
                raise UploadValidationError(
                    f"File too large. Maximum size is {max_bytes // (1024 * 1024)}MB.",
                    status_code=413,
                )
            await asyncio.to_thread(tmp.write, first_chunk)
            first_chunk = await file.read(UPLOAD_CHUNK_BYTES)
    except Exception:
        tmp.close()
        try:
            Path(input_path).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove rejected upload %s", input_path)
        raise
    else:
        tmp.close()
        return UploadResult(path=input_path, suffix=suffix, size_bytes=size)


def cleanup_files(*paths: str | None) -> None:
    for path in paths:
        if not path:
            continue
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            logger.warning("Could not remove temporary file %s", path)
