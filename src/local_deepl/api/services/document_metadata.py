from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_deepl.api.services.artifacts import (
    InvalidArtifactPayloadError,
    InvalidArtifactReferenceError,
    is_opaque_artifact_id,
)

if TYPE_CHECKING:
    from local_deepl.core.document import DocumentBlock, DocumentResult

DOCUMENT_METADATA_ARTIFACT_PREFIX = "metadata"
DOCUMENT_METADATA_REPORT_VERSION = 1
_REPORT_METADATA_KEYS = ("quality", "structure", "sections")
_BLOCK_METADATA_KEYS = ("structure", "section")


def build_document_metadata_report(
    document: "DocumentResult | None",
) -> dict[str, Any] | None:
    """Build a compact, JSON-safe report from local document processor outputs."""
    if document is None:
        return None

    pages: list[dict[str, Any]] = []
    processors: set[str] = set()

    for page in document.pages:
        page_metadata = {
            key: _json_safe_value(value)
            for key in _REPORT_METADATA_KEYS
            if (value := page.metadata.get(key)) is not None
        }
        processors.update(_processors_for_page_metadata(page_metadata))

        block_reports: list[dict[str, Any]] = []
        for block_index, block in enumerate(page.blocks):
            block_report = _block_metadata_report(block_index, block)
            if block_report is not None:
                block_reports.append(block_report)
                processors.update(_processors_for_block_report(block_report))

        if not page_metadata and not block_reports:
            continue

        page_report: dict[str, Any] = {"page_index": page.page_index}
        if page_metadata:
            page_report["metadata"] = page_metadata
        if block_reports:
            page_report["blocks"] = block_reports
        pages.append(page_report)

    if not pages:
        return None

    return {
        "version": DOCUMENT_METADATA_REPORT_VERSION,
        "summary": {
            "page_count": len(document.pages),
            "reported_page_count": len(pages),
            "processors": sorted(processors),
        },
        "pages": pages,
    }


def write_document_metadata_atomic(
    report: Mapping[str, Any],
    *,
    directory: str | os.PathLike[str] | None = None,
    artifact_id: str,
) -> str:
    """Atomically write a document metadata report as a temporary JSON artifact."""
    if not is_opaque_artifact_id(artifact_id):
        raise InvalidArtifactReferenceError(
            "Artifact ID must be a 32-character hex string."
        )

    payload = _json_safe_mapping(report)
    artifact_dir = Path(directory or tempfile.gettempdir()).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_dir / f"{DOCUMENT_METADATA_ARTIFACT_PREFIX}_{artifact_id}.json"
    tmp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=artifact_dir,
            prefix=f".{DOCUMENT_METADATA_ARTIFACT_PREFIX}_{artifact_id}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = tmp.name
            json.dump(payload, tmp, ensure_ascii=False, sort_keys=True)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)
        raise

    return str(target)


def _block_metadata_report(
    block_index: int,
    block: "DocumentBlock",
) -> dict[str, Any] | None:
    metadata = {
        key: _json_safe_value(value)
        for key in _BLOCK_METADATA_KEYS
        if (value := block.metadata.get(key)) is not None
    }
    has_reading_order = block.reading_order is not None

    if not metadata and not has_reading_order:
        return None

    report: dict[str, Any] = {
        "block_index": block_index,
        "bbox": _json_safe_value(block.bbox),
        "kind": block.kind,
    }
    if has_reading_order:
        report["reading_order"] = block.reading_order
    if metadata:
        report["metadata"] = metadata
    return report


def _processors_for_page_metadata(metadata: Mapping[str, Any]) -> set[str]:
    processors: set[str] = set()
    if "quality" in metadata:
        processors.add("quality_analysis")
    if "structure" in metadata:
        processors.add("structure_analysis")
    if "sections" in metadata:
        processors.add("section_analysis")
    return processors


def _processors_for_block_report(report: Mapping[str, Any]) -> set[str]:
    processors: set[str] = set()
    if "reading_order" in report:
        processors.add("reading_order")

    metadata = report.get("metadata")
    if isinstance(metadata, Mapping):
        if "structure" in metadata:
            processors.add("structure_analysis")
        if "section" in metadata:
            processors.add("section_analysis")
    return processors


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        _json_safe_key(key): _json_safe_value(item)
        for key, item in value.items()
        if item is not None
    }


def _json_safe_key(value: object) -> str:
    if not isinstance(value, str):
        raise InvalidArtifactPayloadError("Metadata report keys must be strings.")
    return value


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidArtifactPayloadError(
                "Metadata report numbers must be finite."
            )
        return value
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe_value(item) for item in value]
    raise InvalidArtifactPayloadError(
        f"Metadata report values must be JSON serializable, got {type(value).__name__}."
    )
