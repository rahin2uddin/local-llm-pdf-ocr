from __future__ import annotations

import json
import os
import secrets
import tempfile
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ARTIFACT_TTL_SECONDS = 60 * 60
DEFAULT_MAX_ARTIFACT_ENTRIES = 256

_ARTIFACT_ID_BYTES = 16
_TOKEN_BYTES = 32
_OPAQUE_ID_LENGTH = _ARTIFACT_ID_BYTES * 2
_TOKEN_MIN_LENGTH = 32
_TOKEN_MAX_LENGTH = 256
_TOKEN_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
)
_ID_CHARS = frozenset("0123456789abcdef")

PageText = Mapping[int | str, Sequence[str]]
Clock = Callable[[], float]


class ArtifactStoreError(Exception):
    """Base class for deterministic artifact-store failures."""


class InvalidArtifactReferenceError(ArtifactStoreError, ValueError):
    """Raised when an artifact ID, token, or path fails boundary validation."""


class ArtifactAccessDeniedError(ArtifactStoreError):
    """Raised when an otherwise valid artifact reference has the wrong token."""


class ArtifactNotFoundError(ArtifactStoreError, KeyError):
    """Raised when a valid artifact reference is absent or has expired."""


class InvalidArtifactPayloadError(ArtifactStoreError, ValueError):
    """Raised when page text cannot be serialized as a stable JSON artifact."""


@dataclass(frozen=True, slots=True)
class TextArtifactHandle:
    artifact_id: str
    token: str
    path: str
    expires_at: float


@dataclass(frozen=True, slots=True)
class _TextArtifactEntry:
    token: str
    path: Path
    created_at: float
    expires_at: float


class TextArtifactStore:
    """Token-bound temporary JSON text artifacts with bounded retention."""

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_ARTIFACT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ARTIFACT_ENTRIES,
        clock: Clock = time.time,
        artifact_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive.")

        self._ttl_seconds = float(ttl_seconds)
        self._max_entries = max_entries
        self._clock = clock
        self._artifact_dir = Path(artifact_dir or tempfile.gettempdir()).resolve()
        self._entries: OrderedDict[str, _TextArtifactEntry] = OrderedDict()

    @property
    def artifact_dir(self) -> Path:
        return self._artifact_dir

    def issue_id(self) -> str:
        return secrets.token_hex(_ARTIFACT_ID_BYTES)

    def issue_token(self) -> str:
        return secrets.token_urlsafe(_TOKEN_BYTES)

    def create(self, page_text: PageText) -> TextArtifactHandle:
        """Write page text to a temporary JSON file and register its access token."""

        artifact_id = self.issue_id()
        token = self.issue_token()
        path = write_page_text_atomic(
            page_text,
            directory=self._artifact_dir,
            artifact_id=artifact_id,
        )
        return self.put(artifact_id=artifact_id, token=token, path=path)

    def put(
        self,
        *,
        artifact_id: str,
        token: str,
        path: str | os.PathLike[str],
    ) -> TextArtifactHandle:
        self.cleanup_expired()
        _validate_artifact_id(artifact_id)
        _validate_token(token)

        artifact_path = self._resolve_artifact_path(path)
        now = self._clock()
        expires_at = now + self._ttl_seconds

        previous = self._entries.pop(artifact_id, None)
        if previous is not None and previous.path != artifact_path:
            _delete_file(previous.path)

        self._entries[artifact_id] = _TextArtifactEntry(
            token=token,
            path=artifact_path,
            created_at=now,
            expires_at=expires_at,
        )
        self._evict_overflow()
        return TextArtifactHandle(
            artifact_id=artifact_id,
            token=token,
            path=str(artifact_path),
            expires_at=expires_at,
        )

    def get(self, artifact_id: str, token: str) -> str:
        entry = self._require_entry(artifact_id, token)
        return str(entry.path)

    def pop(self, artifact_id: str, token: str) -> str | None:
        """Remove a token-bound artifact entry without deleting its backing file."""

        _validate_artifact_id(artifact_id)
        _validate_token(token)
        self.cleanup_expired()

        entry = self._entries.get(artifact_id)
        if entry is None:
            return None
        if not secrets.compare_digest(entry.token, token):
            raise ArtifactAccessDeniedError("Artifact token does not match.")

        self._entries.pop(artifact_id, None)
        return str(entry.path)

    def delete(self, artifact_id: str, token: str) -> bool:
        """Remove a token-bound artifact entry and delete its backing file."""

        path = self.pop(artifact_id, token)
        if path is None:
            return False
        _delete_file(Path(path))
        return True

    def cleanup_expired(self) -> list[str]:
        now = self._clock()
        expired_ids = [
            artifact_id
            for artifact_id, entry in self._entries.items()
            if entry.expires_at <= now
        ]
        return [
            str(path) for path in self._remove_entries(expired_ids, delete_files=True)
        ]

    def clear(self) -> list[str]:
        artifact_ids = list(self._entries)
        return [
            str(path) for path in self._remove_entries(artifact_ids, delete_files=True)
        ]

    def __len__(self) -> int:
        self.cleanup_expired()
        return len(self._entries)

    def _require_entry(self, artifact_id: str, token: str) -> _TextArtifactEntry:
        _validate_artifact_id(artifact_id)
        _validate_token(token)
        self.cleanup_expired()

        entry = self._entries.get(artifact_id)
        if entry is None:
            raise ArtifactNotFoundError("Artifact was not found.")
        if not secrets.compare_digest(entry.token, token):
            raise ArtifactAccessDeniedError("Artifact token does not match.")
        return entry

    def _resolve_artifact_path(self, path: str | os.PathLike[str]) -> Path:
        artifact_path = Path(path).resolve()
        try:
            artifact_path.relative_to(self._artifact_dir)
        except ValueError as exc:
            raise InvalidArtifactReferenceError(
                "Artifact path is outside the configured artifact directory."
            ) from exc
        return artifact_path

    def _evict_overflow(self) -> None:
        while len(self._entries) > self._max_entries:
            artifact_id, _entry = next(iter(self._entries.items()))
            self._remove_entries([artifact_id], delete_files=True)

    def _remove_entries(
        self,
        artifact_ids: Sequence[str],
        *,
        delete_files: bool,
    ) -> list[Path]:
        removed_paths: list[Path] = []
        for artifact_id in artifact_ids:
            entry = self._entries.pop(artifact_id, None)
            if entry is None:
                continue
            removed_paths.append(entry.path)
            if delete_files:
                _delete_file(entry.path)
        return removed_paths


def write_page_text_atomic(
    page_text: PageText,
    *,
    directory: str | os.PathLike[str] | None = None,
    artifact_id: str | None = None,
) -> str:
    """Write page text as JSON through a temporary sibling and atomic replace."""

    if artifact_id is None:
        artifact_id = secrets.token_hex(_ARTIFACT_ID_BYTES)
    _validate_artifact_id(artifact_id)

    artifact_dir = Path(directory or tempfile.gettempdir()).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_dir / f"text_{artifact_id}.json"
    payload = _normalize_page_text(page_text)
    tmp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=artifact_dir,
            prefix=f".text_{artifact_id}.",
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
            _delete_file(Path(tmp_path))
        raise

    return str(target)


def is_opaque_artifact_id(value: str) -> bool:
    return len(value) == _OPAQUE_ID_LENGTH and all(ch in _ID_CHARS for ch in value)


def _validate_artifact_id(value: str) -> None:
    if not isinstance(value, str) or not is_opaque_artifact_id(value):
        raise InvalidArtifactReferenceError(
            "Artifact ID must be a 32-character hex string."
        )


def _validate_token(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) < _TOKEN_MIN_LENGTH
        or len(value) > _TOKEN_MAX_LENGTH
        or any(ch not in _TOKEN_CHARS for ch in value)
    ):
        raise InvalidArtifactReferenceError("Artifact token is not well formed.")


def _normalize_page_text(page_text: PageText) -> dict[str, list[str]]:
    if not isinstance(page_text, Mapping):
        raise InvalidArtifactPayloadError("Page text must be a mapping.")

    normalized: dict[str, list[str]] = {}
    for raw_page, raw_lines in page_text.items():
        page = _normalize_page_key(raw_page)
        if isinstance(raw_lines, str) or not isinstance(raw_lines, Sequence):
            raise InvalidArtifactPayloadError(
                "Each page value must be a sequence of strings."
            )
        lines: list[str] = []
        for line in raw_lines:
            if not isinstance(line, str):
                raise InvalidArtifactPayloadError("Page text lines must be strings.")
            lines.append(line)
        normalized[page] = lines
    return normalized


def _normalize_page_key(value: Any) -> str:
    if isinstance(value, bool):
        raise InvalidArtifactPayloadError("Page keys must be non-negative integers.")
    if isinstance(value, int):
        if value < 0:
            raise InvalidArtifactPayloadError(
                "Page keys must be non-negative integers."
            )
        return str(value)
    if isinstance(value, str) and value.isdecimal():
        return value
    raise InvalidArtifactPayloadError("Page keys must be non-negative integers.")


def _delete_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise ArtifactStoreError(f"Could not remove artifact file: {path}") from exc
