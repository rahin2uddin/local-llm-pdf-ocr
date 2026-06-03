from __future__ import annotations

import json
import sys
from importlib import util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_PATH = ROOT / "src" / "pdf_ocr" / "api" / "services" / "artifacts.py"
SPEC = util.spec_from_file_location("artifact_service_under_test", ARTIFACTS_PATH)
assert SPEC is not None
assert SPEC.loader is not None
artifacts = util.module_from_spec(SPEC)
sys.modules[SPEC.name] = artifacts
SPEC.loader.exec_module(artifacts)

ArtifactAccessDeniedError = artifacts.ArtifactAccessDeniedError
ArtifactNotFoundError = artifacts.ArtifactNotFoundError
InvalidArtifactReferenceError = artifacts.InvalidArtifactReferenceError
TextArtifactStore = artifacts.TextArtifactStore
write_page_text_atomic = artifacts.write_page_text_atomic


class ManualClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_artifacts_bind_ids_to_separate_access_tokens(tmp_path: Path):
    store = TextArtifactStore(artifact_dir=tmp_path)

    handle = store.create({0: ["first page"], 1: ["second page"]})

    assert handle.artifact_id != handle.token
    assert store.get(handle.artifact_id, handle.token) == handle.path
    assert json.loads(Path(handle.path).read_text(encoding="utf-8")) == {
        "0": ["first page"],
        "1": ["second page"],
    }


def test_wrong_token_denies_artifact_access(tmp_path: Path):
    store = TextArtifactStore(artifact_dir=tmp_path)
    handle = store.create({0: ["secret"]})
    wrong_token = store.issue_token()

    with pytest.raises(ArtifactAccessDeniedError):
        store.get(handle.artifact_id, wrong_token)

    assert Path(handle.path).exists()


def test_expiry_cleanup_deletes_backing_files(tmp_path: Path):
    clock = ManualClock()
    store = TextArtifactStore(ttl_seconds=5, clock=clock, artifact_dir=tmp_path)
    handle = store.create({0: ["expires"]})
    path = Path(handle.path)

    clock.advance(6)

    assert store.cleanup_expired() == [str(path)]
    assert not path.exists()
    assert store.cleanup_expired() == []
    with pytest.raises(ArtifactNotFoundError):
        store.get(handle.artifact_id, handle.token)


def test_max_entry_eviction_deletes_oldest_backing_file(tmp_path: Path):
    store = TextArtifactStore(max_entries=1, artifact_dir=tmp_path)
    first = store.create({0: ["old"]})
    second = store.create({0: ["new"]})

    assert not Path(first.path).exists()
    assert Path(second.path).exists()
    with pytest.raises(ArtifactNotFoundError):
        store.get(first.artifact_id, first.token)
    assert store.get(second.artifact_id, second.token) == second.path


def test_invalid_artifact_ids_are_rejected(tmp_path: Path):
    store = TextArtifactStore(artifact_dir=tmp_path)
    token = store.issue_token()

    for artifact_id in ("client-id", "A" * 32, "0" * 31, "../" + "0" * 32):
        with pytest.raises(InvalidArtifactReferenceError):
            store.get(artifact_id, token)


def test_delete_is_idempotent_and_removes_backing_file(tmp_path: Path):
    store = TextArtifactStore(artifact_dir=tmp_path)
    handle = store.create({0: ["delete me"]})
    path = Path(handle.path)

    assert store.delete(handle.artifact_id, handle.token) is True
    assert not path.exists()
    assert store.delete(handle.artifact_id, handle.token) is False


def test_pop_is_token_bound_and_idempotent_without_deleting_file(tmp_path: Path):
    store = TextArtifactStore(artifact_dir=tmp_path)
    handle = store.create({0: ["download once"]})
    path = Path(handle.path)

    assert store.pop(handle.artifact_id, handle.token) == str(path)
    assert path.exists()
    assert store.pop(handle.artifact_id, handle.token) is None


def test_page_text_helper_writes_json_atomically_enough_for_callers(tmp_path: Path):
    store = TextArtifactStore(artifact_dir=tmp_path)
    artifact_id = store.issue_id()

    path = write_page_text_atomic(
        {0: ["alpha"]}, directory=tmp_path, artifact_id=artifact_id
    )
    write_page_text_atomic({0: ["beta"]}, directory=tmp_path, artifact_id=artifact_id)

    assert Path(path).name == f"text_{artifact_id}.json"
    assert json.loads(Path(path).read_text(encoding="utf-8")) == {"0": ["beta"]}
    assert list(tmp_path.glob("*.tmp")) == []
