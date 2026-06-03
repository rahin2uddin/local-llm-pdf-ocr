"""Focused tests for the optional async translation dependency boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

from pdf_ocr.core.translation_config import TranslationSettings


def test_translation_base_imports_do_not_require_async_extras():
    script = """
import importlib.abc
import sys

blocked = {"celery", "redis", "langgraph", "chromadb", "sentence_transformers"}

class BlockAsyncExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in blocked:
            raise ImportError(f"blocked optional dependency: {fullname}")
        return None

sys.meta_path.insert(0, BlockAsyncExtras())

from pdf_ocr.api.tasks import process_translation_task
from pdf_ocr.core.translation import chunk_text, evaluate_node

assert chunk_text("hello") == ["hello"]
assert evaluate_node({"source_chunk": ".", "translated_chunk": "", "attempts": 1})["evaluation_score"] == 1.0
assert process_translation_task.__name__ == "process_translation_task"
"""
    env = os.environ.copy()
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_translate_node_uses_injected_settings(monkeypatch):
    import pdf_ocr.core.translation as translation

    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Bonjour"))]
        )

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(completion=fake_completion),
    )

    state = {
        "source_chunk": "Hello",
        "target_language": "French",
        "rag_context": [],
        "translated_chunk": "",
        "evaluation_score": 1.0,
        "feedback": "",
        "attempts": 0,
        "settings": TranslationSettings(
            api_base="https://example.test/v1",
            api_key="test-key",
            model="openai/test-model",
        ),
    }

    result = translation.translate_node(state)

    assert result["translated_chunk"] == "Bonjour"
    assert captured["api_base"] == "https://example.test/v1"
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "openai/test-model"
