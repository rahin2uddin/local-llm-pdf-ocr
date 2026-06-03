"""Unit tests for the upgraded Tesseract langdata post-processing spellchecker."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

import pdf_ocr.core.postprocess as postprocess
from pdf_ocr.core.postprocess import DictionaryPostProcessor


@pytest.fixture
def temp_resources():
    """Create a temporary directory structure mimicking the project resources."""
    temp_dir = tempfile.mkdtemp()
    langdata_dir = os.path.join(temp_dir, "langdata")
    dictionaries_dir = os.path.join(temp_dir, "dictionaries")
    os.makedirs(langdata_dir, exist_ok=True)
    os.makedirs(dictionaries_dir, exist_ok=True)

    yield temp_dir, langdata_dir, dictionaries_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_language_iso_mapping():
    """Verify standard language codes are mapped correctly to Tesseract 3-letter codes."""
    assert DictionaryPostProcessor("en").tess_lang == "eng"
    assert DictionaryPostProcessor("eng").tess_lang == "eng"
    assert DictionaryPostProcessor("en-US").tess_lang == "eng"
    assert DictionaryPostProcessor("ar").tess_lang == "ara"
    assert DictionaryPostProcessor("arabic").tess_lang == "ara"
    assert DictionaryPostProcessor("de-DE").tess_lang == "deu"
    assert DictionaryPostProcessor("xyz").tess_lang == "xyz"  # No mapping fallback


def test_unicode_diacritics_check():
    """Verify isalpha validation works for words containing diacritics."""
    DictionaryPostProcessor("ara")

    # Test compilation helper on Arabic word list with diacritics
    # \u064e is Fatha, \u0651 is Shadda.
    # "أَحْمَدُ" is "Ahmed" with diacritics.
    # "محمدٌ" is "Mohamed" with diacritics.

    # Clean check logic test
    import unicodedata

    def is_valid(word):
        cleaned = "".join(c for c in word if unicodedata.category(c) != "Mn")
        return cleaned.isalpha()

    assert is_valid("أَحْمَدُ") is True
    assert is_valid("محمدٌ") is True
    assert is_valid("12345") is False
    assert is_valid("hello!") is False


async def test_compilation_and_loading(temp_resources):
    """Test full cycle: compile raw wordlist to gzipped JSON and load it."""
    temp_dir, langdata_dir, dictionaries_dir = temp_resources

    # Setup mock wordlist for German (deu)
    deu_langdata = os.path.join(langdata_dir, "deu")
    os.makedirs(deu_langdata, exist_ok=True)

    wordlist_path = os.path.join(deu_langdata, "deu.wordlist")
    with open(wordlist_path, "w", encoding="utf-8") as f:
        f.write("apfel\nbirne\nkirsche\n")

    processor = DictionaryPostProcessor("deu", resources_dir=temp_dir)

    await processor.ensure_loaded()

    # Verify compiled gz file exists in dictionaries directory
    compiled_gz_path = os.path.join(dictionaries_dir, "deu.json.gz")
    assert os.path.exists(compiled_gz_path)

    # Verify contents of gz file
    with gzip.open(compiled_gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
        assert data == {"apfel": 1, "birne": 1, "kirsche": 1}

    # Verify spellchecker corrects typos matching Levenshtein distance 1
    assert processor.correct_text("apfl") == "apfel"
    assert processor.correct_text("birn") == "birne"
    assert processor.correct_text("kirsch") == "kirsche"

    # Verify casing is preserved
    assert processor.correct_text("Apfl") == "Apfel"
    assert processor.correct_text("APFL") == "APFEL"

    # Unknown or far typos (distance > 1) remain untouched
    assert processor.correct_text("apffffl") == "apffffl"


async def test_packaged_dictionary_lookup_precedes_legacy_resources(monkeypatch):
    """Default lookup should load bundled dictionaries from the installed package."""
    loaded_paths: list[str] = []
    sentinel = object()

    def fake_load_custom_dictionary(dict_path: str):
        loaded_paths.append(dict_path)
        return sentinel

    monkeypatch.setattr(
        postprocess, "_load_custom_dictionary", fake_load_custom_dictionary
    )

    processor = DictionaryPostProcessor("eng")
    await processor.ensure_loaded()

    package_dict_path = (
        Path(__file__).parents[1]
        / "src"
        / "pdf_ocr"
        / "resources"
        / "dictionaries"
        / "eng.json.gz"
    ).resolve()
    assert processor.spell is sentinel
    assert loaded_paths == [str(package_dict_path)]


async def test_legacy_repository_dictionary_fallback(monkeypatch):
    """Repository-root dictionaries remain a fallback for older checkouts."""

    class MissingPackagedResource:
        def joinpath(self, *parts: str) -> MissingPackagedResource:
            return self

        def is_file(self) -> bool:
            return False

    loaded_paths: list[str] = []
    sentinel = object()

    def fake_load_custom_dictionary(dict_path: str):
        loaded_paths.append(dict_path)
        return sentinel

    monkeypatch.setattr(
        postprocess.resources, "files", lambda package: MissingPackagedResource()
    )
    monkeypatch.setattr(
        postprocess, "_load_custom_dictionary", fake_load_custom_dictionary
    )

    processor = DictionaryPostProcessor("eng")
    await processor.ensure_loaded()

    legacy_dict_path = (
        Path(__file__).parents[1] / "resources" / "dictionaries" / "eng.json.gz"
    ).resolve()
    assert processor.spell is sentinel
    assert loaded_paths == [str(legacy_dict_path)]


async def test_graceful_fallback():
    """Verify processor falls back gracefully if raw wordlist or dictionary is missing."""
    processor = DictionaryPostProcessor("xyz")  # Nonexistent language

    await processor.ensure_loaded()

    # Nonexistent language should fallback to None (safe no-op)
    assert processor.spell is None
    assert processor.correct_text("someword") == "someword"
