from __future__ import annotations

import pytest
from pydantic import ValidationError

from local_deepl.api.schemas import ConfigUpdate, ProcessSettings
from local_deepl.core.processors import (
    QualityAnalysisProcessor,
    ReadingOrderProcessor,
    StructureAnalysisProcessor,
    build_document_processors,
)


def _process_settings(**overrides):
    values = {
        "api_base": "http://localhost:1234/v1",
        "api_key": "local",
        "model": "local-model",
        "pipeline_mode": "hybrid",
        "dpi": 200,
        "concurrency": 1,
        "dense_mode": "auto",
        "dense_threshold": 60,
        "pages": None,
        "refine": True,
        "max_image_dim": 1024,
        "self_correction": False,
        "binarize": False,
        "dual_engine": False,
        "spellcheck": "none",
        "cross_page": False,
    }
    values.update(overrides)
    return ProcessSettings.model_validate(values)


def test_process_settings_accepts_comma_separated_document_processors():
    settings = _process_settings(
        document_processors="reading_order, quality_analysis, structure_analysis"
    )

    assert [name.value for name in settings.document_processors] == [
        "reading_order",
        "quality_analysis",
        "structure_analysis",
    ]


def test_config_update_accepts_document_processor_list():
    update = ConfigUpdate.model_validate({"document_processors": ["quality_analysis"]})

    assert [name.value for name in update.document_processors or []] == [
        "quality_analysis"
    ]


def test_process_settings_rejects_unknown_document_processor():
    with pytest.raises(ValidationError):
        _process_settings(document_processors="cloud_magic")


def test_build_document_processors_maps_allowed_names():
    processors = build_document_processors(
        ["reading_order", "quality_analysis", "structure_analysis"]
    )

    assert isinstance(processors[0], ReadingOrderProcessor)
    assert isinstance(processors[1], QualityAnalysisProcessor)
    assert isinstance(processors[2], StructureAnalysisProcessor)
