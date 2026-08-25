"""Basic sanity tests for project configuration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))
from config import COMPUTE_POOL, DATABASE, PIPELINE_CONFIG, SCHEMA, WAREHOUSE


def test_config_values_are_strings():
    assert isinstance(DATABASE, str)
    assert isinstance(SCHEMA, str)
    assert isinstance(WAREHOUSE, str)
    assert isinstance(COMPUTE_POOL, str)


def test_pipeline_config_all_string_values():
    for key, value in PIPELINE_CONFIG.items():
        assert isinstance(value, str), f"Config key '{key}' has non-string value: {type(value)}"


def test_pipeline_config_has_required_keys():
    required = ["min_auc_roc", "min_precision", "min_recall", "model_name"]
    for key in required:
        assert key in PIPELINE_CONFIG, f"Missing required config key: {key}"


def test_database_prefix():
    assert DATABASE.startswith("SNOW_MLOPS"), f"Expected SNOW_MLOPS prefix, got: {DATABASE}"
