"""Tests for FinGuard AI configuration and settings."""

import pytest
from pathlib import Path
from config.settings import settings, DesignTokens


def test_settings_initialization():
    assert settings.APP_NAME == "FinGuard AI"
    assert "creditcard.csv" in str(settings.DEFAULT_BENCHMARK_PATH)
    assert len(settings.REQUIRED_BENCHMARK_FEATURES) == 30
    assert settings.TIME_COL == "Time"
    assert settings.AMOUNT_COL == "Amount"
    assert settings.TARGET_COL == "Class"


def test_design_tokens():
    tokens = DesignTokens()
    assert tokens.BG_COLOR == "#0B0D10"
    assert tokens.SIDEBAR_COLOR == "#101318"
    assert tokens.CARD_BG == "#15191F"
    assert tokens.PRIMARY_RED == "#E5484D"
    assert tokens.SUCCESS_GREEN == "#35C98A"


def test_paths_exist_or_resolvable():
    assert isinstance(settings.BASE_DIR, Path)
    assert isinstance(settings.DATA_DIR, Path)
    assert isinstance(settings.MODELS_DIR, Path)