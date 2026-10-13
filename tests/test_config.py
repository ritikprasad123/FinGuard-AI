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
    assert "decision-support" in settings.DISCLAIMER_NOTICE.lower()


def test_design_tokens():
    tokens = DesignTokens()
    assert tokens.BG_COLOR in ["#0B0D10", "#0B0F19"]
    assert tokens.SIDEBAR_COLOR in ["#101318", "#0D1322"]
    assert tokens.CARD_BG in ["#15191F", "#131B2E"]
    assert tokens.PRIMARY_RED in ["#E5484D", "#E11D48"]
    assert tokens.SUCCESS_GREEN in ["#35C98A", "#10B981"]
    assert hasattr(tokens, "ACCENT_BLUE")


def test_paths_exist_or_resolvable():
    assert isinstance(settings.BASE_DIR, Path)
    assert isinstance(settings.DATA_DIR, Path)
    assert isinstance(settings.MODELS_DIR, Path)
