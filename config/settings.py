"""FinGuard AI - Central Settings and Design System Configuration.

Provides immutable project defaults, design tokens, schema definitions,
and directory path resolutions for enterprise financial anomaly detection.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional


@dataclass(frozen=True)
class DesignTokens:
    """FinGuard AI Enterprise Design System Tokens."""
    BG_COLOR: str = "#0B0D10"           # Deep charcoal background
    SIDEBAR_COLOR: str = "#101318"      # Black sidebar
    CARD_BG: str = "#15191F"            # Surface cards
    ELEVATED_CARD_BG: str = "#1B2028"   # Elevated cards
    PRIMARY_RED: str = "#E5484D"        # Primary crimson-red accent
    CRITICAL_RED: str = "#FF4D5A"       # Critical anomaly alert
    SUCCESS_GREEN: str = "#35C98A"      # Normal transaction green
    PRIMARY_TEXT: str = "#F4F6F8"       # Primary text
    SECONDARY_TEXT: str = "#9AA4B2"     # Soft grey secondary text
    BORDER_COLOR: str = "#292F39"       # Subtle card borders
    MUTED_ACCENT: str = "#232936"       # Subtle hover and card background
    ACCENT_BLUE: str = "#3B82F6"        # Informational accent
    WARNING_AMBER: str = "#F59E0B"      # Warning badge accent


@dataclass
class Settings:
    """Application configuration parameters and path management."""
    # Product Metadata
    APP_NAME: str = "FinGuard AI"
    APP_TAGLINE: str = "Enterprise Real-Time Financial Anomaly Detection"
    VERSION: str = "1.0.0-phase1"
    AUTHOR: str = "FinGuard AI Engineering"

    # Base Paths
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @property
    def DATA_DIR(self) -> Path:
        return self.BASE_DIR / "data"

    @property
    def DATA_RAW_DIR(self) -> Path:
        return self.DATA_DIR / "raw"

    @property
    def DATA_PROCESSED_DIR(self) -> Path:
        return self.DATA_DIR / "processed"

    @property
    def DATA_DEMO_DIR(self) -> Path:
        return self.DATA_DIR / "demo"

    @property
    def MODELS_DIR(self) -> Path:
        return self.BASE_DIR / "models"

    @property
    def LOGS_DIR(self) -> Path:
        return self.BASE_DIR / "logs"

    # Benchmark Dataset Specifications
    BENCHMARK_FILENAME: str = "creditcard.csv"

    @property
    def DEFAULT_BENCHMARK_PATH(self) -> Path:
        return self.DATA_RAW_DIR / self.BENCHMARK_FILENAME

    DEMO_FILENAME: str = "finguard_demo_transactions.csv"

    @property
    def DEFAULT_DEMO_PATH(self) -> Path:
        return self.DATA_DEMO_DIR / self.DEMO_FILENAME

    PREPROCESSOR_FILENAME: str = "preprocessor.joblib"

    @property
    def DEFAULT_PREPROCESSOR_PATH(self) -> Path:
        return self.MODELS_DIR / self.PREPROCESSOR_FILENAME

    # Persistent Case Management Database
    CASES_DB_FILENAME: str = "cases.db"

    @property
    def DEFAULT_CASES_DB_PATH(self) -> Path:
        return self.DATA_DIR / self.CASES_DB_FILENAME

    # Dataset Schema Definitions
    TIME_COL: str = "Time"
    AMOUNT_COL: str = "Amount"
    TARGET_COL: str = "Class"

    @property
    def PCA_FEATURE_COLS(self) -> List[str]:
        return [f"V{i}" for i in range(1, 29)]

    @property
    def REQUIRED_BENCHMARK_FEATURES(self) -> List[str]:
        return [self.TIME_COL] + self.PCA_FEATURE_COLS + [self.AMOUNT_COL]

    @property
    def ALL_BENCHMARK_COLUMNS(self) -> List[str]:
        return self.REQUIRED_BENCHMARK_FEATURES + [self.TARGET_COL]

    # Preprocessing Defaults
    SCALER_TYPE: str = "RobustScaler"
    DEFAULT_TEST_SIZE: float = 0.2
    DEFAULT_RANDOM_STATE: int = 42

    # Demo Simulation Defaults
    DEMO_SAMPLES_COUNT: int = 5000
    DEMO_ANOMALY_RATIO: float = 0.005  # Realistic 0.5% fraud rate

    # Investigation & Triage Thresholds
    RISK_CRITICAL_THRESHOLD: float = 0.85
    RISK_HIGH_THRESHOLD: float = 0.65
    RISK_MEDIUM_THRESHOLD: float = 0.40

    # UI Design System
    tokens: DesignTokens = field(default_factory=DesignTokens)


# Global settings singleton
settings = Settings()
