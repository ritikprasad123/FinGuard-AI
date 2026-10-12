"""Central logging infrastructure for FinGuard AI."""

import logging
import sys
from pathlib import Path
from config.settings import settings


def get_logger(name: str = "finguard") -> logging.Logger:
    """Returns a configured logger instance with console and file handlers.

    Args:
        name: Logger name namespace.

    Returns:
        Configured logging.Logger.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        try:
            settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            log_file = settings.LOGS_DIR / "finguard.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            # Fallback gracefully if directory is not writable
            pass

    return logger
