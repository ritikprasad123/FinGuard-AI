"""Formatting and arithmetic helpers for FinGuard AI financial analytics."""

from typing import Union


def format_currency(amount: Union[int, float], symbol: str = "$") -> str:
    """Formats a number as standard financial currency."""
    if amount is None:
        return "N/A"
    return f"{symbol}{amount:,.2f}"


def format_percentage(value: Union[int, float], decimals: int = 2) -> str:
    """Formats a ratio or percentage cleanly."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}%"


def format_integer(val: Union[int, float]) -> str:
    """Formats an integer with thousands commas."""
    if val is None:
        return "N/A"
    return f"{int(val):,}"


def format_seconds_to_time(seconds: Union[int, float]) -> str:
    """Converts elapsed seconds to human readable elapsed time string."""
    if seconds is None:
        return "N/A"
    s = int(seconds)
    hours = s // 3600
    minutes = (s % 3600) // 60
    secs = s % 60
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
