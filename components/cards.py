"""Enterprise KPI cards, banners, and status indicator components."""

from typing import Optional
import streamlit as st
from config.settings import settings


def render_header(title: str, subtitle: str, badge_text: Optional[str] = None, badge_type: str = "neutral") -> None:
    """Renders sleek top-level section header with optional badge."""
    badge_html = ""
    if badge_text:
        badge_class = f"fg-badge-{badge_type}"
        badge_html = f"<span class='fg-badge {badge_class}' style='margin-left: 0.75rem; vertical-align: middle;'>{badge_text}</span>"

    html = f"""
    <div style="margin-bottom: 1.5rem; border-bottom: 1px solid var(--border-color); padding-bottom: 1rem;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h2 style="margin: 0; font-size: 1.65rem; font-weight: 700;">
                    {title}{badge_html}
                </h2>
                <div style="color: var(--secondary-text); font-size: 0.875rem; margin-top: 0.25rem;">
                    {subtitle}
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    status_color: Optional[str] = None,
    badge_text: Optional[str] = None
) -> str:
    """Returns HTML for an enterprise KPI metric card."""
    border_accent = f"border-top: 3px solid {status_color};" if status_color else ""
    badge_html = f"<span class='fg-badge fg-badge-neutral' style='float: right;'>{badge_text}</span>" if badge_text else ""

    return f"""
    <div class="fg-metric-box" style="{border_accent}">
        {badge_html}
        <div class="fg-metric-title">{title}</div>
        <div class="fg-metric-value">{value}</div>
        <div class="fg-metric-subtitle">{subtitle}</div>
    </div>
    """


def render_banner(
    title: str,
    message: str,
    level: str = "info",
    badge: Optional[str] = None
) -> None:
    """Renders enterprise informational or warning banner."""
    css_class = f"fg-banner-{level}"
    badge_html = f"<span class='fg-badge fg-badge-demo' style='margin-right: 0.5rem;'>{badge}</span>" if badge else ""

    html = f"""
    <div class="{css_class}">
        <div style="font-weight: 600; font-size: 0.9rem; margin-bottom: 0.25rem; display: flex; align-items: center;">
            {badge_html}<span>{title}</span>
        </div>
        <div style="font-size: 0.825rem; color: var(--primary-text); opacity: 0.9;">
            {message}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
