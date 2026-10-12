"""Design system injector and layout helpers for Streamlit."""

from pathlib import Path
import streamlit as st
from config.settings import settings


def inject_theme() -> None:
    """Injects custom enterprise dark theme CSS and typography into the Streamlit app."""
    css_file = settings.BASE_DIR / "assets" / "style.css"
    if css_file.is_file():
        css_content = css_file.read_text(encoding="utf-8")
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <style>
            .stApp {{ background-color: {settings.tokens.BG_COLOR}; color: {settings.tokens.PRIMARY_TEXT}; }}
            </style>
            """,
            unsafe_allow_html=True
        )


def configure_page(page_title: str = "FinGuard AI", layout: str = "wide") -> None:
    """Sets standard page configuration and injects CSS design tokens."""
    try:
        st.set_page_config(
            page_title=f"{page_title} | FinGuard AI",
            layout=layout,
            initial_sidebar_state="expanded"
        )
    except Exception:
        # Ignore if set_page_config was already called earlier in this run
        pass
    inject_theme()


# Alias for backward compatibility
inject_custom_css = inject_theme
