"""Setup screen and benchmark dataset acquisition instructions."""

import streamlit as st
from config.settings import settings
from src.data.loader import get_dataset_provenance


def render_setup_card() -> None:
    """Renders professional setup instructions when benchmark dataset is absent."""
    prov = get_dataset_provenance()

    col1, col2 = st.columns([1.2, 1.0], gap="large")

    with col1:
        st.markdown(
            f"""
            <div class="fg-card">
                <h4 style="margin-top: 0; color: #FFFFFF;">Data Ingestion Options</h4>
                <p style="color: var(--secondary-text); font-size: 0.875rem;">
                    FinGuard AI requires transaction data to activate real-time surveillance.
                    Select an ingestion method below:
                </p>
                <div style="margin-top: 1rem; padding: 0.75rem; background: var(--elevated-card-bg); border-radius: 6px; border: 1px solid var(--border-color);">
                    <div style="font-weight: 600; font-size: 0.85rem; color: #FFFFFF;">Option A: Synthetic Demo Data</div>
                    <div style="font-size: 0.8rem; color: var(--secondary-text); margin-top: 0.25rem;">
                        Generate 5,000 synthetic transactions matching real financial distributions. Strictly labelled as DEMO DATA.
                    </div>
                </div>
                <div style="margin-top: 0.75rem; padding: 0.75rem; background: var(--elevated-card-bg); border-radius: 6px; border: 1px solid var(--border-color);">
                    <div style="font-weight: 600; font-size: 0.85rem; color: #FFFFFF;">Option B: Import Verified Benchmark Dataset</div>
                    <div style="font-size: 0.8rem; color: var(--secondary-text); margin-top: 0.25rem;">
                        Upload the official <code>creditcard.csv</code> file or place it in <code>data/raw/creditcard.csv</code>.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="fg-card">
                <h4 style="margin-top: 0; color: #FFFFFF;">Benchmark Dataset Provenance</h4>
                <div style="font-size: 0.8rem; color: var(--secondary-text); line-height: 1.6;">
                    <div><strong style="color: var(--primary-text);">Dataset Name:</strong> {prov['dataset_name']}</div>
                    <div><strong style="color: var(--primary-text);">Institution:</strong> {prov['institution']}</div>
                    <div><strong style="color: var(--primary-text);">Authors:</strong> {prov['creators']}</div>
                    <div><strong style="color: var(--primary-text);">License:</strong> <code>{prov['license']}</code></div>
                    <div><strong style="color: var(--primary-text);">Expected Volume:</strong> {prov['expected_shape']}</div>
                    <div style="margin-top: 0.5rem; border-top: 1px solid var(--border-color); padding-top: 0.5rem;">
                        <strong style="color: var(--primary-text);">Expected File Path:</strong><br>
                        <code>{settings.DEFAULT_BENCHMARK_PATH}</code>
                    </div>
                    <div style="margin-top: 0.5rem;">
                        <strong style="color: var(--primary-text);">Legal Source URL:</strong><br>
                        <a href="{prov['kaggle_ref']}" target="_blank" style="color: var(--accent-blue);">{prov['kaggle_ref']}</a>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
