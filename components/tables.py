"""Table presentation components for financial transaction feeds."""

from typing import Optional
import pandas as pd
import streamlit as st


def render_transaction_table(
    df: pd.DataFrame,
    max_rows: int = 50,
    hide_index: bool = True
) -> None:
    """Renders a styled financial transaction table."""
    if len(df) == 0:
        st.info("No transaction records to display.")
        return

    display_df = df.head(max_rows).copy()

    # Format Currency if Amount column is present
    if "Amount" in display_df.columns:
        display_df["Amount"] = display_df["Amount"].apply(
            lambda v: f"${v:,.2f}" if isinstance(v, (int, float)) else str(v)
        )

    # Format Time if present
    if "Time" in display_df.columns:
        display_df["Time"] = display_df["Time"].apply(
            lambda v: f"{int(v):,}s" if isinstance(v, (int, float)) else str(v)
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=hide_index
    )
