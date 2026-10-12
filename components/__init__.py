"""FinGuard AI UI Design System and Presentation Components."""

from components.theme import configure_page, inject_theme, inject_custom_css
from components.cards import render_header, render_metric_card, render_banner
from components.charts import (
    create_transaction_activity_chart,
    create_amount_distribution_chart,
    create_hourly_volume_chart,
    create_pca_scatter_chart,
    create_anomaly_score_distribution_chart,
    create_confusion_matrix_chart,
    create_threshold_tradeoff_chart,
    create_precision_recall_curve_chart,
    create_transaction_feature_analysis_chart
)
from components.setup_card import render_setup_card
from components.tables import render_transaction_table

__all__ = [
    "configure_page",
    "inject_theme",
    "inject_custom_css",
    "render_header",
    "render_metric_card",
    "render_banner",
    "create_transaction_activity_chart",
    "create_amount_distribution_chart",
    "create_hourly_volume_chart",
    "create_pca_scatter_chart",
    "create_anomaly_score_distribution_chart",
    "create_confusion_matrix_chart",
    "create_threshold_tradeoff_chart",
    "create_precision_recall_curve_chart",
    "create_transaction_feature_analysis_chart",
    "render_setup_card",
    "render_transaction_table"
]
