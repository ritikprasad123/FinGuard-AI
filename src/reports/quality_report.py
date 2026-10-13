"""Quality summary and executive audit report generator for FinGuard AI."""

from typing import Dict, Any, List, Optional
from src.data.validator import DataQualityReport


def generate_executive_quality_summary(report: DataQualityReport) -> Dict[str, Any]:
    """Generates structured KPI summary from DataQualityReport for executive dashboards."""
    return {
        "status": "PASSED" if report.is_valid else "FAILED",
        "total_transactions": f"{report.total_rows:,}",
        "features_count": report.total_columns,
        "completeness_pct": f"{100.0 - report.missing_values_pct:.2f}%",
        "duplicate_records": f"{report.duplicate_rows_count:,} ({report.duplicate_rows_pct:.2f}%)",
        "ground_truth_status": "Verified (Labelled)" if report.has_target_column else "Unlabelled (Unsupervised)",
        "fraud_rate": f"{report.fraud_rate_pct:.4f}%" if report.fraud_rate_pct is not None else "N/A",
        "imbalance_ratio": f"{report.imbalance_ratio:.1f} : 1" if report.imbalance_ratio is not None else "N/A",
        "financial_range": f"${report.min_amount:,.2f} to ${report.max_amount:,.2f}" if report.min_amount is not None else "N/A",
        "error_count": len(report.errors),
        "warning_count": len(report.warnings)
    }


def format_quality_markdown_report(
    report: DataQualityReport,
    dataset_source: Optional[str] = None,
    **kwargs
) -> str:
    """Formats full data quality audit into a structured Markdown document."""
    status_badge = "[PASSED]" if report.is_valid else "[FAILED]"
    lines = [
        f"### Data Quality Audit Report — {status_badge}",
        f"- **Total Transactions Ingested:** {report.total_rows:,}",
        f"- **Total Features Detected:** {report.total_columns}",
        f"- **Missing Value Cells:** {report.missing_values_count:,} ({report.missing_values_pct:.2f}%)",
        f"- **Duplicate Rows:** {report.duplicate_rows_count:,} ({report.duplicate_rows_pct:.2f}%)",
        f"- **Financial Negative Amounts:** {report.negative_amount_count}",
    ]
    if dataset_source:
        lines.append(f"- **Data Ingestion Source:** `{dataset_source}`")
    lines.append("")

    if report.has_target_column:
        lines.append("#### Ground-Truth Class Distribution")
        for cls, count in report.class_distribution.items():
            label = "Fraudulent (Class 1)" if str(cls) == "1" else "Normal (Class 0)"
            pct = (count / report.total_rows) * 100 if report.total_rows > 0 else 0
            lines.append(f"- **{label}:** {count:,} ({pct:.3f}%)")
        if report.imbalance_ratio:
            lines.append(f"- **Imbalance Ratio:** {report.imbalance_ratio:,.1f} : 1")
        lines.append("")

    if report.errors:
        lines.append("#### Validation Errors")
        for err in report.errors:
            lines.append(f"- [ERROR] {err}")
        lines.append("")

    if report.warnings:
        lines.append("#### Quality Warnings")
        for warn in report.warnings:
            lines.append(f"- [WARNING] {warn}")
        lines.append("")

    return "\n".join(lines)


# Backwards compatibility and functional aliases
generate_markdown_quality_report = format_quality_markdown_report
generate_quality_markdown_report = format_quality_markdown_report

__all__ = [
    "generate_executive_quality_summary",
    "format_quality_markdown_report",
    "generate_markdown_quality_report",
    "generate_quality_markdown_report"
]
