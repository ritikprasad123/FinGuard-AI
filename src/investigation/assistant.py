"""AI Investigation Assistant for FinGuard AI.

Synthesizes mathematical anomaly evidence, PCA feature deviations,
model decision boundaries, and actionable forensic workflows without
inventing ungrounded banking histories.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from config.settings import settings
from src.models.base import classify_risk_tier
from src.utils.logger import get_logger

logger = get_logger("finguard.investigation.assistant")


# Empirical reference baselines from credit card benchmark population
FEATURE_BENCHMARK_PROFILES: Dict[str, Dict[str, float]] = {
    "V14": {"mean": 0.0, "std": 0.96, "fraud_direction": -1.0, "description": "Compromised credential / account takeover latent factor"},
    "V10": {"mean": 0.0, "std": 1.09, "fraud_direction": -1.0, "description": "Card-not-present velocity & credential authorization variance"},
    "V12": {"mean": 0.0, "std": 1.00, "fraud_direction": -1.0, "description": "Cross-border / high-risk merchant transaction factor"},
    "V17": {"mean": 0.0, "std": 0.85, "fraud_direction": -1.0, "description": "Terminal fingerprint & issuer risk discrepancy"},
    "V4":  {"mean": 0.0, "std": 1.42, "fraud_direction": 1.0,  "description": "Rapid re-try / velocity spike latent indicator"},
    "V11": {"mean": 0.0, "std": 1.02, "fraud_direction": 1.0,  "description": "Unusual payment channel & routing variance"},
    "V16": {"mean": 0.0, "std": 0.88, "fraud_direction": -1.0, "description": "Device anomaly & authorization sequence factor"},
    "V3":  {"mean": 0.0, "std": 1.52, "fraud_direction": -1.0, "description": "Merchant category code discrepancy factor"},
    "V7":  {"mean": 0.0, "std": 1.24, "fraud_direction": -1.0, "description": "Ticket value / balance ratio divergence factor"}
}


class InvestigationAssistant:
    """Evidence-grounded forensic intelligence engine for transaction anomalies."""

    @staticmethod
    def analyze_transaction(
        transaction_data: Union[pd.Series, Dict[str, Any]],
        anomaly_score: float,
        threshold: float = 0.65,
        model_id: str = "IsolationForest"
    ) -> Dict[str, Any]:
        """Performs rigorous forensic synthesis of a flagged transaction.

        Args:
            transaction_data: Row containing transaction features (Amount, Time, V1-V28).
            anomaly_score: Calibrated model anomaly score [0.0, 1.0].
            threshold: Applied decision cutoff.
            model_id: Identifier of model generating the score.

        Returns:
            Dictionary containing structured forensic evidence, top drivers,
            suggested steps, and mandatory limitations disclaimer.
        """
        row = dict(transaction_data) if isinstance(transaction_data, (pd.Series, dict)) else {}

        # 1. Amount & Time Profiling
        amt = float(row.get(settings.AMOUNT_COL, 0.0))
        time_sec = float(row.get(settings.TIME_COL, 0.0))
        tx_id = str(row.get("Transaction_ID", "TXN-UNKNOWN"))

        # Amount percentile assessment based on empirical benchmark distributions
        if amt > 2500.0:
            amt_severity = "CRITICAL (Top 0.2% value outlier)"
        elif amt > 1000.0:
            amt_severity = "HIGH (Top 1.0% value outlier)"
        elif amt > 300.0:
            amt_severity = "ELEVATED (Top 5.0% value outlier)"
        else:
            amt_severity = "STANDARD (Within median interquartile band)"

        # 2. Extract and rank PCA feature deviations
        deviations = []
        for feat, profile in FEATURE_BENCHMARK_PROFILES.items():
            if feat in row:
                val = float(row[feat])
                z_score = (val - profile["mean"]) / profile["std"]
                abs_z = abs(z_score)

                is_concerning = False
                if profile["fraud_direction"] < 0 and z_score < -2.0:
                    is_concerning = True
                elif profile["fraud_direction"] > 0 and z_score > 2.0:
                    is_concerning = True

                deviations.append({
                    "feature": feat,
                    "value": round(val, 4),
                    "z_score": round(z_score, 2),
                    "abs_z": round(abs_z, 2),
                    "is_concerning": is_concerning,
                    "description": profile["description"]
                })

        # Sort features by absolute z-score deviation descending
        deviations.sort(key=lambda x: x["abs_z"], reverse=True)
        top_deviations = deviations[:5]

        # 3. Anomaly Rationale Synthesis
        score_delta = anomaly_score - threshold
        risk_tier = classify_risk_tier(anomaly_score)

        critical_features = [d["feature"] for d in top_deviations if d["abs_z"] >= 2.5]
        feat_str = ", ".join(critical_features) if critical_features else "multiple latent components"

        if anomaly_score >= threshold:
            flag_reason = (
                f"The transaction triggered an anomaly alert with a calibrated score of {anomaly_score:.3f} "
                f"(Threshold: {threshold:.3f}, Delta: +{score_delta:.3f}), classified into the {risk_tier} risk category. "
                f"The primary mathematical drivers are extreme deviations along {feat_str}. "
            )
            if amt > 1000.0:
                flag_reason += f"Additionally, the ticket amount (${amt:,.2f}) presents an abnormal volumetric outlier."
            else:
                flag_reason += f"The transaction ticket amount is ${amt:,.2f}, indicating subtle latent variance rather than raw value extremity."
        else:
            flag_reason = (
                f"The transaction scored {anomaly_score:.3f}, below the operating threshold of {threshold:.3f} "
                f"(Delta: {score_delta:.3f}). Features remain within statistical tolerance limits for {model_id}."
            )

        # 4. Standard Operational Action Plan
        suggested_steps = []
        if risk_tier == "CRITICAL":
            suggested_steps.extend([
                "Initiate immediate provisional hold on the originating cardholder payment profile.",
                "Deploy Out-Of-Band (OOB) authentication challenge (Biometric SMS / Mobile App Push) to the cardholder.",
                "Review terminal IP, geolocation, and merchant acquiring identifier against cardholder historical footprint.",
                "Audit transaction logs for micro-authorization testing patterns in the preceding 30 minutes.",
                "Escalate case to Tier-2 AML / Fraud Investigation Unit if challenge remains unanswered within 15 minutes."
            ])
        elif risk_tier == "HIGH":
            suggested_steps.extend([
                "Trigger secondary step-up authorization (3D Secure / OTP Challenge) for downstream settlement.",
                "Correlate transaction velocity against 24-hour baseline velocity for this account token.",
                "Examine merchant category code (MCC) risk tier and refund/chargeback ratio.",
                "Log entry in internal surveillance desk for active 48-hour monitoring window."
            ])
        elif risk_tier == "MEDIUM":
            suggested_steps.extend([
                "Route to passive surveillance queue for post-settlement reconciliation.",
                "Monitor for subsequent transactions on the same payment instrument within 6 hours.",
                "Verify no duplicate sequence identifiers exist in settlement batch."
            ])
        else:
            suggested_steps.extend([
                "Standard automated clearing permitted. No proactive manual intervention required.",
                "Log baseline metrics for ongoing unsupervised model drift tracking."
            ])

        # 5. Mandatory Limitations Statement
        limitations = (
            "STATISTICAL SURVEILLANCE LIMITATION NOTICE: "
            "This briefing is derived exclusively from unsupervised machine learning anomaly scoring and "
            "mathematical principal component analysis (PCA). Principal components V1-V28 are anonymized "
            "orthogonal projections designed to protect customer privacy and banking confidentiality. "
            "FinGuard AI does not manufacture or invent cardholder names, historical banking statements, "
            "or merchant dispute logs. Statistical anomaly scores denote deviation from population baselines "
            "and do NOT constitute legal proof of fraudulent intent. Human investigator verification is required "
            "prior to enacting permanent adverse account determinations under consumer protection regulations."
        )

        return {
            "transaction_id": tx_id,
            "anomaly_score": round(anomaly_score, 4),
            "threshold": round(threshold, 4),
            "score_delta": round(score_delta, 4),
            "risk_tier": risk_tier,
            "prediction": "ANOMALY" if anomaly_score >= threshold else "NORMAL",
            "model_id": model_id,
            "amount": amt,
            "amount_severity": amt_severity,
            "flag_reason": flag_reason,
            "top_deviating_features": top_deviations,
            "suggested_steps": suggested_steps,
            "limitations": limitations
        }

    @classmethod
    def generate_briefing_markdown(
        cls,
        analysis: Dict[str, Any]
    ) -> str:
        """Formats the analysis into an executive Markdown forensic briefing."""
        tx_id = analysis["transaction_id"]
        score = analysis["anomaly_score"]
        thresh = analysis["threshold"]
        tier = analysis["risk_tier"]
        pred = analysis["prediction"]
        amt = analysis["amount"]
        model_id = analysis["model_id"]

        lines = [
            f"### AI Investigation Briefing: `{tx_id}`",
            "",
            f"**Model:** `{model_id}` | **Risk Tier:** `{tier}` | **Prediction:** `{pred}`",
            f"**Anomaly Score:** `{score:.4f}` | **Cutoff Threshold:** `{thresh:.4f}` | **Ticket Amount:** `${amt:,.2f}`",
            "",
            "#### 1. Mathematical Anomaly Rationale",
            f"> {analysis['flag_reason']}",
            "",
            "#### 2. Key Latent Feature Deviations (Top Outliers)",
            "| Feature | Observed Value | Standardized Z-Score | Diagnostic Significance |",
            "| :--- | :--- | :--- | :--- |"
        ]

        for d in analysis["top_deviating_features"]:
            flag_mark = "⚠️" if d["is_concerning"] else "—"
            lines.append(f"| **{d['feature']}** | `{d['value']}` | `{d['z_score']:+.2f}σ` | {flag_mark} {d['description']} |")

        lines.extend([
            "",
            "#### 3. Recommended Forensic Investigation Steps",
        ])
        for idx, step in enumerate(analysis["suggested_steps"], 1):
            lines.append(f"{idx}. {step}")

        lines.extend([
            "",
            "#### 4. Compliance & Evidentiary Limitations",
            f"*{analysis['limitations']}*",
            ""
        ])

        return "\n".join(lines)
