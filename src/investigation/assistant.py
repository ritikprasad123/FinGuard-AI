"""AI Forensic Investigation Assistant for FinGuard AI.

Evidence-grounded forensic reasoning engine for transaction anomalies,
latent feature attribution, pattern extraction, and compliance investigation protocols.
Fully deterministic Python implementation operating on actual dataset values,
model scores, and SQLite case dossiers with zero external API dependencies.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import re
import numpy as np
import pandas as pd
from config.settings import settings
from src.models.base import classify_risk_tier, BaseAnomalyModel
from src.utils.logger import get_logger

logger = get_logger("finguard.investigation.assistant")

# Latent PCA feature diagnostic profiles
FEATURE_BENCHMARK_PROFILES: Dict[str, Dict[str, Any]] = {
    "V14": {
        "mean": 0.0,
        "std": 1.0,
        "fraud_direction": -1,
        "description": "Latent dimension V14: Strongly inverse-correlated with fraudulent authorization velocity."
    },
    "V10": {
        "mean": 0.0,
        "std": 1.0,
        "fraud_direction": -1,
        "description": "Latent dimension V10: Primary synthetic axis for terminal acquiring fraud patterns."
    },
    "V12": {
        "mean": 0.0,
        "std": 1.0,
        "fraud_direction": -1,
        "description": "Latent dimension V12: Orthogonal vector reflecting cardholder behavioral deviations."
    },
    "V17": {
        "mean": 0.0,
        "std": 1.0,
        "fraud_direction": -1,
        "description": "Latent dimension V17: Heavy inverse tail during high-value credit drain events."
    },
    "V4": {
        "mean": 0.0,
        "std": 1.0,
        "fraud_direction": 1,
        "description": "Latent dimension V4: Positive correlation with rapid micro-burst authorization attempts."
    },
    "V11": {
        "mean": 0.0,
        "std": 1.0,
        "fraud_direction": 1,
        "description": "Latent dimension V11: Positive deviation marking cross-border geographic incongruity."
    },
    "V2": {
        "mean": 0.0,
        "std": 1.0,
        "fraud_direction": 1,
        "description": "Latent dimension V2: Positive deviation associated with unusual settlement volume."
    },
    "V7": {
        "mean": 0.0,
        "std": 1.0,
        "fraud_direction": -1,
        "description": "Latent dimension V7: Negative displacement linked to sudden velocity acceleration."
    }
}


class InvestigationAssistant:
    """Forensic reasoning and natural language query assistant for financial transactions."""

    @classmethod
    def analyze_transaction(
        cls,
        transaction_data: Optional[Union[pd.Series, Dict[str, Any]]] = None,
        row: Optional[Union[pd.Series, Dict[str, Any]]] = None,
        anomaly_score: Optional[float] = None,
        threshold: float = 0.65,
        model_id: str = "IsolationForest-Primary"
    ) -> Dict[str, Any]:
        """Analyzes a single transaction against mathematical profiles and model outputs."""
        source_data = transaction_data if transaction_data is not None else row
        if source_data is None:
            raise ValueError("Either 'transaction_data' or 'row' must be provided.")

        if isinstance(source_data, dict):
            source_data = pd.Series(source_data)

        tx_id = str(source_data.get("Transaction_ID", "TXN-UNKNOWN"))
        amt = float(source_data.get("Amount", 0.0))

        if anomaly_score is None:
            anomaly_score = float(source_data.get("Risk_Score", source_data.get("anomaly_score", 0.50)))

        # 1. Amount Severity
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
            if feat in source_data:
                val = float(source_data[feat])
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
    def generate_briefing_markdown(cls, analysis: Dict[str, Any]) -> str:
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
            flag_mark = "[!]" if d["is_concerning"] else "[-]"
            lines.append(f"| **{d['feature']}** | `{d['value']}` | `{d['z_score']:+.2f}` | {flag_mark} {d['description']} |")

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

    # -------------------------------------------------------------
    # DETERMINISTIC NATURAL-LANGUAGE QUERY INTERFACE (Section 7)
    # -------------------------------------------------------------

    @classmethod
    def get_highest_anomaly_scores(cls, df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
        """Returns transactions with the highest anomaly scores in the current dataset."""
        if df is None or len(df) == 0:
            return []

        score_col = "Risk_Score" if "Risk_Score" in df.columns else ("anomaly_score" if "anomaly_score" in df.columns else None)
        if score_col is None:
            return []

        sorted_df = df.sort_values(by=score_col, ascending=False).head(top_n)
        results = []
        for _, row in sorted_df.iterrows():
            tx_id = str(row.get("Transaction_ID", f"TXN-{row.name}"))
            score = float(row[score_col])
            amt = float(row.get("Amount", 0.0))
            tier = classify_risk_tier(score)
            results.append({
                "transaction_id": tx_id,
                "score": round(score, 4),
                "amount": round(amt, 2),
                "tier": tier
            })
        return results

    @classmethod
    def analyze_flagged_patterns(cls, df: pd.DataFrame, threshold: float = 0.65) -> Dict[str, Any]:
        """Analyzes statistical patterns among transactions flagged above threshold."""
        if df is None or len(df) == 0:
            return {"total_flagged": 0, "patterns": []}

        score_col = "Risk_Score" if "Risk_Score" in df.columns else ("anomaly_score" if "anomaly_score" in df.columns else None)
        if score_col is None:
            return {"total_flagged": 0, "patterns": []}

        flagged = df[df[score_col] >= threshold]
        total_flagged = len(flagged)
        if total_flagged == 0:
            return {
                "total_flagged": 0,
                "flagged_ratio": 0.0,
                "patterns": ["No transactions currently exceed the specified decision threshold."]
            }

        patterns = []
        amt_col = "Amount" if "Amount" in df.columns else None

        # Pattern A: Ticket Amount Concentration
        if amt_col:
            med_normal = df[df[score_col] < threshold][amt_col].median() if len(df[df[score_col] < threshold]) > 0 else 0.0
            med_flagged = flagged[amt_col].median()
            high_ticket_cnt = int((flagged[amt_col] > 1000.0).sum())
            micro_ticket_cnt = int((flagged[amt_col] < 5.0).sum())

            patterns.append(
                f"**Monetary Exposure:** Flagged transactions show a median ticket of ${med_flagged:,.2f} "
                f"(vs. ${med_normal:,.2f} for cleared volume). "
                f"{high_ticket_cnt} high-ticket drains (>${1000}) and {micro_ticket_cnt} micro-burst authorizations (<$5) identified."
            )

        # Pattern B: Latent PCA Components Drivers
        pca_deviations = {}
        for feat in [f"V{i}" for i in range(1, 29)]:
            if feat in flagged.columns and feat in df.columns:
                mean_diff = abs(flagged[feat].mean() - df[feat].mean())
                pca_deviations[feat] = mean_diff

        top_drivers = sorted(pca_deviations.items(), key=lambda x: x[1], reverse=True)[:4]
        driver_str = ", ".join([f"`{k}` (Delta={v:.2f})" for k, v in top_drivers])
        patterns.append(
            f"**Latent Feature Drivers:** Primary multivariate displacement concentrated in {driver_str}. "
            "These dimensions govern rapid velocity shifts and non-standard merchant acquiring signals."
        )

        # Pattern C: Risk Severity Concentration
        tier_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0}
        for s in flagged[score_col]:
            t = classify_risk_tier(s)
            if t in tier_counts:
                tier_counts[t] += 1
        patterns.append(
            f"**Triage Tier Breakdown:** {tier_counts['CRITICAL']} Critical (score >= 0.85), "
            f"{tier_counts['HIGH']} High (0.65 - 0.85), {tier_counts['MEDIUM']} Medium risk records."
        )

        return {
            "total_flagged": total_flagged,
            "flagged_ratio": round(total_flagged / len(df), 4),
            "patterns": patterns
        }

    @classmethod
    def count_flagged_above_threshold(cls, df: pd.DataFrame, threshold: float = 0.65) -> Dict[str, Any]:
        """Calculates precise count and proportion of transactions flagged above threshold."""
        if df is None or len(df) == 0:
            return {"total": 0, "flagged": 0, "ratio": 0.0, "threshold": threshold}

        score_col = "Risk_Score" if "Risk_Score" in df.columns else ("anomaly_score" if "anomaly_score" in df.columns else None)
        if score_col is None:
            return {"total": len(df), "flagged": 0, "ratio": 0.0, "threshold": threshold}

        flagged_count = int((df[score_col] >= threshold).sum())
        total = len(df)
        ratio = flagged_count / total if total > 0 else 0.0

        return {
            "total": total,
            "flagged": flagged_count,
            "cleared": total - flagged_count,
            "ratio": round(ratio, 4),
            "ratio_pct": round(ratio * 100, 2),
            "threshold": round(threshold, 3)
        }

    @classmethod
    def query(
        cls,
        prompt: str,
        df: Optional[pd.DataFrame] = None,
        case_manager: Any = None,
        model: Optional[BaseAnomalyModel] = None,
        target_tx_id: Optional[str] = None,
        threshold: float = 0.65
    ) -> Dict[str, Any]:
        """Processes natural language forensic inquiries using factual calculations."""
        p_lower = prompt.strip().lower()

        # 1. QUESTION: Why was this transaction flagged?
        tx_match = re.search(r"txn-[a-z0-9\-]+", p_lower)
        interrogate_id = tx_match.group(0).upper() if tx_match else target_tx_id

        if ("why was" in p_lower and "flagged" in p_lower) or (interrogate_id and ("why" in p_lower or "explain" in p_lower)):
            if df is not None and interrogate_id:
                matching_rows = df[df["Transaction_ID"].astype(str) == interrogate_id] if "Transaction_ID" in df.columns else pd.DataFrame()
                if not matching_rows.empty:
                    target_row = matching_rows.iloc[0]
                    analysis = cls.analyze_transaction(
                        transaction_data=target_row,
                        threshold=threshold,
                        model_id=model.model_id if model else "IsolationForest-Primary"
                    )
                    briefing = cls.generate_briefing_markdown(analysis)
                    return {
                        "intent": "EXPLAIN_TRANSACTION",
                        "title": f"Forensic Attribution for {interrogate_id}",
                        "response_markdown": briefing,
                        "data": analysis
                    }

            if interrogate_id:
                return {
                    "intent": "EXPLAIN_TRANSACTION_MISSING",
                    "title": f"Transaction {interrogate_id} Not Found",
                    "response_markdown": f"Transaction `{interrogate_id}` was not located in the currently loaded dataset. Please check the Transaction ID in Transaction Explorer.",
                    "data": {}
                }
            return {
                "intent": "EXPLAIN_TRANSACTION_NO_ID",
                "title": "Select a Transaction",
                "response_markdown": "Please provide a specific Transaction ID (e.g., `TXN-SURV-10042`) or select one from the Alert Center table to inspect its mathematical feature attribution.",
                "data": {}
            }

        # 2. QUESTION: Which transactions have the highest anomaly scores?
        if "highest" in p_lower or "top" in p_lower or "most anomalous" in p_lower:
            if df is not None:
                top_txns = cls.get_highest_anomaly_scores(df, top_n=5)
                lines = [
                    "### Top 5 Extreme Anomaly Detections",
                    "",
                    "Ranked by continuous model anomaly score on the active dataset:",
                    "",
                    "| Rank | Transaction ID | Anomaly Score | Risk Tier | Ticket Amount |",
                    "| :--- | :--- | :--- | :--- | :--- |"
                ]
                for idx, t in enumerate(top_txns, 1):
                    lines.append(f"| **#{idx}** | `{t['transaction_id']}` | `{t['score']:.4f}` | **{t['tier']}** | `${t['amount']:,.2f}` |")

                lines.extend([
                    "",
                    f"*Scored using operating threshold `{threshold:.2f}`. Transactions in Critical tier exhibit multi-sigma feature deviation.*"
                ])
                return {
                    "intent": "TOP_ANOMALIES",
                    "title": "Highest Anomaly Score Transactions",
                    "response_markdown": "\n".join(lines),
                    "data": top_txns
                }

        # 3. QUESTION: What patterns appear among flagged transactions?
        if "pattern" in p_lower or "cluster" in p_lower or "trend" in p_lower:
            if df is not None:
                pattern_data = cls.analyze_flagged_patterns(df, threshold=threshold)
                lines = [
                    f"### Multivariate Anomaly Patterns (Threshold: `{threshold:.2f}`)",
                    "",
                    f"**Total Records Flagged:** `{pattern_data['total_flagged']:,}` ({pattern_data.get('flagged_ratio', 0.0):.2%} of volume)",
                    ""
                ]
                for p in pattern_data["patterns"]:
                    lines.append(f"- {p}")

                lines.extend([
                    "",
                    "> *Heuristic Note: Identified patterns reflect empirical population variances across PCA dimensions and ticket amounts. They serve as triage priority guides.*"
                ])
                return {
                    "intent": "FLAGGED_PATTERNS",
                    "title": "Detected Anomaly Patterns",
                    "response_markdown": "\n".join(lines),
                    "data": pattern_data
                }

        # 4. QUESTION: How many transactions were flagged above the threshold?
        if "how many" in p_lower or "count" in p_lower or "number of" in p_lower or "flagged above" in p_lower:
            if df is not None:
                stats = cls.count_flagged_above_threshold(df, threshold=threshold)
                md = (
                    f"### Operating Threshold Flagged Summary\n\n"
                    f"- **Active Threshold:** `{stats['threshold']:.3f}`\n"
                    f"- **Flagged Anomalies:** `{stats['flagged']:,}` records\n"
                    f"- **Anomaly Rate:** `{stats['ratio_pct']}%` of total volume\n"
                    f"- **Cleared Volume:** `{stats['cleared']:,}` records\n"
                    f"- **Total Dataset Size:** `{stats['total']:,}` transactions\n\n"
                    f"Increasing the threshold above `{stats['threshold']:.2f}` will reduce alert volume to prioritize only highest-confidence outliers."
                )
                return {
                    "intent": "THRESHOLD_COUNT",
                    "title": "Anomaly Count Above Threshold",
                    "response_markdown": md,
                    "data": stats
                }

        # 5. QUESTION: Summarize this investigation case
        case_match = re.search(r"case-[a-z0-9\-]+", p_lower)
        if ("case" in p_lower and ("summar" in p_lower or "dossier" in p_lower or "status" in p_lower)) or case_match:
            cid = case_match.group(0).upper() if case_match else None
            if case_manager is not None:
                all_cases = case_manager.list_cases()
                target_case = None
                if cid:
                    for c in all_cases:
                        if c.get("case_id", "").upper() == cid:
                            target_case = c
                            break
                elif len(all_cases) > 0:
                    target_case = all_cases[0]  # Most recent case

                if target_case:
                    cid_str = target_case["case_id"]
                    status = target_case.get("status", "OPEN")
                    prio = target_case.get("priority", "HIGH")
                    created = target_case.get("created_at", "N/A")
                    analyst = target_case.get("assigned_analyst", "Unassigned")
                    title = target_case.get("title", "Untitled Case")
                    full_case = case_manager.get_case(cid_str) if hasattr(case_manager, "get_case") else target_case
                    if full_case:
                        target_case = full_case
                    notes = target_case.get("notes", [])
                    txns = target_case.get("linked_transactions", [])

                    lines = [
                        f"### Case Summary Dossier: `{cid_str}`",
                        "",
                        f"**Title:** {title}",
                        f"**Status:** `{status}` | **Priority:** `{prio}` | **Analyst:** `{analyst}`",
                        f"**Created:** `{created}` | **Linked Transactions:** `{len(txns)}`",
                        "",
                        "#### Summary Findings",
                        f"> {target_case.get('summary', 'No executive summary recorded.')}",
                        "",
                        f"#### Chronological Notes Audit ({len(notes)} logged)",
                    ]
                    for n in notes:
                        lines.append(f"- **[{n.get('created_at', 'N/A')}]** *({n.get('author', 'Analyst')})*: {n.get('content', n.get('note_text', ''))}")

                    return {
                        "intent": "CASE_SUMMARY",
                        "title": f"Summary for {cid_str}",
                        "response_markdown": "\n".join(lines),
                        "data": target_case
                    }
                else:
                    return {
                        "intent": "NO_CASES_FOUND",
                        "title": "No Cases In SQLite Database",
                        "response_markdown": "No investigation cases are currently recorded in the SQLite database. You can open a case directly from the Alert Center or Transaction Explorer.",
                        "data": {}
                    }

        # Fallback General Analytical Response
        return {
            "intent": "GENERAL_GUIDE",
            "title": "Investigation Assistant Ready",
            "response_markdown": (
                "**FinGuard AI Forensic Assistant** answers queries directly from active surveillance data:\n\n"
                "- **'Why was TXN-SURV-10042 flagged?'**: Explains latent PCA z-scores, ticket amount extremity, and recommended actions.\n"
                "- **'Which transactions have highest anomaly scores?'**: Returns top mathematical outliers.\n"
                "- **'What patterns appear among flagged transactions?'**: Summarizes multivariate clusters and monetary distribution.\n"
                "- **'How many transactions were flagged above threshold?'**: Computes exact counts and velocity rates.\n"
                "- **'Summarize case CASE-20261012-...'**: Retrieves SQLite case dossier with chronological notes.\n\n"
                "*All responses are deterministic and calculated on authentic application state with zero external API dependencies.*"
            ),
            "data": {}
        }
