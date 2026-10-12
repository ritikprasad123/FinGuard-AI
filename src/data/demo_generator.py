"""Synthetic Demo Transaction Generator for FinGuard AI.

Generates clearly marked synthetic transactions with realistic distributions
(lognormal financial amounts, PCA clusters, and synthetic anomaly signatures)
explicitly labeled as DEMO DATA for development, testing, and UI evaluation.
"""

from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("finguard.data.demo_generator")


def generate_synthetic_demo_data(
    n_samples: int = 5000,
    fraud_ratio: float = 0.005,
    random_state: int = 42,
    save_to_file: bool = True
) -> pd.DataFrame:
    """Generates synthetic financial transactions strictly marked as DEMO DATA.

    Simulates retail transaction streams with:
    - Normal transactions: log-normal amounts (median ~$65, max ~$1,500), standard normal PCA features.
    - Fraudulent/Anomalous transactions: heavy tail amounts ($1,200 - $6,500) or micro-bursts,
      and distinct synthetic deviations in V14, V17, V12, V10 (mimicking known fraud patterns).

    Args:
        n_samples: Total number of synthetic transactions to generate.
        fraud_ratio: Proportion of anomalous transactions (default 0.005 = 0.5%).
        random_state: Random seed for deterministic reproducibility.
        save_to_file: Whether to save the generated CSV to settings.DEFAULT_DEMO_PATH.

    Returns:
        DataFrame containing synthetic transactions with all required benchmark features.
    """
    logger.info(f"Generating {n_samples} synthetic DEMO transactions (fraud_ratio={fraud_ratio:.3%}).")
    rng = np.random.RandomState(random_state)

    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_normal = n_samples - n_fraud

    # 1. Simulate Time (in seconds, sequentially over a 48-hour window = 172,800s)
    normal_times = np.sort(rng.uniform(0, 172800, n_normal))
    fraud_times = np.sort(rng.uniform(3600, 169200, n_fraud))

    # 2. Simulate Amount (USD)
    # Normal: Log-normal distribution with typical retail ticket sizes
    normal_amounts = np.round(rng.lognormal(mean=3.8, sigma=1.1, size=n_normal), 2)
    normal_amounts = np.clip(normal_amounts, 0.50, 2500.0)

    # Fraud: Bimodal (either high-ticket rapid drain or micro-authorization test)
    fraud_amounts = []
    for _ in range(n_fraud):
        if rng.rand() < 0.75:
            amt = np.round(rng.uniform(850.0, 5200.0), 2)
        else:
            amt = np.round(rng.uniform(0.99, 4.99), 2)
        fraud_amounts.append(amt)
    fraud_amounts = np.array(fraud_amounts)

    # 3. Simulate PCA Features V1 - V28
    normal_pca = rng.randn(n_normal, 28)

    fraud_pca = rng.randn(n_fraud, 28)
    fraud_pca[:, 13] -= rng.uniform(4.0, 9.0, size=n_fraud)  # V14
    fraud_pca[:, 16] -= rng.uniform(3.5, 8.5, size=n_fraud)  # V17
    fraud_pca[:, 11] -= rng.uniform(3.0, 7.5, size=n_fraud)  # V12
    fraud_pca[:, 9] -= rng.uniform(2.5, 6.0, size=n_fraud)   # V10
    fraud_pca[:, 3] += rng.uniform(2.5, 6.0, size=n_fraud)   # V4
    fraud_pca[:, 10] += rng.uniform(2.0, 5.0, size=n_fraud)  # V11

    # 4. Construct DataFrames
    normal_dict = {
        settings.TIME_COL: normal_times,
        settings.AMOUNT_COL: normal_amounts,
        settings.TARGET_COL: np.zeros(n_normal, dtype=int),
    }
    for i in range(28):
        normal_dict[f"V{i+1}"] = normal_pca[:, i]
    df_normal = pd.DataFrame(normal_dict)

    fraud_dict = {
        settings.TIME_COL: fraud_times,
        settings.AMOUNT_COL: fraud_amounts,
        settings.TARGET_COL: np.ones(n_fraud, dtype=int),
    }
    for i in range(28):
        fraud_dict[f"V{i+1}"] = fraud_pca[:, i]
    df_fraud = pd.DataFrame(fraud_dict)

    df_combined = pd.concat([df_normal, df_fraud], ignore_index=True)
    df_combined = df_combined.sort_values(by=settings.TIME_COL).reset_index(drop=True)

    # 5. Enrich with Enterprise Metadata for UI & Auditing
    base_time = datetime(2026, 10, 10, 0, 0, 0)
    df_combined["Timestamp"] = df_combined[settings.TIME_COL].apply(
        lambda s: (base_time + timedelta(seconds=float(s))).strftime("%Y-%m-%d %H:%M:%S")
    )
    df_combined["Transaction_ID"] = [f"TXN-DEMO-{i+100001:06d}" for i in range(len(df_combined))]
    df_combined["_DATA_SOURCE"] = "DEMO DATA (SYNTHETIC)"
    df_combined["_IS_DEMO"] = True

    risk_scores = []
    reasons = []
    for idx, row in df_combined.iterrows():
        is_fraud = int(row[settings.TARGET_COL]) == 1
        amt = float(row[settings.AMOUNT_COL])
        v14_val = float(row["V14"])
        v17_val = float(row["V17"])

        if is_fraud:
            score = round(rng.uniform(0.86, 0.99), 3)
            reason = f"Critical PCA deviation (V14={v14_val:.1f}, V17={v17_val:.1f}) | High outlier score"
        elif amt > 1800:
            score = round(rng.uniform(0.68, 0.82), 3)
            reason = f"High transaction value (${amt:,.2f}) exceeds 99th percentile"
        elif v14_val < -2.5 or v17_val < -2.5:
            score = round(rng.uniform(0.45, 0.64), 3)
            reason = "Mild latent feature variance anomaly"
        else:
            score = round(rng.uniform(0.01, 0.28), 3)
            reason = "Standard behavioral pattern"

        risk_scores.append(score)
        reasons.append(reason)

    df_combined["Risk_Score"] = risk_scores
    df_combined["Risk_Reason"] = reasons

    feature_cols = settings.REQUIRED_BENCHMARK_FEATURES
    ordered_cols = ["Transaction_ID", "Timestamp"] + feature_cols + [settings.TARGET_COL, "Risk_Score", "Risk_Reason", "_DATA_SOURCE", "_IS_DEMO"]
    df_combined = df_combined[ordered_cols]

    if save_to_file:
        try:
            settings.DATA_DEMO_DIR.mkdir(parents=True, exist_ok=True)
            df_combined.to_csv(settings.DEFAULT_DEMO_PATH, index=False)
            logger.info(f"Saved demo dataset to: {settings.DEFAULT_DEMO_PATH}")
        except Exception as exc:
            logger.warning(f"Could not persist demo dataset to disk: {exc}")

    return df_combined
