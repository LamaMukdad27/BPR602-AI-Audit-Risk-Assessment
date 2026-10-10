
"""Inference functions for the AI-Powered Audit Risk Assessment Assistant."""

from typing import Any

import numpy as np
import pandas as pd


# Validated project risk methodology.
FRAUD_WEIGHT = 0.70
ANOMALY_WEIGHT = 0.30

RISK_THRESHOLDS = {
    "moderate": 0.30,
    "high": 0.60,
    "critical": 0.80,
}


def run_inference(
    model_features: pd.DataFrame,
    preprocessor: Any,
    fraud_model: Any,
    isolation_forest: Any,
    anomaly_scaler: Any,
    anomaly_score_scaler: Any,
    feature_definitions: dict,
) -> pd.DataFrame:
    """
    Run fraud detection, anomaly detection, and combined risk scoring.

    Input must contain the engineered features required by the
    saved models, not the original unprocessed GL file.
    """

    if model_features.empty:
        raise ValueError("The uploaded GL contains no records.")

    if not model_features.columns.is_unique:
        raise ValueError("Duplicate feature column names were found.")

    numeric_features = feature_definitions["numeric_features"]
    categorical_features = feature_definitions["categorical_features"]
    anomaly_features = feature_definitions["anomaly_features"]

    model_columns = numeric_features + categorical_features

    missing_model_features = [
        col for col in model_columns
        if col not in model_features.columns
    ]

    missing_anomaly_features = [
        col for col in anomaly_features
        if col not in model_features.columns
    ]

    if missing_model_features:
        raise ValueError(
            "Missing required model features: "
            + ", ".join(missing_model_features)
        )

    if missing_anomaly_features:
        raise ValueError(
            "Missing required anomaly features: "
            + ", ".join(missing_anomaly_features)
        )

    # Preserve all input columns, but select model features in
    # the exact order documented by the saved feature definitions.
    results = model_features.copy()
    X_model = results[model_columns].copy()

    # Convert numerical model inputs to numeric values.
    for column in numeric_features:
        X_model[column] = pd.to_numeric(
            X_model[column], errors="coerce"
        )

    # Match the saved preprocessing pipeline and XGBoost model.
    processed_features = preprocessor.transform(X_model)

    fraud_probability = fraud_model.predict_proba(
        processed_features
    )[:, 1]

    # Isolation Forest uses its own ordered feature set and scaler.
    X_anomaly = results[anomaly_features].copy()

    for column in anomaly_features:
        X_anomaly[column] = pd.to_numeric(
            X_anomaly[column], errors="coerce"
        )

    if X_anomaly.isna().any().any():
        missing_counts = X_anomaly.isna().sum()
        affected = missing_counts[missing_counts > 0].to_dict()

        raise ValueError(
            "Missing or non-numeric values remain in Isolation "
            f"Forest inputs: {affected}. Complete feature "
            "preparation before running inference."
        )

    anomaly_scaled = anomaly_scaler.transform(X_anomaly)

    # Lower Isolation Forest decision scores indicate more unusual
    # observations, so negate the scores before normalization.
    raw_anomaly_score = -isolation_forest.decision_function(
        anomaly_scaled
    )

    normalized_anomaly_score = anomaly_score_scaler.transform(
        raw_anomaly_score.reshape(-1, 1)
    ).ravel()

    normalized_anomaly_score = np.clip(
        normalized_anomaly_score, 0.0, 1.0
    )

    # Apply the project's validated 70% / 30% risk methodology.
    combined_risk_score = np.clip(
        FRAUD_WEIGHT * fraud_probability
        + ANOMALY_WEIGHT * normalized_anomaly_score,
        0.0,
        1.0,
    )

    risk_category = np.select(
        [
            combined_risk_score < RISK_THRESHOLDS["moderate"],
            combined_risk_score < RISK_THRESHOLDS["high"],
            combined_risk_score < RISK_THRESHOLDS["critical"],
        ],
        ["Low", "Moderate", "High"],
        default="Critical",
    )

    results["fraud_probability"] = fraud_probability
    results["anomaly_score"] = normalized_anomaly_score
    results["combined_risk_score"] = combined_risk_score
    results["risk_category"] = risk_category

    return results
