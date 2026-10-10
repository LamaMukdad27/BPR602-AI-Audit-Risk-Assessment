
"""Inference functions for the AI-Powered Audit Risk Assessment Assistant."""

from typing import Any

import numpy as np
import pandas as pd


def run_inference(
    model_features: pd.DataFrame,
    preprocessor: Any,
    fraud_model: Any,
    isolation_forest: Any,
    anomaly_scaler: Any,
    anomaly_score_scaler: Any,
    anomaly_features: list[str],
    fraud_weight: float = 0.70,
    anomaly_weight: float = 0.30,
    risk_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Run fraud detection, anomaly detection, and combined risk scoring.

    Parameters
    ----------
    model_features:
        Engineered GL features. Must contain the columns expected by
        the saved preprocessor and Isolation Forest.
    preprocessor:
        Fitted training-time preprocessing pipeline.
    fraud_model:
        Fitted XGBoost fraud classifier.
    isolation_forest:
        Fitted Isolation Forest model.
    anomaly_scaler:
        Fitted scaler for Isolation Forest input features.
    anomaly_score_scaler:
        Fitted scaler used to normalize anomaly scores.
    anomaly_features:
        Exact ordered list of Isolation Forest input features.
    fraud_weight, anomaly_weight:
        Weights used in the combined risk score.
    risk_thresholds:
        Optional category thresholds. Defaults to the validated
        project thresholds.

    Returns
    -------
    pandas.DataFrame
        Input features plus fraud probability, anomaly score,
        combined risk score, and risk category.
    """

    if model_features.empty:
        raise ValueError("The input contains no GL records.")

    if not np.isclose(fraud_weight + anomaly_weight, 1.0):
        raise ValueError("Risk weights must sum to 1.0.")

    if not model_features.columns.is_unique:
        raise ValueError("Duplicate feature column names were found.")

    required_anomaly_features = list(anomaly_features)
    missing_anomaly = [
        col for col in required_anomaly_features
        if col not in model_features.columns
    ]

    if missing_anomaly:
        raise ValueError(
            "Missing Isolation Forest features: "
            + ", ".join(missing_anomaly)
        )

    # Use the saved preprocessor and classifier.
    processed_features = preprocessor.transform(model_features)
    fraud_probability = fraud_model.predict_proba(
        processed_features
    )[:, 1]

    # Reproduce the training-time anomaly scoring sequence.
    anomaly_input = model_features[required_anomaly_features]
    anomaly_scaled = anomaly_scaler.transform(anomaly_input)

    # Isolation Forest assigns lower decision_function scores
    # to more anomalous observations.
    raw_anomaly_score = -isolation_forest.decision_function(
        anomaly_scaled
    )

    normalized_anomaly_score = anomaly_score_scaler.transform(
        raw_anomaly_score.reshape(-1, 1)
    ).ravel()

    normalized_anomaly_score = np.clip(
        normalized_anomaly_score, 0.0, 1.0
    )

    combined_risk_score = np.clip(
        fraud_weight * fraud_probability
        + anomaly_weight * normalized_anomaly_score,
        0.0,
        1.0,
    )

    thresholds = risk_thresholds or {
        "moderate": 0.30,
        "high": 0.60,
        "critical": 0.80,
    }

    moderate_threshold = thresholds["moderate"]
    high_threshold = thresholds["high"]
    critical_threshold = thresholds["critical"]

    if not (
        0 <= moderate_threshold
        <= high_threshold
        <= critical_threshold
        <= 1
    ):
        raise ValueError("Risk thresholds must be ordered between 0 and 1.")

    risk_category = np.select(
        [
            combined_risk_score < moderate_threshold,
            combined_risk_score < high_threshold,
            combined_risk_score < critical_threshold,
        ],
        ["Low", "Moderate", "High"],
        default="Critical",
    )

    results = model_features.copy()
    results["fraud_probability"] = fraud_probability
    results["anomaly_score"] = normalized_anomaly_score
    results["combined_risk_score"] = combined_risk_score
    results["risk_category"] = risk_category

    return results
