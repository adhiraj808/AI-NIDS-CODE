"""
Threat Analysis and Batch Inference Module for AI-NIDS.
Processes unknown/test network captures, executes offline threat classification,
generates confidence scores, and formats output tables for SOC analysis.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd

from config.settings import BENIGN_LABEL, MALICIOUS_LABEL
from src.preprocessing import PreprocessingPipeline

logger = logging.getLogger("AI-NIDS.Prediction")


class ThreatClassifier:
    """
    Inference engine that takes arbitrary network flow CSVs,
    preprocesses them using a fitted pipeline, and runs prediction with a trained model.
    """

    def __init__(self, model: Any, preprocessor: PreprocessingPipeline):
        self.model = model
        self.preprocessor = preprocessor

    def classify_traffic(
        self,
        df_raw: pd.DataFrame,
        confidence_threshold: float = 0.50,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Runs batch classification on input flow records.

        Args:
            df_raw: Incoming network flows DataFrame
            confidence_threshold: Cutoff above which flow is tagged as MALICIOUS

        Returns:
            df_results: Original DataFrame enriched with 'Predicted_Label', 'Malicious_Probability', 'Threat_Level'
            summary: Aggregated threat statistics
        """
        logger.info("Executing offline threat classification on %d flow records...", len(df_raw))

        # Check if incoming data contains raw Label or attack columns and exclude them from feature transformation
        df_features = df_raw.copy()
        if "Label" in df_features.columns:
            df_features = df_features.drop(columns=["Label"])
        if "Original_Attack_Type" in df_features.columns:
            df_features = df_features.drop(columns=["Original_Attack_Type"])

        # Transform features using fitted pipeline
        X = self.preprocessor.transform(df_features)

        # Predict probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(X)
            probs = 1.0 / (1.0 + np.exp(-decision))
        else:
            # Fallback to hard predictions
            preds = self.model.predict(X)
            probs = preds.astype(float)

        # Apply threshold
        binary_preds = (probs >= confidence_threshold).astype(int)
        predicted_labels = np.where(binary_preds == 1, MALICIOUS_LABEL, BENIGN_LABEL)

        # Threat severity mapping based on confidence
        threat_levels = []
        for p in probs:
            if p >= 0.85:
                threat_levels.append("CRITICAL")
            elif p >= 0.65:
                threat_levels.append("HIGH")
            elif p >= confidence_threshold:
                threat_levels.append("MEDIUM")
            elif p >= 0.30:
                threat_levels.append("LOW (Suspect)")
            else:
                threat_levels.append("CLEAN")

        # Construct enriched results DataFrame
        df_results = df_raw.copy()
        df_results.insert(0, "Predicted_Label", predicted_labels)
        df_results.insert(1, "Malicious_Probability", np.round(probs, 4))
        df_results.insert(2, "Threat_Level", threat_levels)

        total = len(df_results)
        malicious_count = int((binary_preds == 1).sum())
        benign_count = total - malicious_count

        summary = {
            "total_inspected_flows": total,
            "benign_flows": benign_count,
            "malicious_flows": malicious_count,
            "malicious_ratio_percent": round((malicious_count / total) * 100, 2) if total else 0.0,
            "critical_threats": threat_levels.count("CRITICAL"),
            "high_threats": threat_levels.count("HIGH"),
            "medium_threats": threat_levels.count("MEDIUM"),
            "mean_malicious_probability": round(float(np.mean(probs)), 4),
            "confidence_threshold_applied": confidence_threshold,
        }

        logger.info(
            "Classification complete: %d BENIGN, %d MALICIOUS (CRITICAL: %d, HIGH: %d)",
            benign_count,
            malicious_count,
            summary["critical_threats"],
            summary["high_threats"],
        )

        return df_results, summary
