"""
Explainable AI (XAI) Module for AI-NIDS.
Provides tree-based feature importance extraction, SHAP (SHapley Additive exPlanations)
analysis on representative samples, and domain-informed cybersecurity interpretations.
Distinguishes statistical importance from causal mechanisms.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd

from config.settings import SHAP_MAX_DISPLAY, SHAP_SAMPLE_SIZE

logger = logging.getLogger("AI-NIDS.Explainability")

# Gracefully handle shap import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP is not installed in the current environment. Tree feature importances remain fully operational.")


# Domain interpretations for key CICIDS2017 flow features
CYBERSECURITY_FEATURE_TAXONOMY = {
    "Flow Duration": "Total duration of the bidirectional network flow in microseconds. Short bursts often signify DoS or scan probing.",
    "Total Fwd Packets": "Number of packets transmitted in the forward direction. High volumes with small payloads indicate SYN flood or DDoS.",
    "Total Backward Packets": "Response packets returned by destination host. Very low backward packets indicate unreachable targets or half-open scans.",
    "Total Length of Fwd Packets": "Total bytes sent in forward direction. Large values can indicate data exfiltration or buffer overflow payloads.",
    "Fwd Packet Length Max": "Maximum packet size in forward direction. Distinct signatures differentiate interactive traffic from automated exploit payloads.",
    "Bwd Packet Length Max": "Maximum packet size in reverse direction. High values often correspond to downloaded payloads or web attack responses.",
    "Flow Bytes/s": "Network transfer rate. Anomalously high values signify volumetric DoS attacks; unusually low values with long duration signify slow-and-low attacks (Slowloris).",
    "Flow Packets/s": "Flow transmission frequency. Spikes correlate strongly with scanning and volumetric flooding.",
    "Flow IAT Mean": "Mean Inter-Arrival Time between consecutive packets. Automated scripts exhibit rigid, low IAT variance compared to human traffic.",
    "Fwd IAT Total": "Total inter-arrival time for forward packets. High duration with low packet count reflects keep-alive or slow DoS probing.",
    "SYN Flag Count": "Count of SYN packets initiating TCP handshakes. High count without corresponding ACK flags indicates SYN flood or port scanning.",
    "FIN Flag Count": "Count of FIN packets terminating connections. Abrupt FIN sweeps are common in reconnaissance tools like Nmap.",
    "RST Flag Count": "Count of RST packets indicating connection resets. Common in port scans encountering closed ports.",
    "PSH Flag Count": "Count of Push flags forcing immediate transmission without buffering. Exploits frequently set PSH to trigger immediate packet inspection.",
    "ACK Flag Count": "Count of Acknowledgement packets. Essential for measuring normal TCP handshake completion.",
    "Init_Win_bytes_forward": "Initial TCP window advertisement in forward flow. Distinct OS-level TCP/IP stack fingerprinting signature.",
    "Init_Win_bytes_backward": "Initial TCP window advertisement in reverse flow.",
    "min_seg_size_forward": "Minimum segment size seen in forward direction, revealing TCP header options.",
    "Average Packet Size": "Mean packet size across the entire flow. Low average packet sizes correlate with probe packets and TCP control floods.",
}


class ModelExplainer:
    """
    Computes and formats feature importances and SHAP values for tree-based models (RF, XGBoost)
    and linear coefficients for baseline models.
    """

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def get_feature_importance(self, top_n: int = SHAP_MAX_DISPLAY) -> pd.DataFrame:
        """
        Extracts built-in feature importance from tree-based ensembles (Gini / MDI or gain)
        or absolute coefficients from logistic regression.
        """
        if hasattr(self.model, "feature_importances_"):
            raw_importances = self.model.feature_importances_
            metric_type = "Gini / Split Gain"
        elif hasattr(self.model, "coef_"):
            # Logistic regression absolute coefficient magnitudes
            raw_importances = np.abs(self.model.coef_[0])
            metric_type = "Absolute Weight Magnitude (|w|)"
        else:
            raise AttributeError(f"Model {type(self.model)} does not provide feature_importances_ or coef_.")

        # Normalize to percentage
        total = np.sum(raw_importances)
        normalized = (raw_importances / total) * 100 if total > 0 else raw_importances

        min_len = min(len(self.feature_names), len(raw_importances))
        df = pd.DataFrame({
            "Feature": self.feature_names[:min_len],
            "Importance": raw_importances[:min_len],
            "Relative_Percentage": np.round(normalized[:min_len], 2),
            "Metric_Type": metric_type,
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        df["Rank"] = df.index + 1
        df["Cybersecurity_Context"] = df["Feature"].map(
            lambda f: CYBERSECURITY_FEATURE_TAXONOMY.get(f, "Standard network flow statistical descriptor.")
        )

        return df.head(top_n)

    def compute_shap_explanations(
        self,
        X_data: np.ndarray,
        sample_size: int = SHAP_SAMPLE_SIZE,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Computes SHAP values on a stratified or representative sample of data.
        Returns serialized representation suitable for plotting and reporting.
        """
        if not SHAP_AVAILABLE:
            return {
                "status": "unavailable",
                "message": "SHAP library is not installed. Run 'pip install shap' to enable SHAP visualizer.",
                "sample_size": 0,
            }

        # Subsample for computational feasibility on standard CPU / Apple Silicon
        n_samples = len(X_data)
        if n_samples > sample_size:
            rng = np.random.RandomState(random_state)
            indices = rng.choice(n_samples, size=sample_size, replace=False)
            X_sub = X_data[indices]
        else:
            X_sub = X_data

        logger.info("Computing SHAP values on %d samples with %d features...", len(X_sub), X_sub.shape[1])

        try:
            # Tree explainer for tree models, linear for logistic regression
            if hasattr(self.model, "estimators_") or "XGB" in str(type(self.model)):
                explainer = shap.TreeExplainer(self.model)
                shap_raw = explainer.shap_values(X_sub)
                # Binary classification handling across SHAP versions
                if isinstance(shap_raw, list):
                    shap_vals = shap_raw[1] if len(shap_raw) > 1 else shap_raw[0]
                elif hasattr(shap_raw, "ndim") and shap_raw.ndim == 3:
                    shap_vals = shap_raw[:, :, 1]
                elif hasattr(shap_raw, "values"):
                    shap_vals = shap_raw.values[:, :, 1] if shap_raw.values.ndim == 3 else shap_raw.values
                else:
                    shap_vals = shap_raw
            else:
                # Kernel or Linear explainer
                background = shap.sample(X_sub, min(50, len(X_sub)))
                explainer = shap.LinearExplainer(self.model, background)
                shap_raw = explainer.shap_values(X_sub)
                if isinstance(shap_raw, list):
                    shap_vals = shap_raw[1] if len(shap_raw) > 1 else shap_raw[0]
                elif hasattr(shap_raw, "ndim") and shap_raw.ndim == 3:
                    shap_vals = shap_raw[:, :, 1]
                else:
                    shap_vals = shap_raw

            # Mean absolute SHAP per feature
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
            ranked_idx = np.argsort(mean_abs_shap)[::-1][:SHAP_MAX_DISPLAY]

            top_features = [self.feature_names[i] for i in ranked_idx]
            top_shap_magnitudes = mean_abs_shap[ranked_idx].tolist()

            return {
                "status": "success",
                "sample_size": len(X_sub),
                "shap_values": shap_vals,
                "X_sample": X_sub,
                "feature_names": self.feature_names,
                "top_features": top_features,
                "top_shap_magnitudes": [round(val, 6) for val in top_shap_magnitudes],
                "mean_abs_shap_all": mean_abs_shap,
            }

        except Exception as err:
            logger.error("SHAP computation failed: %s", err)
            return {
                "status": "error",
                "message": f"SHAP computation encountered an issue: {str(err)}",
                "sample_size": 0,
            }

    def get_shap_feature_importance(
        self,
        X_data: np.ndarray,
        sample_size: int = SHAP_SAMPLE_SIZE,
        top_n: int = SHAP_MAX_DISPLAY,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Extracts feature importance strictly using game-theoretic SHAP values rather than Gini impurity.
        Returns a ranked DataFrame matching get_feature_importance schema.
        """
        shap_res = self.compute_shap_explanations(
            X_data=X_data,
            sample_size=sample_size,
            random_state=random_state,
        )

        if shap_res["status"] != "success":
            logger.warning("SHAP calculation unavailable; falling back to Gini importance.")
            return self.get_feature_importance(top_n=top_n)

        mean_abs_shap = shap_res["mean_abs_shap_all"]
        total = np.sum(mean_abs_shap)
        normalized = (mean_abs_shap / total) * 100 if total > 0 else mean_abs_shap

        min_len = min(len(self.feature_names), len(mean_abs_shap))
        df = pd.DataFrame({
            "Feature": self.feature_names[:min_len],
            "Importance": mean_abs_shap[:min_len],
            "Relative_Percentage": np.round(normalized[:min_len], 2),
            "Metric_Type": "Mean |SHAP Value| (Game-Theoretic Attribution)",
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        df["Rank"] = df.index + 1
        df["Cybersecurity_Context"] = df["Feature"].map(
            lambda f: CYBERSECURITY_FEATURE_TAXONOMY.get(f, "Standard network flow statistical descriptor.")
        )

        return df.head(top_n)

    @staticmethod
    def causality_caveat_statement() -> str:
        """
        Academic disclaimer regarding correlation vs causality in ML security.
        """
        return (
            "Academic Methodological Note: High feature importance (Gini gain or SHAP value) identifies statistical association "
            "and predictive utility within the analyzed dataset, but does NOT constitute proof of root-cause causality. In network "
            "security, statistical features (e.g. Destination Port or Flow Packets/s) may correlate with attacks because specific "
            "tools happen to flood certain ports, rather than the port itself causing malicious behavior. Adversaries who alter packet "
            "rates or pad packet lengths can evade detectors reliant strictly on non-causal statistical artifacts."
        )
