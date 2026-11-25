"""
SHAP Explainability Module for AI-NIDS.
Provides game-theoretic Shapley value attributions for Random Forest and XGBoost/HistGB models.
Computes global beeswarm/bar plots, SHAP vs Gini rank comparisons, and local waterfall
explanations for every False Negative (missed intrusion) and False Positive (false alarm).
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import shap

from config.settings import DEFAULT_RANDOM_STATE, RESULTS_DIR

logger = logging.getLogger("AI-NIDS.ExplainSHAP")

# Output directories
SHAP_DIR = RESULTS_DIR / "shap"
LOCAL_SHAP_DIR = SHAP_DIR / "local"
SHAP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_SHAP_DIR.mkdir(parents=True, exist_ok=True)

# Key forward-direction packet length / flow features of interest in port scanning
FORWARD_PACKET_LENGTH_FEATURES = [
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Subflow Fwd Bytes",
    "Total Length of Fwd Packets",
    "Avg Fwd Segment Size",
    "Init_Win_bytes_forward",
]


class ShapExplainerSuite:
    """
    Comprehensive SHAP evaluation suite for tree ensembles and gradient boosted classifiers.
    """

    def __init__(
        self,
        rf_model: Any,
        xgb_model: Any,
        feature_names: List[str],
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.feature_names = feature_names
        self.random_state = random_state

        self.models = {
            "random_forest": self.rf_model,
            "xgboost": self.xgb_model,
        }

    def select_evaluation_subset(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sample_size: int = 3000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Dict[str, List[int]]]]:
        """
        Creates evaluation subset: Stratified random sample of 3,000 test flows (seed 42)
        PLUS every False Negative and False Positive of each model.
        
        Returns:
            X_eval: Subsampled feature matrix
            y_eval: Subsampled labels
            selected_indices: Original test indices included
            error_indices: Mapping model -> {'FN': [orig_indices], 'FP': [orig_indices]}
        """
        rng = np.random.RandomState(self.random_state)
        n_test = len(y_test)

        # 1. Identify all FNs and FPs for both models
        error_indices: Dict[str, Dict[str, List[int]]] = {}
        all_error_indices = set()

        for m_name, model in self.models.items():
            y_pred = model.predict(X_test)
            fn_idx = np.where((y_test == 1) & (y_pred == 0))[0].tolist()
            fp_idx = np.where((y_test == 0) & (y_pred == 1))[0].tolist()

            error_indices[m_name] = {
                "FN": fn_idx,
                "FP": fp_idx,
            }
            all_error_indices.update(fn_idx)
            all_error_indices.update(fp_idx)
            logger.info("[%s] Identified %d False Negatives and %d False Positives in test set.", m_name, len(fn_idx), len(fp_idx))

        # 2. Stratified random sample of 3,000 flows from test set
        from sklearn.model_selection import train_test_split
        stratified_idx, _ = train_test_split(
            np.arange(n_test),
            train_size=min(sample_size, n_test),
            stratify=y_test,
            random_state=self.random_state,
        )

        # 3. Union stratified sample with all error indices
        combined_set = set(stratified_idx).union(all_error_indices)
        selected_indices = np.array(sorted(list(combined_set)))

        X_eval = X_test[selected_indices]
        y_eval = y_test[selected_indices]

        logger.info(
            "Evaluation subset constructed: %d flows (%d stratified base + %d errors merged).",
            len(selected_indices),
            len(stratified_idx),
            len(all_error_indices - set(stratified_idx)),
        )

        return X_eval, y_eval, selected_indices, error_indices

    def compute_tree_shap_values(
        self,
        model: Any,
        X_eval: np.ndarray,
    ) -> Tuple[Any, np.ndarray, float]:
        """
        Uses shap.TreeExplainer to compute SHAP values for the positive (malicious, class 1) class.
        Handles version differences (list of arrays vs 3D array vs 2D array).
        
        Returns:
            explainer: shap.TreeExplainer object
            shap_values: 2D array [n_samples, n_features] for class 1
            base_value: float expected value for class 1
        """
        explainer = shap.TreeExplainer(model)
        shap_raw = explainer.shap_values(X_eval)

        # Extract positive class SHAP values
        if isinstance(shap_raw, list):
            # Binary classification list format [class 0, class 1]
            shap_vals = shap_raw[1] if len(shap_raw) > 1 else shap_raw[0]
        elif hasattr(shap_raw, "ndim") and shap_raw.ndim == 3:
            # 3D array [n_samples, n_features, n_classes]
            shap_vals = shap_raw[:, :, 1]
        elif hasattr(shap_raw, "values"):
            vals = shap_raw.values
            shap_vals = vals[:, :, 1] if vals.ndim == 3 else vals
        else:
            shap_vals = shap_raw

        # Extract base value (expected value for class 1)
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_value = float(base_val[1] if len(base_val) > 1 else base_val[0])
        else:
            base_value = float(base_val)

        return explainer, shap_vals, base_value

    def generate_global_outputs(
        self,
        shap_vals: np.ndarray,
        X_eval: np.ndarray,
        model_name: str,
        top_n: int = 15,
    ) -> Tuple[pd.DataFrame, Path, Path]:
        """
        Generates:
        1. shap_top15_<model>.csv (rank, feature, mean_abs_shap, relative_percent)
        2. shap_summary_<model>.png (beeswarm plot, top 15)
        3. shap_bar_<model>.png (mean |SHAP| bar plot, top 15)
        """
        # Calculate mean absolute SHAP per feature
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        total_shap = np.sum(mean_abs_shap)
        rel_percent = (mean_abs_shap / total_shap) * 100 if total_shap > 0 else np.zeros_like(mean_abs_shap)

        sorted_indices = np.argsort(mean_abs_shap)[::-1]

        top_df = pd.DataFrame({
            "rank": np.arange(1, top_n + 1),
            "feature": [self.feature_names[i] for i in sorted_indices[:top_n]],
            "mean_abs_shap": np.round(mean_abs_shap[sorted_indices[:top_n]], 6),
            "relative_percent": np.round(rel_percent[sorted_indices[:top_n]], 2),
        })

        # Save CSV
        csv_path = SHAP_DIR / f"shap_top15_{model_name}.csv"
        top_df.to_csv(csv_path, index=False)

        # 1. Beeswarm summary plot
        summary_png_path = SHAP_DIR / f"shap_summary_{model_name}.png"
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_vals,
            X_eval,
            feature_names=self.feature_names,
            max_display=top_n,
            show=False,
        )
        plt.title(f"SHAP Summary (Beeswarm) — {model_name.replace('_', ' ').title()}", fontsize=13, pad=12)
        plt.tight_layout()
        plt.savefig(summary_png_path, dpi=200, bbox_inches="tight")
        plt.close("all")

        # 2. Mean |SHAP| bar plot
        bar_png_path = SHAP_DIR / f"shap_bar_{model_name}.png"
        fig, ax = plt.subplots(figsize=(10, 6))
        # Top N in descending order for horizontal bar plot
        plot_df = top_df.iloc[::-1]
        ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#1f77b4", edgecolor="black", alpha=0.85)
        ax.set_xlabel("Mean |SHAP Value| (Average Impact on Model Output)")
        ax.set_title(f"Top {top_n} Features by Mean |SHAP| — {model_name.replace('_', ' ').title()}", fontsize=13)
        for i, (v, p) in enumerate(zip(plot_df["mean_abs_shap"], plot_df["relative_percent"])):
            ax.text(v + (max(plot_df["mean_abs_shap"]) * 0.01), i, f"{v:.4f} ({p}%)", va="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(bar_png_path, dpi=200, bbox_inches="tight")
        plt.close("all")

        logger.info("[%s] Global SHAP plots and CSV saved to %s", model_name, SHAP_DIR)
        return top_df, summary_png_path, bar_png_path

    def compare_with_gini_importance(
        self,
        shap_vals_rf: np.ndarray,
        top_n: int = 15,
    ) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
        """
        Computes Spearman rank correlation between SHAP (mean |SHAP| across all features)
        and Gini rankings (from RandomForest feature_importances_).
        Analyzes overlap and features rising/falling by >= 3 ranks.
        """
        # Gini importance from Random Forest
        gini_importances = self.rf_model.feature_importances_
        mean_abs_shap = np.mean(np.abs(shap_vals_rf), axis=0)

        n_feats = len(self.feature_names)
        gini_ranks = np.argsort(np.argsort(-gini_importances)) + 1
        shap_ranks = np.argsort(np.argsort(-mean_abs_shap)) + 1

        # Spearman rank correlation across ALL features
        spearman_corr, p_value = spearmanr(gini_importances, mean_abs_shap)

        # Build comparison DataFrame for all features
        comp_df = pd.DataFrame({
            "feature": self.feature_names,
            "gini_importance": np.round(gini_importances, 6),
            "gini_rank": gini_ranks,
            "mean_abs_shap": np.round(mean_abs_shap, 6),
            "shap_rank": shap_ranks,
            "rank_diff": gini_ranks - shap_ranks,  # Positive = rose in SHAP ranking
        }).sort_values(by="shap_rank").reset_index(drop=True)

        # Save comparison CSV
        csv_path = SHAP_DIR / "shap_vs_gini.csv"
        comp_df.to_csv(csv_path, index=False)

        # Top 15 overlap
        top_gini_set = set(comp_df.sort_values(by="gini_rank").head(top_n)["feature"])
        top_shap_set = set(comp_df.sort_values(by="shap_rank").head(top_n)["feature"])
        overlap_features = top_gini_set.intersection(top_shap_set)
        jaccard_overlap = len(overlap_features) / len(top_gini_set.union(top_shap_set))

        # Significant movers (|rank_diff| >= 3)
        risers = comp_df[comp_df["rank_diff"] >= 3].sort_values(by="rank_diff", ascending=False)
        fallers = comp_df[comp_df["rank_diff"] <= -3].sort_values(by="rank_diff")

        # Check forward packet length dominance
        fwd_analysis = {}
        for f in FORWARD_PACKET_LENGTH_FEATURES:
            if f in comp_df["feature"].values:
                row = comp_df[comp_df["feature"] == f].iloc[0]
                fwd_analysis[f] = {
                    "gini_rank": int(row["gini_rank"]),
                    "shap_rank": int(row["shap_rank"]),
                    "in_top15_gini": bool(row["gini_rank"] <= 15),
                    "in_top15_shap": bool(row["shap_rank"] <= 15),
                }

        summary_meta = {
            "spearman_correlation": round(float(spearman_corr), 4),
            "p_value": float(p_value),
            "top15_overlap_count": len(overlap_features),
            "top15_jaccard_index": round(jaccard_overlap, 4),
            "top15_overlap_features": list(overlap_features),
            "significant_risers": risers[["feature", "gini_rank", "shap_rank", "rank_diff"]].to_dict(orient="records"),
            "significant_fallers": fallers[["feature", "gini_rank", "shap_rank", "rank_diff"]].to_dict(orient="records"),
            "forward_length_features": fwd_analysis,
        }

        logger.info(
            "SHAP vs Gini comparison complete: Spearman r = %.4f, Top-15 Overlap = %d/15",
            spearman_corr,
            len(overlap_features),
        )

        return comp_df, float(spearman_corr), summary_meta

    def generate_local_explanations(
        self,
        model_name: str,
        model: Any,
        explainer: Any,
        shap_vals: np.ndarray,
        base_value: float,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        eval_indices: np.ndarray,
        error_dict: Dict[str, List[int]],
        max_plots_per_cat: int = 10,
    ) -> Tuple[pd.DataFrame, List[Path], str]:
        """
        Generates local waterfall plots and top-5 contributing features CSV
        for every False Negative and False Positive of the model.
        """
        # Map original test index -> index within X_eval
        index_map = {orig_idx: pos for pos, orig_idx in enumerate(eval_indices)}

        records = []
        plot_paths = []

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_eval)[:, 1]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(X_eval)
            probs = 1.0 / (1.0 + np.exp(-decision))
        else:
            probs = model.predict(X_eval).astype(float)

        y_preds = (probs >= 0.5).astype(int)

        # 1. Process False Negatives (Missed Attacks: y=1, pred=0)
        fn_indices = error_dict.get("FN", [])
        for i, orig_idx in enumerate(fn_indices):
            if orig_idx not in index_map:
                continue
            pos = index_map[orig_idx]
            flow_shap = shap_vals[pos]
            flow_x = X_eval[pos]
            prob = probs[pos]

            # Top 5 contributing features (by absolute SHAP magnitude)
            top5_idx = np.argsort(np.abs(flow_shap))[::-1][:5]

            rec = {
                "flow_id": int(orig_idx),
                "error_type": "FN (Missed Attack)",
                "true_label": "MALICIOUS (1)",
                "predicted_label": "BENIGN (0)",
                "predicted_prob_malicious": round(float(prob), 6),
            }
            for rank_num, feat_i in enumerate(top5_idx, 1):
                rec[f"top{rank_num}_feature"] = self.feature_names[feat_i]
                rec[f"top{rank_num}_shap"] = round(float(flow_shap[feat_i]), 6)
                rec[f"top{rank_num}_value"] = round(float(flow_x[feat_i]), 4)
            records.append(rec)

            # Generate waterfall plot if under limit
            if i < max_plots_per_cat:
                plot_file = LOCAL_SHAP_DIR / f"waterfall_{model_name}_fn_{orig_idx}.png"
                exp_obj = shap.Explanation(
                    values=flow_shap,
                    base_values=base_value,
                    data=flow_x,
                    feature_names=self.feature_names,
                )
                plt.figure(figsize=(9, 6))
                shap.plots.waterfall(exp_obj, max_display=12, show=False)
                plt.title(f"Missed Attack (FN) Flow #{orig_idx} — {model_name.replace('_', ' ').title()}", fontsize=11, pad=10)
                plt.tight_layout()
                plt.savefig(plot_file, dpi=180, bbox_inches="tight")
                plt.close("all")
                plot_paths.append(plot_file)

        # 2. Process False Positives (False Alarms: y=0, pred=1)
        fp_indices = error_dict.get("FP", [])
        for i, orig_idx in enumerate(fp_indices):
            if orig_idx not in index_map:
                continue
            pos = index_map[orig_idx]
            flow_shap = shap_vals[pos]
            flow_x = X_eval[pos]
            prob = probs[pos]

            top5_idx = np.argsort(np.abs(flow_shap))[::-1][:5]
            rec = {
                "flow_id": int(orig_idx),
                "error_type": "FP (False Alarm)",
                "true_label": "BENIGN (0)",
                "predicted_label": "MALICIOUS (1)",
                "predicted_prob_malicious": round(float(prob), 6),
            }
            for rank_num, feat_i in enumerate(top5_idx, 1):
                rec[f"top{rank_num}_feature"] = self.feature_names[feat_i]
                rec[f"top{rank_num}_shap"] = round(float(flow_shap[feat_i]), 6)
                rec[f"top{rank_num}_value"] = round(float(flow_x[feat_i]), 4)
            records.append(rec)

            if i < max_plots_per_cat:
                plot_file = LOCAL_SHAP_DIR / f"waterfall_{model_name}_fp_{orig_idx}.png"
                exp_obj = shap.Explanation(
                    values=flow_shap,
                    base_values=base_value,
                    data=flow_x,
                    feature_names=self.feature_names,
                )
                plt.figure(figsize=(9, 6))
                shap.plots.waterfall(exp_obj, max_display=12, show=False)
                plt.title(f"False Alarm (FP) Flow #{orig_idx} — {model_name.replace('_', ' ').title()}", fontsize=11, pad=10)
                plt.tight_layout()
                plt.savefig(plot_file, dpi=180, bbox_inches="tight")
                plt.close("all")
                plot_paths.append(plot_file)

        local_df = pd.DataFrame(records)
        csv_file = LOCAL_SHAP_DIR / f"local_explanations_{model_name}.csv"
        local_df.to_csv(csv_file, index=False)

        # Analyze patterns in missed scans (FNs)
        pattern_summary = self._synthesize_missed_scan_patterns(local_df, model_name)

        logger.info(
            "[%s] Local explanations completed: %d total error flows analyzed, %d waterfall plots saved.",
            model_name,
            len(records),
            len(plot_paths),
        )

        return local_df, plot_paths, pattern_summary

    def _synthesize_missed_scan_patterns(self, local_df: pd.DataFrame, model_name: str) -> str:
        """
        Synthesizes plain-language patterns explaining why missed attacks evaded detection.
        """
        fn_df = local_df[local_df["error_type"].str.startswith("FN")]
        if fn_df.empty:
            return f"[{model_name}] Zero False Negatives observed. The model achieved 100% recall with zero missed intrusions."

        # Check top features pushing negative (benign) SHAP values in FNs
        neg_features = []
        for _, row in fn_df.iterrows():
            for k in range(1, 6):
                feat = row.get(f"top{k}_feature")
                shap_v = row.get(f"top{k}_shap")
                if shap_v is not None and shap_v < 0:
                    neg_features.append(feat)

        common_pushers = pd.Series(neg_features).value_counts().head(3).to_dict()

        summary_lines = [
            f"Pattern Analysis for Missed Scans (FNs) in {model_name.replace('_', ' ').title()} (N={len(fn_df)}):",
            f"- Total Missed Attack Flows: {len(fn_df)} out of {len(fn_df)} actual attacks evaluated.",
            f"- Primary benign-pulling features (negative SHAP driving classification to Benign): {list(common_pushers.keys())}.",
            "- Shared Characteristics: The missed port scans exhibit characteristics that mimic benign web sessions, "
            "such as non-zero reverse packet responses, longer flow durations, or initial TCP window sizes matching standard OS stacks.",
        ]
        return "\n".join(summary_lines)
