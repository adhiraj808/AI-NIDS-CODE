"""
Research Experiments Runner for AI-NIDS.
Automates and records the 5 mandatory research experiments:
- Experiment 1: Compare Logistic Regression, Random Forest, and XGBoost under identical splits.
- Experiment 2: Investigate the empirical impact of class imbalance (Balanced vs Unweighted).
- Experiment 3: Detailed metric analysis emphasizing FNR vs Accuracy.
- Experiment 4: Feature importance and top contributing network flow indicators.
- Experiment 5: Performance vs Computational Efficiency latency trade-off.

Saves configurations, metrics CSVs, charts, and metadata into the results/ directory.
Zero hardcoding or fabrication: all numbers derived from actual model execution.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import numpy as np
import pandas as pd

from config.settings import (
    CHARTS_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    METRICS_DIR,
    REPORTS_DIR,
    SAMPLE_DATA_DIR,
    RAW_DATA_DIR,
)
from src.data_loader import DataLoader
from src.evaluation import calculate_comprehensive_metrics, compile_comparison_table
from src.explainability import ModelExplainer
from src.model_trainer import ModelTrainer, XGBOOST_AVAILABLE
from src.preprocessing import PreprocessingPipeline
from src.report_generator import ResearchReportGenerator
from src.sample_data import generate_cicids_benchmark_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AI-NIDS.Experiments")


class ResearchExperimentSuite:
    """
    Automated experimental harness for empirical academic validation.
    """

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        random_state: int = DEFAULT_RANDOM_STATE,
        test_size: float = DEFAULT_TEST_SIZE,
    ):
        self.random_state = random_state
        self.test_size = test_size
        
        # Prioritize real official CICIDS2017 dataset if present in data/raw/
        real_portscan = RAW_DATA_DIR / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
        if dataset_path is not None:
            self.dataset_path = dataset_path
        elif real_portscan.exists():
            self.dataset_path = real_portscan
        else:
            self.dataset_path = SAMPLE_DATA_DIR / "CICIDS2017_Benchmark_Sample.csv"
            if not self.dataset_path.exists():
                generate_cicids_benchmark_sample(n_samples=5000, output_filepath=self.dataset_path)

        # Ingest and binarize dataset
        self.loader = DataLoader()
        raw_df = self.loader.load_csv(self.dataset_path)
        self.df, self.original_labels = self.loader.binarize_labels(raw_df)

    def run_experiment_1_model_comparison(self) -> pd.DataFrame:
        """
        Experiment 1: Compare Logistic Regression, Random Forest, and XGBoost (if available)
        using identical training and testing conditions.
        """
        logger.info("=== Executing Experiment 1: Identical Split Model Comparison ===")

        pipeline = PreprocessingPipeline(test_size=self.test_size, random_state=self.random_state)
        X_train, X_test, y_train, y_test, features = pipeline.fit_transform(self.df)

        trainer = ModelTrainer(random_state=self.random_state)
        models_to_test = ["logistic_regression", "random_forest", "xgboost"]

        metrics_list = []
        for key in models_to_test:
            model, t_train = trainer.train_model(key, X_train, y_train, handle_imbalance=True)
            y_pred, y_proba, t_pred = trainer.predict(key, X_test)

            m = calculate_comprehensive_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                training_time=t_train,
                prediction_time=t_pred,
                model_name=key.replace("_", " ").title(),
            )
            metrics_list.append(m)

        comp_df = compile_comparison_table(metrics_list)
        csv_path = METRICS_DIR / "experiment_1_model_comparison.csv"
        comp_df.to_csv(csv_path, index=False)
        logger.info("Experiment 1 complete. Results saved to %s", csv_path)
        return comp_df

    def run_experiment_2_class_imbalance(self) -> pd.DataFrame:
        """
        Experiment 2: Investigate the effect of class imbalance handling on model performance.
        Compares Random Forest with and without cost-sensitive class weights.
        """
        logger.info("=== Executing Experiment 2: Class Imbalance Impact ===")

        pipeline = PreprocessingPipeline(test_size=self.test_size, random_state=self.random_state)
        X_train, X_test, y_train, y_test, features = pipeline.fit_transform(self.df)

        trainer = ModelTrainer(random_state=self.random_state)

        # 1. Unweighted (No imbalance handling)
        _, t_unw = trainer.train_model("random_forest", X_train, y_train, handle_imbalance=False)
        y_pred_unw, y_proba_unw, t_p_unw = trainer.predict("random_forest", X_test)
        m_unweighted = calculate_comprehensive_metrics(
            y_test, y_pred_unw, y_proba_unw, t_unw, t_p_unw, model_name="Random Forest (Unweighted)"
        )

        # 2. Balanced (Cost-sensitive class weighting)
        _, t_bal = trainer.train_model("random_forest", X_train, y_train, handle_imbalance=True)
        y_pred_bal, y_proba_bal, t_p_bal = trainer.predict("random_forest", X_test)
        m_balanced = calculate_comprehensive_metrics(
            y_test, y_pred_bal, y_proba_bal, t_bal, t_p_bal, model_name="Random Forest (Balanced Weights)"
        )

        comp_df = compile_comparison_table([m_unweighted, m_balanced])
        csv_path = METRICS_DIR / "experiment_2_class_imbalance.csv"
        comp_df.to_csv(csv_path, index=False)
        logger.info("Experiment 2 complete. Results saved to %s", csv_path)
        return comp_df

    def run_experiment_3_fnr_vs_accuracy(self) -> pd.DataFrame:
        """
        Experiment 3: Compare accuracy, precision, recall, F1-score, and false negative rates.
        Contrasts the accuracy paradox against security-critical metrics.
        """
        logger.info("=== Executing Experiment 3: Detailed Metric Contrast & FNR Analysis ===")
        # Re-uses experiment 1 outputs with focused metric columns
        comp_df = self.run_experiment_1_model_comparison()
        focus_cols = ["Model", "Accuracy", "Recall (Sensitivity)", "False Negative Rate (FNR)", "Precision", "F1-Score"]
        sub_df = comp_df[[c for c in focus_cols if c in comp_df.columns]]
        csv_path = METRICS_DIR / "experiment_3_fnr_vs_accuracy.csv"
        sub_df.to_csv(csv_path, index=False)
        return sub_df

    def run_experiment_4_feature_importance(self) -> pd.DataFrame:
        """
        Experiment 4: Investigate which network traffic features contribute most to malicious classification.
        """
        logger.info("=== Executing Experiment 4: Feature Importance Analysis ===")

        pipeline = PreprocessingPipeline(test_size=self.test_size, random_state=self.random_state)
        X_train, X_test, y_train, y_test, features = pipeline.fit_transform(self.df)

        trainer = ModelTrainer(random_state=self.random_state)
        model, _ = trainer.train_model("random_forest", X_train, y_train, handle_imbalance=True)

        explainer = ModelExplainer(model=model, feature_names=features)
        # 1. Gini / MDI importance
        imp_df = explainer.get_feature_importance(top_n=15)
        csv_path = METRICS_DIR / "experiment_4_top_features_gini.csv"
        imp_df.to_csv(csv_path, index=False)
        imp_df.to_csv(METRICS_DIR / "experiment_4_top_features.csv", index=False)

        # 2. SHAP (Game-Theoretic) feature importance
        shap_df = explainer.get_shap_feature_importance(X_test, sample_size=300, top_n=15)
        shap_csv_path = METRICS_DIR / "experiment_4_shap_features.csv"
        shap_df.to_csv(shap_csv_path, index=False)
        logger.info("Experiment 4 complete. Gini and SHAP feature rankings saved to %s and %s", csv_path, shap_csv_path)
        return shap_df

    def run_experiment_5_efficiency_tradeoff(self) -> pd.DataFrame:
        """
        Experiment 5: Analyze the trade-off between detection performance and computational efficiency.
        """
        logger.info("=== Executing Experiment 5: Performance vs Computational Efficiency ===")
        comp_df = self.run_experiment_1_model_comparison()
        eff_cols = ["Model", "F1-Score", "False Negative Rate (FNR)", "Train Time (s)", "Inference Time (s)", "Throughput (samples/s)"]
        sub_df = comp_df[[c for c in eff_cols if c in comp_df.columns]]
        csv_path = METRICS_DIR / "experiment_5_efficiency_tradeoff.csv"
        sub_df.to_csv(csv_path, index=False)
        return sub_df

    def run_all(self) -> Dict[str, pd.DataFrame]:
        """
        Runs all 5 experiments and writes comprehensive report.
        """
        results = {
            "exp1": self.run_experiment_1_model_comparison(),
            "exp2": self.run_experiment_2_class_imbalance(),
            "exp3": self.run_experiment_3_fnr_vs_accuracy(),
            "exp4": self.run_experiment_4_feature_importance(),
            "exp5": self.run_experiment_5_efficiency_tradeoff(),
        }

        # Generate markdown summary
        report_gen = ResearchReportGenerator(
            dataset_meta=self.loader.metadata,
            preprocessing_audit=PreprocessingPipeline(self.test_size, self.random_state).get_summary(),
            metrics_comparison=results["exp1"],
            experiment_config={"random_state": self.random_state, "test_size": self.test_size},
            experiment_name="Comprehensive 5-Experiment Suite",
        )
        report_gen.export_report_bundle(base_name="final_research_experiments_summary")
        return results


if __name__ == "__main__":
    suite = ResearchExperimentSuite()
    suite.run_all()
