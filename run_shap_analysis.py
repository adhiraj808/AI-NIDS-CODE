#!/usr/bin/env python3
"""
CLI Runner Script for AI-NIDS SHAP Explainability Analysis.
Performs:
1. Pipeline sanity check & exact replication verification
2. Global SHAP analysis for Random Forest and Gradient Boosting
3. SHAP vs. Gini importance rank comparison (Spearman correlation & movers)
4. Local waterfall explanations for every False Negative (missed scan) & False Positive
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE, RAW_DATA_DIR
from src.data_loader import DataLoader
from src.evaluation import calculate_comprehensive_metrics
from src.explain_shap import ShapExplainerSuite, SHAP_DIR, LOCAL_SHAP_DIR
from src.model_trainer import ModelTrainer, XGBOOST_AVAILABLE
from src.preprocessing import PreprocessingPipeline


def run_sanity_check(X_train, X_test, y_train, y_test, trainer):
    """
    Step 1: Sanity Check Verification.
    Verifies model classes, hyperparameters, confusion matrices, and metrics alignment.
    """
    print("\n" + "=" * 80)
    print("STEP 1: PIPELINE SANITY CHECK & MODEL VERIFICATION")
    print("=" * 80)

    expected_xgboost_acc = 0.999883
    expected_xgboost_fnr = 0.000220

    models_info = {}

    for m_key in ["logistic_regression", "random_forest", "xgboost"]:
        model, t_train = trainer.train_model(m_key, X_train, y_train, handle_imbalance=True)
        y_pred, y_proba, t_pred = trainer.predict(m_key, X_test)
        metrics = calculate_comprehensive_metrics(y_test, y_pred, y_proba, t_train, t_pred, model_name=m_key)

        model_class = type(model)
        params = model.get_params()

        print(f"\n--- Model Key: '{m_key}' ---")
        print(f"  Exact Model Class: {model_class.__module__}.{model_class.__name__}")
        print(f"  Key Hyperparameters:")
        for k in ["max_iter", "n_estimators", "max_depth", "learning_rate", "class_weight", "solver", "C", "random_state"]:
            if k in params:
                print(f"    - {k}: {params[k]}")

        print(f"  Confusion Matrix on Test Partition (N={len(y_test):,}):")
        print(f"    - True Negatives  (TN): {metrics['true_negatives']:,}")
        print(f"    - False Positives (FP): {metrics['false_positives']:,} (FPR: {metrics['false_positive_rate']:.6f})")
        print(f"    - False Negatives (FN): {metrics['false_negatives']:,} (FNR: {metrics['false_negative_rate']:.6f})")
        print(f"    - True Positives  (TP): {metrics['true_positives']:,}")
        print(f"  Key Rates:")
        print(f"    - Accuracy:  {metrics['accuracy']:.6f}")
        print(f"    - Precision: {metrics['precision']:.6f}")
        print(f"    - Recall:    {metrics['recall']:.6f}")
        print(f"    - F1-Score:  {metrics['f1_score']:.6f}")
        print(f"    - ROC-AUC:   {metrics['roc_auc']:.6f}")

        models_info[m_key] = {
            "model": model,
            "metrics": metrics,
            "class": model_class,
        }

    # Verify Gradient Boosting engine identity
    xgb_class_name = type(models_info["xgboost"]["model"]).__name__
    print("\n" + "-" * 80)
    print("GRADIENT BOOSTING ENGINE CLARIFICATION:")
    if "HistGradientBoostingClassifier" in xgb_class_name:
        print(f"  * Model C Engine: sklearn.ensemble.HistGradientBoostingClassifier")
        print("  * Reason: XGBoost C-library (libxgboost.dylib) requires OpenMP (libomp.dylib).")
        print("    The system transparently utilized scikit-learn's optimized HistGradientBoostingClassifier fallback.")
    else:
        print(f"  * Model C Engine: native XGBoost (xgboost.XGBClassifier)")

    print("\nGINI IMPORTANCE ATTRIBUTION CLARIFICATION:")
    print("  * Existing Gini importance ranking (top 15) strictly derives from: Model B (RandomForestClassifier).")
    print("  * Reason: Random Forest natively calculates Gini/MDI (Mean Decrease in Impurity) split gain")
    print("    via `feature_importances_`, whereas HistGB does not expose split Gini gain directly.")
    print("-" * 80)

    # Verification of matching metrics
    actual_xgb_acc = models_info["xgboost"]["metrics"]["accuracy"]
    actual_xgb_fnr = models_info["xgboost"]["metrics"]["false_negative_rate"]

    if abs(actual_xgb_acc - expected_xgboost_acc) > 1e-5 or abs(actual_xgb_fnr - expected_xgboost_fnr) > 1e-5:
        print(f"ERROR: Sanity check metrics mismatch! Expected Acc={expected_xgboost_acc}, got {actual_xgb_acc}")
        sys.exit(1)
    else:
        print("✅ Sanity check passed! Metrics match exact existing results (Acc=0.999883, FNR=0.000220).")

    return models_info


def main():
    start_time = time.time()
    print("=" * 80)
    print("AI-NIDS: SHAP GAME-THEORETIC EXPLAINABILITY PIPELINE")
    print("=" * 80)

    dataset_path = RAW_DATA_DIR / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)

    # 1. Ingest dataset
    print(f"\n[1/5] Ingesting dataset: {dataset_path.name}...")
    loader = DataLoader()
    raw_df = loader.load_csv(dataset_path)
    df_bin, _ = loader.binarize_labels(raw_df)

    # 2. Preprocess dataset (strictly on training split)
    print(f"[2/5] Running preprocessing pipeline (Stratified 80/20, seed={DEFAULT_RANDOM_STATE})...")
    pipeline = PreprocessingPipeline(test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE)
    X_train, X_test, y_train, y_test, feature_names = pipeline.fit_transform(df_bin)
    print(f"      Train Partition: {X_train.shape[0]:,} flows | Test Partition: {X_test.shape[0]:,} flows")
    print(f"      Features: {len(feature_names)} statistical network flow descriptors.")

    # 3. Train models & run sanity check
    trainer = ModelTrainer(random_state=DEFAULT_RANDOM_STATE)
    models_info = run_sanity_check(X_train, X_test, y_train, y_test, trainer)

    # 4. Initialize SHAP Explainer Suite
    suite = ShapExplainerSuite(
        rf_model=models_info["random_forest"]["model"],
        xgb_model=models_info["xgboost"]["model"],
        feature_names=feature_names,
        random_state=DEFAULT_RANDOM_STATE,
    )

    # 5. Select Evaluation Subset (Stratified 3,000 + All FNs & FPs)
    print("\n" + "=" * 80)
    print("STEP 2: COMPUTING SHAP ATTRIBUTIONS ON HELD-OUT TEST DATA")
    print("=" * 80)
    print("Constructing stratified evaluation subset (3,000 test flows + all model error flows)...")
    X_eval, y_eval, eval_indices, error_indices = suite.select_evaluation_subset(X_test, y_test, sample_size=3000)
    print(f"Evaluation set size: {len(X_eval):,} flows.")

    # Compute SHAP for both models
    shap_results = {}
    for m_name in ["random_forest", "xgboost"]:
        print(f"\nComputing TreeExplainer SHAP values for '{m_name}'...")
        t0 = time.time()
        explainer, shap_vals, base_val = suite.compute_tree_shap_values(models_info[m_name]["model"], X_eval)
        t_elapsed = time.time() - t0
        print(f"  ↳ Finished in {t_elapsed:.2f} seconds. Base value (log-odds / expected output): {base_val:.4f}")
        print(f"  ↳ SHAP values shape: {shap_vals.shape} (strictly explaining positive/malicious class)")

        shap_results[m_name] = {
            "explainer": explainer,
            "shap_vals": shap_vals,
            "base_val": base_val,
        }

    # 6. Global Outputs & Comparison
    print("\n" + "=" * 80)
    print("STEP 3: GLOBAL SHAP RANKINGS & TOP-15 FEATURE TABLES")
    print("=" * 80)

    generated_files = []

    for m_name in ["random_forest", "xgboost"]:
        top_df, summary_png, bar_png = suite.generate_global_outputs(
            shap_vals=shap_results[m_name]["shap_vals"],
            X_eval=X_eval,
            model_name=m_name,
            top_n=15,
        )
        generated_files.extend([
            SHAP_DIR / f"shap_top15_{m_name}.csv",
            summary_png,
            bar_png,
        ])

        print(f"\n--- TOP 15 INFLUENTIAL FEATURES: {m_name.upper().replace('_', ' ')} (SHAP) ---")
        print(top_df.to_string(index=False))

    # 7. Comparison with Gini Importance
    print("\n" + "=" * 80)
    print("STEP 4: SHAP VS. GINI IMPORTANCE COMPARISON")
    print("=" * 80)
    comp_df, spearman_corr, meta = suite.compare_with_gini_importance(
        shap_vals_rf=shap_results["random_forest"]["shap_vals"],
        top_n=15,
    )
    generated_files.append(SHAP_DIR / "shap_vs_gini.csv")

    print(f"Spearman Rank Correlation (All {len(feature_names)} features): {spearman_corr:.4f} (p-value: {meta['p_value']:.2e})")
    print(f"Top-15 Feature Set Overlap: {meta['top15_overlap_count']}/15 features (Jaccard similarity: {meta['top15_jaccard_index']:.4f})")
    print(f"Shared Top-15 Features:\n  {meta['top15_overlap_features']}")

    print("\nSignificant Risers in SHAP Ranking (Rose by >= 3 ranks compared to Gini):")
    risers_df = pd.DataFrame(meta["significant_risers"]).head(8)
    if not risers_df.empty:
        print(risers_df.to_string(index=False))
    else:
        print("  None")

    print("\nSignificant Fallers in SHAP Ranking (Fell by >= 3 ranks compared to Gini):")
    fallers_df = pd.DataFrame(meta["significant_fallers"]).head(8)
    if not fallers_df.empty:
        print(fallers_df.to_string(index=False))
    else:
        print("  None")

    print("\nStatus of Forward-Direction Packet Length / Flow Features:")
    for feat, status in meta["forward_length_features"].items():
        print(f"  * {feat:<30} | Gini Rank: {status['gini_rank']:2d} | SHAP Rank: {status['shap_rank']:2d} | Dominant in SHAP: {status['in_top15_shap']}")

    # 8. Local Explanations for Error Flows
    print("\n" + "=" * 80)
    print("STEP 5: LOCAL WATERFALL EXPLANATIONS FOR FALSE NEGATIVES & FALSE POSITIVES")
    print("=" * 80)

    for m_name in ["random_forest", "xgboost"]:
        local_df, plot_paths, pattern_summary = suite.generate_local_explanations(
            model_name=m_name,
            model=models_info[m_name]["model"],
            explainer=shap_results[m_name]["explainer"],
            shap_vals=shap_results[m_name]["shap_vals"],
            base_value=shap_results[m_name]["base_val"],
            X_eval=X_eval,
            y_eval=y_eval,
            eval_indices=eval_indices,
            error_dict=error_indices[m_name],
            max_plots_per_cat=10,
        )
        generated_files.append(LOCAL_SHAP_DIR / f"local_explanations_{m_name}.csv")
        generated_files.extend(plot_paths)

        print(f"\n--- {m_name.upper().replace('_', ' ')} ERROR ANALYSIS ---")
        print(pattern_summary)
        print(f"Saved {len(plot_paths)} waterfall plots to {LOCAL_SHAP_DIR}/")

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"SHAP ANALYSIS COMPLETE! Total elapsed wall-clock runtime: {total_time:.2f} seconds.")
    print("=" * 80)

    print("\nList of All Generated Files:")
    for f in generated_files:
        print(f"  - {f.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
