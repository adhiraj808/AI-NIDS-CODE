"""
Model Training Page for AI-NIDS.
Configures hyperparameter splits, triggers training on held-out partitions,
records empirical latencies, and saves model artifacts via joblib.
"""

from datetime import datetime
import streamlit as st
import pandas as pd

from config.settings import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE, MODEL_CONFIGS
from src.evaluation import calculate_comprehensive_metrics
from src.model_trainer import ModelTrainer, XGBOOST_AVAILABLE
from src.preprocessing import PreprocessingPipeline


def render_model_training_page():
    st.title("⚙️ Model Training & Experiment Execution")
    st.markdown(
        "Train baseline, ensemble, and gradient-boosted classifiers on strictly partitioned training sets."
    )
    st.markdown("---")

    df = st.session_state.get("dataset", None)
    if df is None:
        st.warning("⚠️ No dataset loaded. Please navigate to **Dataset Analysis** to load a dataset before training.")
        return

    # Configuration Form
    st.subheader("1. Experimental Configuration")
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

    with col_cfg1:
        test_size = st.slider("Held-Out Test Size (Stratified)", min_value=0.10, max_value=0.40, value=DEFAULT_TEST_SIZE, step=0.05)
    with col_cfg2:
        random_seed = st.number_input("Reproducible Random Seed", value=DEFAULT_RANDOM_STATE, step=1)
    with col_cfg3:
        handle_imbalance = st.checkbox("Handle Class Imbalance (Cost-sensitive weighting)", value=True, help="Applies class_weight='balanced' to compensate for majority benign traffic")

    st.subheader("2. Select Algorithms for Comparison")
    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        train_lr = st.checkbox("Model A: Logistic Regression", value=True, help="Baseline linear classification model")
    with col_m2:
        train_rf = st.checkbox("Model B: Random Forest", value=True, help="Non-linear tree ensemble classifier")
    with col_m3:
        xgb_label = "Model C: XGBoost Classifier" if XGBOOST_AVAILABLE else "Model C: Gradient Boosted Trees (HistGB)"
        train_xgb = st.checkbox(xgb_label, value=True, help="Gradient-boosted decision trees classifier")

    st.markdown("---")

    if st.button("🚀 Launch Training & Evaluation Pipeline", key="btn_train"):
        selected_models = []
        if train_lr:
            selected_models.append("logistic_regression")
        if train_rf:
            selected_models.append("random_forest")
        if train_xgb:
            selected_models.append("xgboost")

        if not selected_models:
            st.error("Please select at least one algorithm to train.")
            return

        status_box = st.status("Training models in sequence...", expanded=True)

        # Step 1: Preprocessing with leakage prevention
        status_box.write("Executing Preprocessing Pipeline (Median Imputation & StandardScaler fit on Train only)...")
        pipeline = PreprocessingPipeline(
            test_size=test_size,
            random_state=random_seed,
            apply_scaling=True,
        )

        try:
            X_train, X_test, y_train, y_test, feature_names = pipeline.fit_transform(df)
            status_box.write(
                f"✅ Partition complete: Train={len(y_train):,} samples, Test={len(y_test):,} samples, Features={len(feature_names)}."
            )
        except Exception as err:
            status_box.update(label="Preprocessing failed!", state="error")
            st.error(f"Preprocessing error: {err}")
            return

        trainer = ModelTrainer(random_state=random_seed)
        metrics_results = []
        models_predictions = {}

        # Step 2: Train and evaluate each selected model
        for model_key in selected_models:
            display_name = MODEL_CONFIGS[model_key]["display_name"]
            status_box.write(f"⏳ Fitting {display_name}...")

            try:
                model, t_train = trainer.train_model(
                    model_key=model_key,
                    X_train=X_train,
                    y_train=y_train,
                    handle_imbalance=handle_imbalance,
                )
                status_box.write(f"   ↳ {display_name} trained in {t_train:.3f}s. Evaluating on test set...")

                y_pred, y_proba, t_pred = trainer.predict(model_key, X_test)
                metrics = calculate_comprehensive_metrics(
                    y_true=y_test,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    training_time=t_train,
                    prediction_time=t_pred,
                    model_name=display_name,
                )
                metrics_results.append(metrics)
                models_predictions[display_name] = (y_test, y_proba)

                # Persist trained model
                trainer.save_model(model_key)
                status_box.write(f"   ↳ Saved {model_key}.joblib to models/trained/")

            except Exception as err:
                status_box.write(f"❌ Error training {display_name}: {err}")

        status_box.update(label="Training and Evaluation Completed!", state="complete")

        # Save to session state
        st.session_state["pipeline"] = pipeline
        st.session_state["trainer"] = trainer
        st.session_state["metrics_history"] = metrics_results
        st.session_state["models_predictions"] = models_predictions
        st.session_state["feature_names"] = feature_names
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["latest_experiment"] = {
            "name": "Standard Comparative Benchmark",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models": [MODEL_CONFIGS[k]["display_name"] for k in selected_models],
            "test_size": test_size,
            "random_state": random_seed,
            "handle_imbalance": handle_imbalance,
        }

        st.success("🎉 Experiments completed successfully! Results are now visible in **Model Comparison**.")

    # Show existing trained models in session if available
    if "metrics_history" in st.session_state:
        st.markdown("---")
        st.subheader("3. Current Session Training Summary")
        from src.evaluation import compile_comparison_table
        summary_table = compile_comparison_table(st.session_state["metrics_history"])
        st.dataframe(summary_table, use_container_width=True)
