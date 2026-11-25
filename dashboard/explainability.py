"""
Explainability Page for AI-NIDS.
Visualizes top influential network traffic features, computes SHAP values on representative samples,
and provides domain-grounded cybersecurity context distinguishing statistical importance from causality.
"""

import streamlit as st
import pandas as pd
import numpy as np

from config.settings import SHAP_MAX_DISPLAY, SHAP_SAMPLE_SIZE
from src.explainability import (
    CYBERSECURITY_FEATURE_TAXONOMY,
    ModelExplainer,
    SHAP_AVAILABLE,
)


def render_explainability_page():
    st.title("🔬 Explainable AI (XAI) & Threat Attribution")
    st.markdown(
        "Interrogate black-box decisions, inspect top predictive network flow features, and evaluate SHAP values."
    )
    st.markdown("---")

    trainer = st.session_state.get("trainer", None)
    feature_names = st.session_state.get("feature_names", None)
    X_test = st.session_state.get("X_test", None)

    if trainer is None or not trainer.trained_models or feature_names is None:
        st.warning("⚠️ No trained models available. Please train models in **Model Training** first.")
        return

    # Model selector
    trained_keys = list(trainer.trained_models.keys())
    selected_model_key = st.selectbox(
        "Select Model to Explain:",
        trained_keys,
        format_func=lambda k: trainer.model_configs.get(k, {}).get("display_name", k),
    )

    model = trainer.trained_models[selected_model_key]
    explainer = ModelExplainer(model=model, feature_names=feature_names)

    # 1. Feature Importance Ranking (SHAP vs Gini selection)
    st.subheader(f"1. Top {SHAP_MAX_DISPLAY} Most Influential Features")
    
    col_opt1, col_opt2 = st.columns([1, 1])
    with col_opt1:
        xai_method = st.radio(
            "Attribution Methodology:",
            ["SHAP (Game-Theoretic Shapley Values)", "Gini / Split Gain (Tree-Based MDI)"],
            index=0,
            help="SHAP computes fair marginal contribution of each feature; Gini measures mean reduction in node impurity.",
        )
    with col_opt2:
        shap_n_samples = st.slider(
            "SHAP Background Sample Size",
            min_value=50,
            max_value=min(500, len(X_test)) if X_test is not None else 300,
            value=min(150, len(X_test)) if X_test is not None else 150,
            step=25,
            disabled=(xai_method != "SHAP (Game-Theoretic Shapley Values)"),
        )

    try:
        with st.spinner("Calculating feature attributions..."):
            if xai_method == "SHAP (Game-Theoretic Shapley Values)" and X_test is not None:
                importance_df = explainer.get_shap_feature_importance(
                    X_data=X_test,
                    sample_size=shap_n_samples,
                    top_n=SHAP_MAX_DISPLAY,
                )
            else:
                importance_df = explainer.get_feature_importance(top_n=SHAP_MAX_DISPLAY)

        col_tbl, col_cht = st.columns([1, 1])
        with col_tbl:
            st.dataframe(importance_df[["Rank", "Feature", "Relative_Percentage", "Metric_Type"]], use_container_width=True)

        with col_cht:
            st.markdown(f"**Relative Influence ({importance_df['Metric_Type'].iloc[0]}):**")
            chart_df = importance_df.set_index("Feature")[["Relative_Percentage"]]
            st.bar_chart(chart_df)

    except Exception as err:
        st.warning(f"Feature importance not directly extractable for this model: {err}")

    st.markdown("---")

    # 2. Global SHAP Summary & Visualizations
    st.subheader("2. Global SHAP Visualizations (Beeswarm & Mean |SHAP|)")
    from src.explain_shap import SHAP_DIR, LOCAL_SHAP_DIR
    
    summary_img = SHAP_DIR / f"shap_summary_{selected_model_key}.png"
    bar_img = SHAP_DIR / f"shap_bar_{selected_model_key}.png"

    if summary_img.exists() and bar_img.exists():
        tab_bee, tab_bar = st.tabs(["🐝 Beeswarm Summary Plot", "📊 Mean |SHAP| Bar Plot"])
        with tab_bee:
            st.image(str(summary_img), caption=f"SHAP Summary (Beeswarm) — {selected_model_key.replace('_', ' ').title()}", use_container_width=True)
        with tab_bar:
            st.image(str(bar_img), caption=f"Top 15 Features by Mean |SHAP| — {selected_model_key.replace('_', ' ').title()}", use_container_width=True)
    else:
        st.info("Run `python run_shap_analysis.py` to generate high-resolution global beeswarm and bar plots.")

    st.markdown("---")

    # 3. SHAP vs. Gini Importance Comparison Table
    st.subheader("3. SHAP vs. Gini Importance Comparison")
    shap_gini_csv = SHAP_DIR / "shap_vs_gini.csv"
    if shap_gini_csv.exists():
        comp_df = pd.read_csv(shap_gini_csv)
        st.markdown(
            "Comparing Game-Theoretic Shapley values against traditional tree-based Gini/MDI split gain. "
            "Features rising in SHAP have positive rank diffs."
        )
        st.dataframe(comp_df.head(20), use_container_width=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write("**Top Movers (Rose by >= 3 Ranks in SHAP):**")
            risers = comp_df[comp_df["rank_diff"] >= 3].sort_values(by="rank_diff", ascending=False)
            st.dataframe(risers[["feature", "gini_rank", "shap_rank", "rank_diff"]].head(8), use_container_width=True)
        with col_m2:
            st.write("**Top Fallers (Fell by >= 3 Ranks in SHAP):**")
            fallers = comp_df[comp_df["rank_diff"] <= -3].sort_values(by="rank_diff")
            st.dataframe(fallers[["feature", "gini_rank", "shap_rank", "rank_diff"]].head(8), use_container_width=True)
    else:
        st.info("SHAP vs Gini comparison table will appear once `python run_shap_analysis.py` is executed.")

    st.markdown("---")

    # 4. Local Explanations for Error Flows (Missed Attacks & False Alarms)
    st.subheader("4. Local Waterfall Explanations (False Negatives & False Positives)")
    st.markdown(
        "Inspect individual network flows misclassified by the model to diagnose failure modes."
    )
    local_csv = LOCAL_SHAP_DIR / f"local_explanations_{selected_model_key}.csv"
    if local_csv.exists():
        err_df = pd.read_csv(local_csv)
        st.write(f"**Identified Error Flows for `{selected_model_key}` ({len(err_df)} total):**")
        st.dataframe(err_df, use_container_width=True)

        flow_ids = err_df["flow_id"].tolist()
        if flow_ids:
            chosen_flow = st.selectbox(
                "Select Flow ID to View Waterfall Explanation:",
                flow_ids,
                format_func=lambda fid: f"Flow #{fid} — {err_df[err_df['flow_id'] == fid]['error_type'].iloc[0]}",
            )
            err_type_tag = "fn" if "FN" in err_df[err_df["flow_id"] == chosen_flow]["error_type"].iloc[0] else "fp"
            wf_file = LOCAL_SHAP_DIR / f"waterfall_{selected_model_key}_{err_type_tag}_{chosen_flow}.png"

            if wf_file.exists():
                st.image(str(wf_file), caption=f"Waterfall Attribution for Flow #{chosen_flow}", use_container_width=True)
            else:
                st.write(f"Waterfall plot file not found at {wf_file.name}")
    else:
        st.info("Run `python run_shap_analysis.py` to generate error flow waterfall plots.")

    st.markdown("---")

    # 5. Domain Cybersecurity Context
    st.subheader("5. Cybersecurity Domain Interpretation of Top Features")
    st.markdown("How specific network traffic features correlate with known threat behavior:")

    taxonomy_df = pd.DataFrame([
        {"Network Flow Feature": k, "Cybersecurity Interpretation": v}
        for k, v in list(CYBERSECURITY_FEATURE_TAXONOMY.items())[:12]
    ])
    st.dataframe(taxonomy_df, use_container_width=True)

    st.markdown("---")

    # 6. Critical Academic Disclaimer: Correlation vs. Causality
    st.subheader("6. Methodological Distinction: Correlation vs. Causality")
    st.info(explainer.causality_caveat_statement())
