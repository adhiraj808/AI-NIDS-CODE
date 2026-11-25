"""
Model Comparison Page for AI-NIDS.
Presents empirical benchmarking tables, comparative multi-metric charts,
ROC curves, confusion matrices, and security risk diagnostics.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.evaluation import (
    compile_comparison_table,
    compute_roc_curves_data,
    explain_metric_significance,
)


def render_model_comparison_page():
    st.title("📈 Model Comparison & Benchmark Analytics")
    st.markdown(
        "Evaluate comparative trade-offs between linear baseline, random forest, and gradient boosting."
    )
    st.markdown("---")

    metrics_history = st.session_state.get("metrics_history", [])
    if not metrics_history:
        st.warning("⚠️ No trained model metrics available. Please train models in **Model Training** first.")
        return

    # 1. Performance Comparison Table
    st.subheader("1. Comprehensive Model Performance Benchmark")
    comp_df = compile_comparison_table(metrics_history)
    st.dataframe(comp_df, use_container_width=True)

    # Allow downloading comparison CSV
    csv_bytes = comp_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Comparison Table (CSV)",
        data=csv_bytes,
        file_name="model_comparison_results.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # 2. Metric Comparison Bar Charts
    st.subheader("2. Comparative Performance Charts")

    tab_f1, tab_pr, tab_fnr, tab_lat = st.tabs([
        "🎯 F1-Score & Accuracy",
        "⚖️ Precision vs Recall",
        "🚨 Security Risk (FNR vs FPR)",
        "⏱️ Computational Latency",
    ])

    with tab_f1:
        st.markdown("**F1-Score and Accuracy Comparison:**")
        chart_df = comp_df.set_index("Model")[["F1-Score", "Accuracy"]]
        st.bar_chart(chart_df)

    with tab_pr:
        st.markdown("**Precision vs. Recall Trade-Off:**")
        chart_pr = comp_df.set_index("Model")[["Precision", "Recall (Sensitivity)"]]
        st.bar_chart(chart_pr)

    with tab_fnr:
        st.markdown("**False Negative Rate (Missed Attacks) vs False Positive Rate (Spurious Alerts):**")
        st.caption("Lower is better for both rates. High FNR implies undetected penetrations; high FPR causes SOC analyst burnout.")
        chart_risk = comp_df.set_index("Model")[["False Negative Rate (FNR)", "False Positive Rate (FPR)"]]
        st.bar_chart(chart_risk)

    with tab_lat:
        st.markdown("**Training vs Inference Latency (Seconds):**")
        chart_lat = comp_df.set_index("Model")[["Train Time (s)", "Inference Time (s)"]]
        st.bar_chart(chart_lat)

    st.markdown("---")

    # 3. Confusion Matrices
    st.subheader("3. Confusion Matrices for Evaluated Models")
    cols = st.columns(len(metrics_history))

    for idx, m in enumerate(metrics_history):
        with cols[idx]:
            st.markdown(f"**{m['model_name']}**")
            cm = np.array(m["confusion_matrix"])

            cm_display = pd.DataFrame(
                cm,
                index=["Actual BENIGN (0)", "Actual MALICIOUS (1)"],
                columns=["Pred BENIGN (0)", "Pred MALICIOUS (1)"],
            )
            st.dataframe(cm_display, use_container_width=True)

            st.write(
                f"- **True Negatives (TN):** `{m['true_negatives']:,}`\n"
                f"- **False Positives (FP):** `{m['false_positives']:,}` (FPR: `{m['false_positive_rate']:.4f}`)\n"
                f"- **False Negatives (FN):** `{m['false_negatives']:,}` (FNR: `{m['false_negative_rate']:.4f}`)\n"
                f"- **True Positives (TP):** `{m['true_positives']:,}`"
            )

    st.markdown("---")

    # 4. ROC Curve Analysis
    st.subheader("4. Receiver Operating Characteristic (ROC) Analysis")
    models_predictions = st.session_state.get("models_predictions", {})
    if models_predictions:
        roc_data = compute_roc_curves_data(models_predictions)

        st.markdown("Threshold-independent discriminative power across classifiers:")
        roc_summary = []
        for model_name, r in roc_data.items():
            roc_summary.append({"Model": model_name, "ROC-AUC Score": r["auc"]})

        st.dataframe(pd.DataFrame(roc_summary), use_container_width=True)

        # Build a consolidated curve table for Streamlit line_chart
        # Sample points to construct smooth curve
        st.caption("ROC Curve Points (FPR vs TPR):")
        for m_name, r in roc_data.items():
            fpr_pts = r["fpr"]
            tpr_pts = r["tpr"]
            # Downsample for visualization if very fine
            step = max(1, len(fpr_pts) // 50)
            sample_df = pd.DataFrame({
                "False Positive Rate": fpr_pts[::step],
                f"{m_name} (TPR)": tpr_pts[::step],
            }).set_index("False Positive Rate")
            st.line_chart(sample_df)

    st.markdown("---")

    # 5. Domain Significance Reference Guide
    st.subheader("5. Metric Significance in Cyber Threat Analysis")
    insights = explain_metric_significance()
    for title, desc in insights.items():
        with st.expander(f"📌 {title}"):
            st.write(desc)
