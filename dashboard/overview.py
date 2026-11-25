"""
Dashboard Overview Page for AI-NIDS.
Provides high-level cybersecurity telemetry, dataset health metrics,
best model highlights, and experimental status cards.
"""

import streamlit as st
import pandas as pd


def render_overview_page():
    st.title("🛡️ AI-NIDS: Executive Threat & System Overview")
    st.markdown(
        "*Research Prototype: AI in Cybersecurity and Defense: Risks, Benefits, and Frameworks*"
    )
    st.markdown("---")

    # Retrieve session state objects
    df = st.session_state.get("dataset", None)
    metrics_history = st.session_state.get("metrics_history", [])
    trained_models = st.session_state.get("trained_models", {})

    # Top Telemetry Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    if df is not None:
        total_rows = len(df)
        label_col = "Label" if "Label" in df.columns else None

        if label_col:
            is_benign = df[label_col].astype(str).str.upper() == "BENIGN"
            benign_cnt = int(is_benign.sum())
            malicious_cnt = total_rows - benign_cnt
            malicious_pct = (malicious_cnt / total_rows) * 100 if total_rows > 0 else 0.0
        else:
            benign_cnt = "N/A"
            malicious_cnt = "N/A"
            malicious_pct = 0.0

        col1.metric("Dataset Total Records", f"{total_rows:,}", delta=f"{len(df.columns)} Features")
        col2.metric("Benign Flows", f"{benign_cnt:,}" if isinstance(benign_cnt, int) else benign_cnt, delta="Normal Traffic", delta_color="normal")
        col3.metric("Malicious Flows", f"{malicious_cnt:,}" if isinstance(malicious_cnt, int) else malicious_cnt, delta="Threat Traffic", delta_color="inverse")
        col4.metric("Threat Exposure Rate", f"{malicious_pct:.2f}%", delta="Class Ratio" if malicious_pct < 50 else "High Threat Density")
    else:
        col1.metric("Dataset Records", "0", "No Data Loaded")
        col2.metric("Benign Flows", "0", "Awaiting Ingestion")
        col3.metric("Malicious Flows", "0", "Awaiting Ingestion")
        col4.metric("Threat Exposure", "0.00%", "Offline")

    st.markdown("---")

    # Center Section: Best Model & Experiment Status
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🏆 Model Benchmark Champion")
        if metrics_history:
            metric_criterion = st.selectbox(
                "Optimization Criterion:",
                ["F1-Score", "False Negative Rate (FNR)", "Recall (Sensitivity)", "Accuracy", "Inference Time (s)"],
                index=0,
            )

            metrics_df = pd.DataFrame(metrics_history)

            if metric_criterion in ["False Negative Rate (FNR)", "Inference Time (s)"]:
                # Minimize
                best_idx = metrics_df[metric_criterion].idxmin()
                best_val = metrics_df.loc[best_idx, metric_criterion]
                direction = "lowest"
            else:
                # Maximize
                best_idx = metrics_df[metric_criterion].idxmax()
                best_val = metrics_df.loc[best_idx, metric_criterion]
                direction = "highest"

            best_model_name = metrics_df.loc[best_idx, "Model"]

            st.success(
                f"**Top Ranked Model:** {best_model_name}\n\n"
                f"Achieved {direction} **{metric_criterion}** of `{best_val:.6f}` across held-out test splits."
            )

            st.markdown(
                f"- **Model Key:** `{metrics_df.loc[best_idx, 'Model']}`\n"
                f"- **F1-Score:** `{metrics_df.loc[best_idx, 'F1-Score']:.4f}`\n"
                f"- **FNR (Missed Intrusions):** `{metrics_df.loc[best_idx, 'False Negative Rate (FNR)']:.4f}`\n"
                f"- **Throughput:** `{metrics_df.loc[best_idx, 'Throughput (samples/s)']:.1f} samples/sec`"
            )
        else:
            st.info(
                "No models trained yet. Navigate to **Model Training** in the sidebar to train "
                "Logistic Regression, Random Forest, or XGBoost."
            )

    with col_right:
        st.subheader("🔬 Latest Experiment Status")
        if "latest_experiment" in st.session_state:
            exp = st.session_state["latest_experiment"]
            st.write(f"**Experiment Name:** {exp.get('name', 'N/A')}")
            st.write(f"**Executed At:** {exp.get('timestamp', 'N/A')}")
            st.write(f"**Models Evaluated:** {', '.join(exp.get('models', []))}")
            st.write(f"**Test Split Ratio:** {exp.get('test_size', 0.20):.2f}")
            st.write(f"**Random Seed:** `{exp.get('random_state', 42)}`")
        else:
            st.info("No experiment completed in this session yet.")
            st.markdown(
                "**Quick Workflow:**\n"
                "1. Inspect data in **Dataset Analysis**\n"
                "2. Train models in **Model Training**\n"
                "3. Analyze trade-offs in **Model Comparison**\n"
                "4. Inspect model decisions in **Explainability**\n"
                "5. Upload novel captures in **Threat Analysis**\n"
                "6. Export reports in **Research Report**"
            )

    st.markdown("---")
    st.subheader("💡 Academic Cybersecurity Note: Why Accuracy is Insufficient")
    st.markdown(
        """
        > **The Intrusion Detection Evaluation Paradox:**
        > In production network perimeters, over 95% of traffic flows are benign. An algorithm that naively predicts
        > every packet as `BENIGN` achieves over 95% accuracy while completely failing to detect a single intrusion
        > (**0% Recall, 100% False Negative Rate**).
        >
        > In mission-critical cybersecurity, **False Negatives (missed intrusions)** allow threat actors to establish
        > persistence, conduct internal reconnaissance, and exfiltrate enterprise assets undetected. Therefore,
        > our research framework prioritizes **F1-Score, ROC-AUC, and minimization of False Negative Rate (FNR)**.
        """
    )
