"""
Threat Analysis Page for AI-NIDS.
Performs offline threat classification on novel flow captures, scores malicious probability,
categorizes threat levels, highlights suspicious anomalies, and exports SOC triage reports.
"""

from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np

from config.settings import BENIGN_LABEL, MALICIOUS_LABEL
from src.prediction import ThreatClassifier


def render_threat_analysis_page():
    st.title("🔍 Threat Analysis & Network Traffic Inspection")
    st.markdown(
        "Upload unclassified network flow records for offline threat classification and anomaly scoring."
    )
    st.markdown("---")

    pipeline = st.session_state.get("pipeline", None)
    trainer = st.session_state.get("trainer", None)

    if pipeline is None or trainer is None or not trainer.trained_models:
        st.warning("⚠️ No trained models available. Please train at least one model in **Model Training** first.")
        return

    # Model and Threshold Selection
    st.subheader("1. Threat Engine Configuration")
    col_eng1, col_eng2 = st.columns(2)

    with col_eng1:
        trained_keys = list(trainer.trained_models.keys())
        selected_model_key = st.selectbox(
            "Select Detection Model:",
            trained_keys,
            format_func=lambda k: trainer.model_configs.get(k, {}).get("display_name", k),
        )
    with col_eng2:
        conf_threshold = st.slider(
            "Malicious Classification Threshold:",
            min_value=0.10,
            max_value=0.95,
            value=0.50,
            step=0.05,
            help="Probabilities at or above this threshold will be flagged as MALICIOUS",
        )

    st.markdown("---")

    # Ingestion of traffic to inspect
    st.subheader("2. Supply Traffic Capture to Inspect")
    tab_novel_upload, tab_heldout = st.tabs(["📁 Upload Network Flow Capture (CSV)", "🧪 Inspect Held-Out Test Samples"])

    traffic_to_inspect = None

    with tab_novel_upload:
        uploaded_traffic = st.file_uploader(
            "Upload Traffic Flow CSV for Threat Analysis:",
            type=["csv"],
            key="threat_file_uploader",
        )
        if uploaded_traffic is not None:
            try:
                traffic_to_inspect = pd.read_csv(uploaded_traffic, low_memory=False)
                traffic_to_inspect.columns = [c.strip() for c in traffic_to_inspect.columns]
                st.success(f"Loaded {len(traffic_to_inspect):,} flow records for inspection.")
            except Exception as err:
                st.error(f"Error reading uploaded file: {err}")

    with tab_heldout:
        df_full = st.session_state.get("dataset", None)
        if df_full is not None:
            st.markdown("Take a slice of the active dataset to simulate incoming live traffic streams:")
            n_slice = st.slider("Sample slice size", min_value=10, max_value=500, value=50, step=10)
            if st.button("Extract Traffic Stream Slice"):
                traffic_to_inspect = df_full.sample(n=n_slice, random_state=42).reset_index(drop=True)
                st.session_state["cached_threat_slice"] = traffic_to_inspect
                st.success(f"Extracted {len(traffic_to_inspect)} flows.")
            elif "cached_threat_slice" in st.session_state:
                traffic_to_inspect = st.session_state["cached_threat_slice"]

    st.markdown("---")

    # Execute Inspection
    if traffic_to_inspect is not None and not traffic_to_inspect.empty:
        st.subheader("3. Inspection Execution & Threat Categorization")

        selected_model = trainer.get_model(selected_model_key)
        classifier = ThreatClassifier(model=selected_model, preprocessor=pipeline)

        with st.spinner("Analyzing traffic flows through neural/tree inference engine..."):
            results_df, summary = classifier.classify_traffic(
                traffic_to_inspect, confidence_threshold=conf_threshold
            )

        # Telemetry metrics row
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Inspected Flows", f"{summary['total_inspected_flows']:,}")
        col_m2.metric("Benign Activity", f"{summary['benign_flows']:,}")
        col_m3.metric("Detected Malicious", f"{summary['malicious_flows']:,}", delta=f"{summary['malicious_ratio_percent']}%", delta_color="inverse")
        col_m4.metric("Critical Threat Level", f"{summary['critical_threats']:,}", delta="Urgent SOC Triage", delta_color="inverse")

        st.markdown("---")

        # Threat Visual Breakdown
        st.markdown("**Threat Severity Distribution:**")
        threat_counts = results_df["Threat_Level"].value_counts().reset_index()
        threat_counts.columns = ["Threat Level", "Count"]
        st.bar_chart(threat_counts.set_index("Threat Level"))

        st.markdown("---")

        # Filter and Highlight Results
        st.markdown("**Detailed Flow Inspection Results:**")
        filter_opt = st.radio(
            "Filter Display Records:",
            ["All Records", "Malicious / Suspicious Flows Only", "Critical & High Severity Only"],
            horizontal=True,
        )

        display_df = results_df
        if filter_opt == "Malicious / Suspicious Flows Only":
            display_df = results_df[results_df["Predicted_Label"] == MALICIOUS_LABEL]
        elif filter_opt == "Critical & High Severity Only":
            display_df = results_df[results_df["Threat_Level"].isin(["CRITICAL", "HIGH"])]

        st.write(f"Displaying **{len(display_df):,}** of **{len(results_df):,}** flows:")
        st.dataframe(display_df.head(200), use_container_width=True)

        # Download button
        csv_export = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Classified Threat Intelligence Report (CSV)",
            data=csv_export,
            file_name="nids_threat_analysis_report.csv",
            mime="text/csv",
        )
    else:
        st.info("Awaiting traffic input. Please upload a CSV or extract a slice above.")
