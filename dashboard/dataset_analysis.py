"""
Dataset Analysis Page for AI-NIDS.
Provides exploratory data analysis, class distributions, attack category breakdowns,
missing/duplicate diagnostics, and feature distribution histograms.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from config.settings import RAW_DATA_DIR, SAMPLE_DATA_DIR
from src.data_loader import DataLoader
from src.sample_data import generate_cicids_benchmark_sample


def render_dataset_analysis_page():
    st.title("📊 Dataset Ingestion & Exploratory Analysis")
    st.markdown(
        "Inspect and audit network traffic captures conforming to the **CICIDS2017** benchmark standard."
    )
    st.markdown("---")

    # Ingestion Source Options
    st.subheader("1. Ingest Network Traffic Dataset")
    tab_upload, tab_sample, tab_raw = st.tabs(["📁 Upload Custom CSV", "⚡ Load CICIDS2017 Benchmark Sample", "🗂️ Browse Local data/raw/"])

    loader = DataLoader()

    with tab_upload:
        st.markdown("Upload any CSV capture from the Canadian Institute for Cybersecurity (CICIDS2017).")
        uploaded_file = st.file_uploader("Select CSV File", type=["csv"], help="Uploads a single day's PCAP flow CSV (e.g. Wednesday-WorkingHours.pcap_ISCX.csv)")
        max_rows = st.number_input("Max records to load (0 = load full file)", min_value=0, max_value=2000000, value=10000, step=5000)

        if st.button("Ingest Uploaded File", key="btn_upload"):
            if uploaded_file is not None:
                with st.spinner("Parsing and standardizing uploaded CSV..."):
                    limit = None if max_rows == 0 else int(max_rows)
                    try:
                        df_loaded = loader.load_csv(uploaded_file, max_rows=limit)
                        df_bin, original_labels = loader.binarize_labels(df_loaded)
                        st.session_state["dataset"] = df_bin
                        st.session_state["dataset_meta"] = loader.metadata
                        st.session_state["original_labels"] = original_labels
                        st.success(f"Successfully ingested {len(df_bin):,} flows from uploaded file!")
                    except Exception as err:
                        st.error(f"Failed to load dataset: {err}")
            else:
                st.warning("Please upload a CSV file first.")

    with tab_sample:
        st.markdown(
            "Quickly initialize the system with an authentic **CICIDS2017 Benchmark Sample** "
            "(78 flow features, simulated DoS, PortScan, BruteForce, and Benign traffic) "
            "for testing and research exploration without waiting for multi-gigabyte downloads."
        )
        sample_path = SAMPLE_DATA_DIR / "CICIDS2017_Benchmark_Sample.csv"
        n_sample_rows = st.slider("Benchmark Sample Size", min_value=1000, max_value=20000, value=5000, step=1000)

        if st.button("Load / Generate Benchmark Sample", key="btn_sample"):
            with st.spinner("Preparing benchmark dataset..."):
                df_sample = generate_cicids_benchmark_sample(n_samples=n_sample_rows, output_filepath=sample_path)
                df_bin, original_labels = loader.binarize_labels(df_sample)
                st.session_state["dataset"] = df_bin
                st.session_state["dataset_meta"] = loader.metadata
                st.session_state["original_labels"] = original_labels
                st.success(f"Loaded {len(df_bin):,} benchmark flows into memory!")

    with tab_raw:
        st.markdown(f"Files currently residing in `{RAW_DATA_DIR}`:")
        raw_files = list(RAW_DATA_DIR.glob("*.csv"))
        if raw_files:
            selected_raw = st.selectbox("Select file from data/raw/:", [f.name for f in raw_files])
            limit_raw = st.number_input("Max records to read", min_value=1000, max_value=1000000, value=25000, step=5000)
            if st.button("Load Raw File", key="btn_raw"):
                with st.spinner(f"Loading {selected_raw}..."):
                    target_file = RAW_DATA_DIR / selected_raw
                    df_raw = loader.load_csv(target_file, max_rows=limit_raw)
                    df_bin, original_labels = loader.binarize_labels(df_raw)
                    st.session_state["dataset"] = df_bin
                    st.session_state["dataset_meta"] = loader.metadata
                    st.session_state["original_labels"] = original_labels
                    st.success(f"Loaded {len(df_bin):,} flows from {selected_raw}!")
        else:
            st.info(
                f"No CSV files found in `{RAW_DATA_DIR}`. Follow instructions in `data/README.md` to download "
                f"the official CICIDS2017 dataset from UNB."
            )

    st.markdown("---")

    # If dataset is loaded, show exploratory analytics
    df = st.session_state.get("dataset", None)
    if df is None:
        st.info("Awaiting dataset ingestion. Please choose an option above to load data.")
        return

    st.subheader("2. Dataset Preview & Dimensions")
    st.write(f"**Shape:** `{df.shape[0]:,} rows × {df.shape[1]} columns`")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("3. Class & Attack Distributions")
    col_dist1, col_dist2 = st.columns(2)

    with col_dist1:
        st.markdown("**Binary Class Distribution (Target):**")
        if "Label" in df.columns:
            class_counts = df["Label"].value_counts().reset_index()
            class_counts.columns = ["Class", "Count"]
            class_counts["Percentage"] = (class_counts["Count"] / len(df) * 100).round(2)
            st.dataframe(class_counts, use_container_width=True)
            # Use native Streamlit chart
            chart_df = class_counts.set_index("Class")[["Count"]]
            st.bar_chart(chart_df)

    with col_dist2:
        st.markdown("**Original Attack Categories (Multi-Class):**")
        orig_labels = st.session_state.get("original_labels", None)
        if orig_labels is not None:
            attack_counts = orig_labels.value_counts().reset_index()
            attack_counts.columns = ["Attack Type", "Count"]
            attack_counts["Percentage"] = (attack_counts["Count"] / len(df) * 100).round(2)
            st.dataframe(attack_counts, use_container_width=True)
            chart_att = attack_counts.set_index("Attack Type")[["Count"]]
            st.bar_chart(chart_att)
        else:
            st.write("Original labels not available.")

    st.markdown("---")
    st.subheader("4. Data Quality, Missing Values & Duplicates")
    col_q1, col_q2, col_q3 = st.columns(3)

    n_dupes = int(df.duplicated().sum())
    n_nans = int(df.isna().sum().sum())
    numeric_df = df.select_dtypes(include=[np.number])
    n_infs = int(np.isinf(numeric_df.values).sum()) if not numeric_df.empty else 0

    col_q1.metric("Duplicate Rows", f"{n_dupes:,}", help="Identical repeated flows")
    col_q2.metric("Missing (NaN) Entries", f"{n_nans:,}", help="Fields requiring imputation")
    col_q3.metric("Infinite (Inf) Values", f"{n_infs:,}", help="Zero-duration divisions common in raw CICIDS2017")

    st.markdown("---")
    st.subheader("5. Feature Distributions & Summary Statistics")
    numeric_features = [c for c in numeric_df.columns if c not in ["Source Port", "Destination Port", "Protocol"]]
    if numeric_features:
        selected_feat = st.selectbox("Inspect Feature Distribution:", numeric_features, index=0)
        feat_series = df[selected_feat].dropna()

        col_stat1, col_stat2 = st.columns([1, 2])
        with col_stat1:
            st.write(f"**Descriptive Statistics for `{selected_feat}`:**")
            st.write(feat_series.describe())
        with col_stat2:
            st.write(f"**Histogram Preview (`{selected_feat}`):**")
            # Downsample for histogram visualization if large
            vals = feat_series.values
            if len(vals) > 5000:
                vals = np.random.choice(vals, 5000, replace=False)
            hist_counts, bin_edges = np.histogram(vals, bins=30)
            hist_df = pd.DataFrame({"Bin": [f"{bin_edges[i]:.1f}" for i in range(len(hist_counts))], "Frequency": hist_counts})
            st.bar_chart(hist_df.set_index("Bin"))
