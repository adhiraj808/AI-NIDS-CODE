"""
Research Report Dashboard View for AI-NIDS.
Generates exportable research syntheses, empirical findings, and limitations
for academic term paper dissemination.
"""

from datetime import datetime
import streamlit as st
import pandas as pd

from src.evaluation import compile_comparison_table
from src.report_generator import ResearchReportGenerator


def render_research_report_page():
    st.title("📑 Academic Research Report & Experiment Export")
    st.markdown(
        "*Formal Experiment Synthesis for Academic Paper: 'AI in Cybersecurity and Defense: Risks, Benefits, and Frameworks'*"
    )
    st.markdown("---")

    metrics_history = st.session_state.get("metrics_history", [])
    pipeline = st.session_state.get("pipeline", None)
    dataset_meta = st.session_state.get("dataset_meta", {})
    latest_exp = st.session_state.get("latest_experiment", {})

    if not metrics_history or pipeline is None:
        st.warning("⚠️ No completed experiments found in this session. Please train and evaluate models in **Model Training** first.")
        return

    comp_df = compile_comparison_table(metrics_history)
    preprocessing_audit = pipeline.get_summary()

    report_gen = ResearchReportGenerator(
        dataset_meta=dataset_meta,
        preprocessing_audit=preprocessing_audit,
        metrics_comparison=comp_df,
        experiment_config=latest_exp,
        experiment_name=latest_exp.get("name", "AI-NIDS Empirical Evaluation"),
    )

    # Preview report
    st.subheader("1. Report Preview (Generated Strictly from Verified Results)")
    md_report = report_gen.generate_markdown()
    st.markdown(md_report)

    st.markdown("---")

    # Export Buttons
    st.subheader("2. Export Artifacts for Paper Submission")
    col_exp1, col_exp2, col_exp3 = st.columns(3)

    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    with col_exp1:
        st.download_button(
            "📄 Export Markdown (.md)",
            data=md_report.encode("utf-8"),
            file_name=f"ai_nids_research_report_{timestamp_tag}.md",
            mime="text/markdown",
        )

    with col_exp2:
        csv_bytes = comp_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📊 Export Metrics Table (.csv)",
            data=csv_bytes,
            file_name=f"ai_nids_metrics_{timestamp_tag}.csv",
            mime="text/csv",
        )

    with col_exp3:
        if st.button("💾 Persist Report Bundle to results/"):
            saved_paths = report_gen.export_report_bundle()
            st.success(f"Artifacts saved to results/ directory:\n- {saved_paths['markdown'].name}\n- {saved_paths['json'].name}\n- {saved_paths['csv'].name}")
