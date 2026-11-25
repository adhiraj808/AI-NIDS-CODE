"""
AI-NIDS: AI-based Network Intrusion Detection and Threat Analysis System.
Streamlit Main Application Entry Point.
Academic Paper Project: 'AI in Cybersecurity and Defense: Risks, Benefits, and Frameworks'
"""

import sys
from pathlib import Path

# Add project root to sys.path for absolute imports
root_path = Path(__file__).resolve().parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st

from dashboard.dataset_analysis import render_dataset_analysis_page
from dashboard.explainability import render_explainability_page
from dashboard.model_comparison import render_model_comparison_page
from dashboard.model_training import render_model_training_page
from dashboard.overview import render_overview_page
from dashboard.research_report import render_research_report_page
from dashboard.threat_analysis import render_threat_analysis_page

# Configure Streamlit page settings
st.set_page_config(
    page_title="AI-NIDS | Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Navigation
st.sidebar.title("🛡️ AI-NIDS Engine")
st.sidebar.caption("Research Prototype v1.0.0")
st.sidebar.caption("Host: Apple Silicon / macOS")
st.sidebar.markdown("---")

PAGES = {
    "Dashboard Overview": render_overview_page,
    "Dataset Analysis": render_dataset_analysis_page,
    "Model Training": render_model_training_page,
    "Model Comparison": render_model_comparison_page,
    "Threat Analysis": render_threat_analysis_page,
    "Explainability (XAI)": render_explainability_page,
    "Research Report": render_research_report_page,
}

selected_page = st.sidebar.radio(
    "Navigation Menu",
    list(PAGES.keys()),
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Term Paper Project:**\n\n"
    "*AI in Cybersecurity and Defense: Risks, Benefits, and Frameworks*\n\n"
    "Target: CICIDS2017 Benchmark"
)

# Render chosen page
render_func = PAGES[selected_page]
try:
    render_func()
except Exception as e:
    st.error(f"An unexpected error occurred on the '{selected_page}' view: {e}")
    st.exception(e)
