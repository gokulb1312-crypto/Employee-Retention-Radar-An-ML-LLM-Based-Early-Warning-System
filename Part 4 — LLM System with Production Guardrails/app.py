"""Streamlit dashboard for Employee Retention Radar."""
from pathlib import Path

import pandas as pd
import streamlit as st

from guardrails import InMemoryRateLimiter, audit
from service import predict_and_assess

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "Part 1 — Data Preparation & Exploration" / "outputs" / "employee_retention_cleaned.csv"

st.set_page_config(page_title="Employee Retention Radar", page_icon="📡", layout="wide")
st.title("📡 Employee Retention Radar")
st.caption("An ML & LLM-Based Early Warning System — decision support only; human review is required.")

@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run Part 1 first to create the cleaned dataset.")
    return pd.read_csv(DATA_PATH)

if "limiter" not in st.session_state:
    st.session_state.limiter = InMemoryRateLimiter()

try:
    data = load_data()
except FileNotFoundError as error:
    st.error(str(error)); st.stop()

employee_id = st.selectbox("Select employee ID", data["EmployeeID"].tolist())
row = data.loc[data["EmployeeID"] == employee_id].iloc[0]
left, right = st.columns(2)
left.metric("Department", row.Department); left.metric("Tenure", f"{row.YearsAtCompany} years")
right.metric("Job satisfaction", f"{row.JobSatisfaction}/4"); right.metric("Overtime", row.OverTime)

if st.button("Generate retention assessment", type="primary"):
    if not st.session_state.limiter.allow():
        st.warning("Rate limit reached. Wait a minute before another assessment."); st.stop()
    try:
        with st.spinner("Running the model and requesting a structured assessment..."):
            probability, assessment = predict_and_assess(row)
        st.metric("Predicted attrition probability", f"{probability:.1%}")
        st.subheader(f"Risk level: {assessment.risk_level.title()}")
        st.write(assessment.summary)
        col1, col2 = st.columns(2)
        col1.markdown("**Observed risk factors**\n" + "\n".join(f"- {x}" for x in assessment.risk_factors))
        col2.markdown("**Suggested human-reviewed actions**\n" + "\n".join(f"- {x}" for x in assessment.recommended_actions))
        if assessment.human_review_required:
            st.info("Human review is required before acting on this recommendation.")
    except Exception as error:
        audit(
            "assessment_failed",
            int(employee_id) if employee_id is not None else 0,
            type(error).__name__,
        )
        st.error("Assessment could not complete. Run Parts 1–2 and configure the API key; details were logged safely.")

st.divider()
st.caption("Guardrails: input validation, minimal LLM data, schema validation, retries/timeouts, rate limiting, and metadata-only logs.")

