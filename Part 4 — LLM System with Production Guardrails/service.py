import importlib.util
import sys
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import]
import pandas as pd
from dotenv import load_dotenv
from guardrails import DashboardRequest, audit  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
load_dotenv(BASE_DIR / ".env")  # Part 4's local configuration takes precedence.
LLM_STRUCTURED_PATH = ROOT / "Part 3 — LLM Integration & Structured Outputs" / "llm_structured.py"
if not LLM_STRUCTURED_PATH.exists():
    raise FileNotFoundError(f"LLM structured module not found at {LLM_STRUCTURED_PATH}")
spec = importlib.util.spec_from_file_location("llm_structured", str(LLM_STRUCTURED_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load llm_structured from {LLM_STRUCTURED_PATH}")
llm_structured = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_structured)
EmployeeContext = llm_structured.EmployeeContext
assess_retention = llm_structured.assess_retention


MODEL_PATH = ROOT / "Part 2 — Supervised Machine Learning" / "outputs" / "attrition_model.joblib"

def predict_and_assess(row: pd.Series):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run Part 2 first.")
    model = joblib.load(MODEL_PATH)
    features = row.drop(labels=["Attrition", "EmployeeID"], errors="ignore").to_frame().T
    probability = float(model.predict_proba(features)[0, 1])
    request = DashboardRequest(employee_id=int(row.EmployeeID), probability=probability, age=int(row.Age), department=str(row.Department), monthly_income=float(row.MonthlyIncome), years_at_company=float(row.YearsAtCompany), job_satisfaction=float(row.JobSatisfaction), work_life_balance=float(row.WorkLifeBalance), overtime=str(row.OverTime))
    context = EmployeeContext(employee_id=request.employee_id, age=request.age, department=request.department, monthly_income=request.monthly_income, years_at_company=request.years_at_company, job_satisfaction=request.job_satisfaction, work_life_balance=request.work_life_balance, overtime=request.overtime, predicted_attrition_probability=request.probability)
    audit("assessment_requested", request.employee_id, "model_probability_calculated")
    assessment = assess_retention(context)
    audit("assessment_completed", request.employee_id, f"risk_level={assessment.risk_level}")
    return probability, assessment
