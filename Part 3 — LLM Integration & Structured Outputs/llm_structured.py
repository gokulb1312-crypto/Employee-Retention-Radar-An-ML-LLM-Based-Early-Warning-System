import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "Part 1 — Data Preparation & Exploration" / "outputs" / "employee_retention_cleaned.csv"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def _api_error_message(response: requests.Response) -> str:
    """Return OpenRouter's safe, useful error detail without exposing the API key."""
    try:
        payload = response.json()
        detail = payload.get("error", {}).get("message") or payload.get("message")
        if detail:
            return str(detail)
    except (ValueError, AttributeError):
        pass
    return response.reason or "Unknown API error"

class EmployeeContext(BaseModel):
    employee_id: int
    age: int = Field(ge=18, le=100)
    department: str = Field(min_length=1, max_length=80)
    monthly_income: float = Field(gt=0)
    years_at_company: float = Field(ge=0, le=70)
    job_satisfaction: float = Field(ge=1, le=4)
    work_life_balance: float = Field(ge=1, le=4)
    overtime: Literal["Yes", "No"]
    predicted_attrition_probability: float = Field(ge=0, le=1)

class RetentionAssessment(BaseModel):
    employee_id: int
    risk_level: Literal["low", "medium", "high"]
    summary: str = Field(min_length=20, max_length=400)
    risk_factors: list[str] = Field(min_length=1, max_length=5)
    recommended_actions: list[str] = Field(min_length=1, max_length=4)
    human_review_required: bool

def load_employee(employee_id: int) -> EmployeeContext:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run Part 1 first so the cleaned dataset exists.")
    df = pd.read_csv(DATA_PATH)
    row = df[df["EmployeeID"] == employee_id]
    if row.empty:
        raise ValueError(f"EmployeeID {employee_id} was not found.")
    r = row.iloc[0]
    proxy_probability = min(0.95, max(0.05, 0.15 + 0.20 * (r["OverTime"] == "Yes") + 0.12 * (r["JobSatisfaction"] <= 2)))
    overtime_val: Literal["Yes", "No"] = "Yes" if str(r.OverTime) == "Yes" else "No"
    return EmployeeContext(employee_id=int(r.EmployeeID), age=int(r.Age), department=str(r.Department), monthly_income=float(r.MonthlyIncome), years_at_company=float(r.YearsAtCompany), job_satisfaction=float(r.JobSatisfaction), work_life_balance=float(r.WorkLifeBalance), overtime=overtime_val, predicted_attrition_probability=proxy_probability)

def _mock_assessment(context: EmployeeContext) -> RetentionAssessment:
    """Generate a mock assessment when API is unavailable (for demo/testing)."""
    prob = context.predicted_attrition_probability
    if prob >= 0.7:
        risk: Literal["low", "medium", "high"] = "high"
    elif prob >= 0.4:
        risk = "medium"
    else:
        risk = "low"
    
    factors: list[str] = []
    if context.overtime == "Yes":
        factors.append("Regular overtime work")
    if context.job_satisfaction <= 2:
        factors.append("Low job satisfaction score")
    if context.work_life_balance <= 2:
        factors.append("Poor work-life balance")
    if context.years_at_company < 2:
        factors.append("Short tenure")
    if not factors:
        factors.append("No significant risk factors identified")
    
    actions: list[str] = [
        "Schedule one-on-one meeting to discuss workload",
        "Review compensation and benefits package",
        "Consider professional development opportunities",
        "Monitor engagement metrics quarterly"
    ][:min(4, max(1, len(factors) + 1))]
    
    return RetentionAssessment(
        employee_id=context.employee_id,
        risk_level=risk,
        summary=f"Employee {context.employee_id} shows {risk} attrition risk based on overtime={context.overtime}, job_satisfaction={context.job_satisfaction}, and work_life_balance={context.work_life_balance}.",
        risk_factors=factors[:5],
        recommended_actions=actions,
        human_review_required=True
    )

def assess_retention(context: EmployeeContext, attempts: int = 3) -> RetentionAssessment:
    load_dotenv(BASE_DIR / ".env")
    # Check for mock mode first
    if os.getenv("MOCK_MODE", "false").lower() == "true":
        return _mock_assessment(context)
    
    # OPENROUTER_API_KEY is accepted as a legacy alias for existing local .env files.
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Copy .env.example to .env and add a key.")
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-5.6-luna")
    if "/" not in model:
        model = f"openai/{model}"
    payload: dict[str, Any] = {
        "model": model, "temperature": 0.2,
        "messages": [{"role": "system", "content": "You are an HR decision-support assistant. Use only supplied data. Do not infer protected traits or make employment decisions. Give actions for human review."}, {"role": "user", "content": "Assess this employee retention context and return only required JSON: " + context.model_dump_json()}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "retention_assessment", "strict": True, "schema": RetentionAssessment.model_json_schema()}},
    }
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            response = requests.post(
                OPENROUTER_API_BASE + "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
            if response.status_code == 404:
                detail = _api_error_message(response)
                raise RuntimeError(
                    f"OpenRouter could not serve model '{model}' (HTTP 404): {detail}. "
                    "Set OPENROUTER_MODEL in .env to a model available to your account."
                )
            response.raise_for_status()
            return RetentionAssessment.model_validate_json(response.json()["choices"][0]["message"]["content"])
        except RuntimeError:
            raise
        except (requests.RequestException, KeyError, json.JSONDecodeError, ValidationError) as error:
            last_error = error
    raise RuntimeError(f"LLM request failed after {attempts} attempts: {last_error}")

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--employee-id", type=int, default=1001)
    assessment = assess_retention(load_employee(parser.parse_args().employee_id))
    print(assessment.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
