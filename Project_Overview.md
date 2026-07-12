# Employee Retention Radar — An ML & LLM-Based Early Warning System (Overview)

# Why it matters:

Employee Attrition Prediction is a machine learning project that predicts whether an employee is likely to leave a company. This helps HR teams identify at-risk employees and take steps to improve retention.

# Business Problem:

Replacing an employee is expensive and time-consuming. Companies want to answer questions such as:

Which employees are most likely to resign?
What factors contribute to employee attrition?
How can HR reduce employee turnover?

The machine learning model predicts:

Target Variable: Attrition
Yes → Employee is likely to leave.
No → Employee is likely to stay.

This predicts employee attrition risk and turns a model result into a structured, safety-conscious retention recommendation.

# Objectives: 

Build a classification model to predict employee attrition.
Identify the most important factors influencing resignations.
Provide actionable insights for HR.
Deploy the model as a web application.

# Run the parts in order from the repository root:

```powershell
python "Part 1 — Data Preparation & Exploration/data_prep_eda.py"
python "Part 2 — Supervised Machine Learning/train_model.py"
python "Part 3 — LLM Integration & Structured Outputs/llm_structured.py" --employee-id 1001
streamlit run "Part 4 — LLM System with Production Guardrails/app.py"
```

Install dependencies first with `python -m pip install -r requirements.txt`. Parts 3 and 4 require an `OPENAI_API_KEY` in a local `.env` file; Parts 1 and 2 do not.

