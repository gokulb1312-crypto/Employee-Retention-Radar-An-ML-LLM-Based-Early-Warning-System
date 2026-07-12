# Part 4 — LLM System with Production Guardrails

The final Streamlit dashboard combines the saved ML classifier with a schema-validated LLM recommendation.

```powershell
Copy-Item .env.example .env
# add OPENAI_API_KEY to .env
streamlit run app.py
```

Run Parts 1 and 2 first. Required: `OPENAI_API_KEY`; optional: `OPENAI_MODEL`. See `explain.txt` for guardrails and limitations.

