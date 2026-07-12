# Part 4 — LLM System with Production Guardrails

The final Streamlit dashboard combines the saved ML classifier with a schema-validated LLM recommendation.

Run Parts 1 and 2 first. Required: `OPENAI_API_KEY`; optional: `OPENAI_MODEL`.

1. The Streamlit app selects a cleaned employee record. It is decision support, never an automated employment decision.
2. service.py loads the Part 2 pipeline and calculates an attrition probability from the trained feature set.
3. guardrails.py validates ranges and sanitises text. Only the minimal fields required for the assessment reach the LLM; protected traits are excluded.
4. Part 3 enforces the JSON schema, timeout, and retry policy. A failed call gives a safe dashboard error instead of unverified output.
5. The dashboard has a five-call-per-minute demo rate limit. Production should use Redis or equivalent shared storage.
6. logs/retention_radar.log contains only event metadata and error types, never API keys, prompts, or complete records.
7. The interface requires human review because statistical and LLM output can be wrong or biased.

Run streamlit run app.py

or (not launch streamlit you run part4 app.py first and next you enter the path in terminal)

Eg : stream=streamlit run "e:\Employee-Retention-Radar-An-ML-LLM-Based-Early-Warning-System\Part 4 — LLM System with Production Guardrails\app.py"