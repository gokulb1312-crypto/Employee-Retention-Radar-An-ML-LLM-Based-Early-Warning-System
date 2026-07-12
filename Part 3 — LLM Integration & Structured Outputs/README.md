# Part 3 — LLM Integration & Structured Outputs

Calls an LLM via HTTP POST and validates a retention recommendation against a strict Pydantic schema.

```powershell
Copy-Item .env.example .env
# add OPENROUTER_API_KEY to .env
python llm_structured.py --employee-id 1001
```

Required: `OPENROUTER_API_KEY`; optional: `OPENAI_MODEL`. See `explain.txt` for prompt, validation, and retry decisions.
