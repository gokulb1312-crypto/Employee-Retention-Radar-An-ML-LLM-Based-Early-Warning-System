# Part 3 — LLM Integration & Structured Outputs

Calls an LLM via HTTP POST and validates a retention recommendation against a strict Pydantic schema.

Required: `OPENROUTER_API_KEY`; optional: `OPENAI_MODEL`.

1. .env and set OPENROUTER_API_KEY; .env is ignored by Git.
2. EmployeeContext validates the input before the API call.
3. RetentionAssessment is a strict Pydantic output contract: risk level, summary, factors, actions, and human-review flag are all required.
4. The script makes an HTTP POST with JSON-schema response formatting. Its prompt prohibits protected-trait inference and automated employment decisions.
5. Pydantic parses the returned JSON. Network, schema, and malformed-response errors are retried three times and never silently accepted.
6. This standalone demo uses a transparent risk proxy; the dashboard replaces it with Part 2's saved model probability.

Run: python llm_structured.py (default eid set you change your preference --employee-id 1001)
