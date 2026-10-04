# Testing rules
- pytest + pytest-asyncio; tests in `tests/` per SPEC.md section 19.
- Mock Gmail, Calendar, GitHub, Pinecone; use a fake LLM for deterministic graph tests (fakes in `app/testing/fakes.py`).
- Mandatory dangerous-case tests: send never runs without approval; draft never sends; duplicate calendar event not created twice; ambiguous date -> needs_clarification; worker restart keeps jobs; missed email within/outside `MAX_LATE_MINUTES`; duplicate worker cannot double-send; interrupted approval resumes after restart; rejecting email in a workflow keeps the event but sends nothing.
- No test may hit real external services.
- Every bug fix gets a regression test.
- After writing code run `pytest -q` and show the result. If something cannot be tested, say why.
