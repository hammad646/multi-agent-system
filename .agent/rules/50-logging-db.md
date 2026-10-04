# Logging and database rules
- structlog JSON logs. Every line has `request_id`; workflow lines also `workflow_id`. Log graph run, router decision, tool call (name, status, duration_ms), errors, approvals.
- Never log OAuth tokens, API keys, or email bodies (unless `LOG_EMAIL_BODIES=true`).
- SQLAlchemy 2 behind repositories (`DocumentRepository`, `ScheduledEmailRepository`, `WorkflowRepository`). No SQL elsewhere.
- Models must be PostgreSQL-compatible: no SQLite-only types/functions; UUID string ids; UTC timestamps.
- Scheduler claim must be atomic: `UPDATE ... SET status='sending' WHERE id=:id AND status='pending'` and proceed only if one row changed.
