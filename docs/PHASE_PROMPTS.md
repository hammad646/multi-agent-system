# One prompt per phase (or use /start-phase)

start phase 1: Foundation (layout, config, logging with request_id, state, errors, db + repositories, fakes skeleton, .env.example). Then freeze requirements.lock.
start phase 2: RAG (ingest, hybrid retrieve + RRF + rerank, RAG agent). Verify page citations and "not found".
start phase 3: GitHub MCP server + agent.
start phase 4: Google auth + Calendar MCP server + agent (ambiguity rule, duplicate protection).
start phase 5: Gmail MCP server + email agent + persistent scheduler worker (atomic claim, missed-window logic).
start phase 6: Router, policy, guard, graph, service, SQLite checkpointer, CLI. MVP. Test kill-and-resume at an approval prompt.
start phase 7 (stretch): schedule_and_email workflow with rejection semantics and WorkflowRepository.
start phase 8 (stretch): FastAPI routes over ChatService + minimal web UI (chat, approval card with Approve/Reject/Edit, scheduled emails, documents) + optional API_KEY.
start phase 9 (stretch): hardening, README (incl. Google OAuth Testing-vs-production token note), tracing, Postgres compatibility check.
start phase 10 (NOT YET SPECIFIED): free single-VM online demo. Only after the MVP works locally; ask me to write the spec for it first.

After each phase: "Run /review-phase and show me the table."
On failure: paste the error and run /fix-bug.
