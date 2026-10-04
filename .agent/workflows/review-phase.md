---
description: Review the current phase against SPEC.md and hard rules
---
1. Read `AGENTS.md` and `SPEC.md`.
2. Check against the 14 hard rules (stdout in MCP servers, secrets, approvals in code, naive datetimes, draft-vs-send, structured errors, checkpointer type, SQL only in repositories, thin CLI/API, env names).
3. Check the phase's acceptance criteria one by one.
4. Run `pytest -q`.
5. Output a table: item, pass/fail, evidence. List risks and missing tests. Do not change code unless asked.
