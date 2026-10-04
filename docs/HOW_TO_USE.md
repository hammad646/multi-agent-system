# How to use this pack with Antigravity
1. Create a new empty folder, copy everything from this pack into it (including hidden `.agent/`).
2. `git init && git add . && git commit -m "pack"`
3. Open the folder as a workspace in Antigravity.
4. Paste docs/KICKOFF_PROMPT.md as your first message. Check its 8-line summary is correct.
5. Say "start phase 1". Review the diff, run tests, commit. Repeat per phase.
6. After each phase run /review-phase. If something breaks run /fix-bug.
Note: if your Antigravity version does not pick up `.agent/rules` or workflows automatically, AGENTS.md carries the same rules; paste its content in the first message.

SPEC.md is v2: it supersedes the earlier BUILD_SPEC.md. Use only the files in this folder.

The first message to Antigravity is docs/KICKOFF_PROMPT.md. It makes the agent read the spec, summarize it, and show a guardrail table before writing any code, and it requires a guardrail self-audit at the end of every phase.
