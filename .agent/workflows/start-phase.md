---
description: Implement one build phase from SPEC.md
---
1. Ask for the phase number N if not given.
2. Read `AGENTS.md` and `SPEC.md` (sections 0, 3, 4, 5, and the entry for phase N in section 18).
3. List files to create or change, plus assumptions.
4. Implement the phase following the layout and hard rules.
5. Write the phase tests (mocks only) and include the mandatory dangerous-case tests relevant to the phase.
6. Run `pytest -q` and fix failures.
7. Report: files changed, test results, each acceptance check with the exact manual command, anything not verified (and whether MOCK_MODE covered it).
8. Stop. Do not start the next phase.
