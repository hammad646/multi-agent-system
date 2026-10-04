---
description: Diagnose and fix a bug with a regression test
---
1. Reproduce the failure using the command or error the user pasted.
2. Find the root cause. State it in two sentences before changing code.
3. Apply the smallest fix. Do not refactor unrelated code.
4. Add a regression test that fails before the fix and passes after.
5. Run `pytest -q` and report results.
