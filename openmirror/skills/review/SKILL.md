---
name: review
description: Review uncommitted changes (or a branch, a commit range, or named files) for bugs before they are committed.
argument-hint: "[a branch, a commit range or files — the uncommitted changes if empty]"
---

Review a change for defects, the way a careful colleague would before it is committed.

What to review: $ARGUMENTS

If that is empty, review the uncommitted changes: `git diff HEAD`, plus any untracked files `git status` shows. Read the diff, then read enough of the code around each change to know what it does in context — its callers, the types involved, the tests that cover it.

Look for these, in this order:

1. Things that are wrong: a logic error, a case not handled, an off-by-one, a race, an error swallowed, a resource never released, a check that can be walked around.
2. Things that break something else: a changed signature whose callers were not updated, a change in behaviour that a test or a user depends on.
3. Things that will hurt later: new behaviour with no test, a name that says something the code does not do.

Report each finding with its file and line, what goes wrong and in what situation, and a suggested fix, most serious first. Say how sure you are when you are not. Leave out matters of taste and anything a formatter would fix. If you find nothing wrong, say so plainly — do not invent findings to have something to report.

Do not change any files. This is a review, not a fix.
