---
name: init
description: Write an AGENTS.md for this project — how to build it, test it and work in it — so every later session starts out knowing.
---

Write an `AGENTS.md` at the root of this project: the file openmirror, and most other coding agents, read at the start of every session. It is for an agent that has never seen this code, and it should save that agent the first ten minutes of every session it ever has here.

Find out before writing anything:

1. What the project is, in one sentence — from the README, the package manifest, the top-level layout.
2. How to build it, run it and run its tests, exactly: the commands themselves, not "use the build tool". Look at the manifest's scripts, a Makefile or justfile, the CI configuration. If the test command is cheap, run it to confirm it works.
3. How the code is laid out: which directories hold what, and where the entry points are.
4. The conventions a newcomer would get wrong — formatting and lint settings, naming, how errors are handled, how tests are written, anything a CONTRIBUTING file insists on.

Then write it. Keep it short, a page rather than a manual: it is read in full at the start of every session, and every line costs that much every time. Put commands in code blocks. Include only what you checked; leave out what you would be guessing. No advice that would suit any project ("write clean code").

If an AGENTS.md or CLAUDE.md is already there, read it first and improve it rather than replacing it: keep what is still true, correct what is not, add what is missing, and tell the person what you changed.
