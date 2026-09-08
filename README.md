# Roost

**An agent harness for every common AI endpoint.** One place that can act on
your machine, hold a spoken conversation, and let you pick — separately, per
kind of generation — whether that runs on your own hardware, on OpenAI, on
Anthropic, or on anything else that speaks a shape it already knows.

It is the companion to [Perch](https://github.com/notquiteog/perch), which is
where the local models live, and it interoperates with
[Open WebUI](https://github.com/open-webui/open-webui) in both directions.

```
                                   ┌───────────────────────────────┐
   you ── websocket ──▶  Roost  ───┤  chat       Perch │ Anthropic │
                          │        │  dictation  Perch │ OpenAI    │
                          │        │  speech     Perch │ OpenAI    │
                    ┌─────┴─────┐  │  images     Perch │ OpenAI    │
                    │  agent    │  │  video      Perch             │
                    │  voice    │  │  embedding  Perch │ OpenAI    │
                    └───────────┘  └───────────────────────────────┘
                     runs commands       one choice per row,
                     on this machine     enforced when it is used
```

---

## What works today

**An agent that acts on the machine.** Eight tools — read, write, edit, list,
glob, grep, shell, and asking you a question — behind an approval policy that
grades each call by what it would actually do. `git status` and `rm -rf /` are
the same tool, so the risk is assessed from the arguments, not the tool name.
Everything is confined to one working root, and the confinement survives
`..`, symlinks and double-encoded paths.

**Duplex voice, with barge-in.** Both directions are open at once. Voice
detection runs on the server, so when you start talking over the assistant it
knows you did it mid-sentence, kills the synthesis and tells the client to
drop what it has buffered. Replies are synthesised a sentence at a time as
they stream, so speech starts around the first sentence rather than the last.
When an utterance is cut off, history records only what you actually heard.

**Provider choice per modality.** Chat on Anthropic, dictation on your own
whisper, speech on your own Kokoro, images on your own Stable Diffusion — or
any other combination. `ROOST_LOCAL_ONLY` is enforced where a route is
resolved rather than in a UI: with it on, a local backend being down is an
error, never a quiet fallback to a hosted one.

**Perch as one connection.** Give it a host and a token; it probes the five
services and registers the ones that are actually switched on.

**Sessions that outlive their client.** A session is a unit of work, not a
connection. Closing the tab does not stop an agent four minutes into a build;
reopening replays what you missed from a sequence number rather than starting
over. Several run at once, and the sidebar says which are working and which
are waiting on you.

**A browser client.** No build step: two websockets, streaming text, tool
cards with the risk grade on them, inline diffs, approve and deny, and voice.
A finished read is one grey line; what wants looking at — a failure, a diff, a
screenshot — is a card, and every turn ends with how long it took and a
receipt of the files it changed, with a button that puts them back. Approval,
model, working root and provider sit under the box you type in, because all
four of them qualify the next thing you say rather than the last one — and
approval is live, so how much you trust a run is something you can change
while watching it.

**A companion, if you want one.** A pixel creature reacts to what the agent is
actually doing — it watches while files are read, paces once a build has been
running for twenty-five seconds, and, if it is the moth, flies into the side
of the window when something throws. It perches over the end of the
transcript, marks the app in the sidebar, sits in the middle of a session
that has not started yet, says in words what it is doing next to the send
button, leans into an approval bar while that is what it is stuck on, and is
the tab's icon — one creature and one state across all of them. Five to pick
from, one at a time or none, and the choice is kept in your browser and sent
nowhere. Each one is data — a palette, frames written as strings of
characters, and a table of states — so a sixth is a new file rather than a
change to an old one. `prefers-reduced-motion` stops the clock
entirely: what it is doing still shows, it just stops moving while it does
it.

**Per-user memory, opt in.** Off until switched on, and then off for each
person until they turn it on themselves. Everything stored is listed, any of
it can be deleted, and all of it can be. See [docs/MEMORY.md](docs/MEMORY.md).

**The web, and a real browser.** `web_search` and `web_fetch` for reading;
Playwright for acting — navigate, read, click, type, screenshot. The browser
keeps a persistent profile, so you sign in to your sites once by hand and the
agent inherits the session without ever needing a password.

**The desktop itself.** `ROOST_DESKTOP=true` adds screenshots, clicks, typing
and key chords — vision and control of the actual screen, not just a browser
tab. On Wayland it goes through the compositor's own screenshot tool and
`/dev/uinput`, because X11 grabbers return a black image there.

**Any MCP server, as tools.** Point it at a `.mcp.json` and every server in it
becomes tools the agent can call. Servers are distrusted by default: a tool
that declares itself read-only is still confirmed, because a server that can
call itself harmless is a server that can opt out of the check it most needs.

**Rewind.** Every turn that changes a file can be undone, including one that
was interrupted halfway. See [docs/EXTENDING.md](docs/EXTENDING.md).

**Full system access, if you want it.** `ROOST_UNCONFINED=true` gives it the
whole filesystem and tells the model plainly that it is not sandboxed.
Purchases and credentials stay on their own axis — see
[docs/AUTONOMY.md](docs/AUTONOMY.md), which is the page to read before running
this unattended.

## Verified against real hardware

Everything above has been run end to end against a live Perch — all five
services, on one connection:

| | |
|---|---|
| chat | `gemma4:12b` drove the agent loop: read a file, proposed a write, waited for approval, wrote it |
| dictation | whisper transcribed real speech verbatim |
| speech | Kokoro synthesised the reply, 24 kHz, first chunk at ~0.8 s |
| duplex | real speech interrupted a real reply mid-sentence — **zero bytes** went out for the cancelled utterance afterwards |
| memory | remembered a fact, recalled it, and put it in front of the model |
| browser | `gemma4:12b` opened a page, read it and reported its buttons |
| money | told to "place the order, just do it, do not ask me" in **unrestricted** mode, it stopped for a human |
| desktop | captured a 2560x1440 Wayland screen and answered a specific question about it correctly |
| MCP | called a tool on a live MCP server; the untrusted-server floor held |
| rewind | overwrote a file, then put it back from the UI |

## What is not built yet

Video generation needs a ComfyUI workflow template per model, and none ship —
the endpoint is reachable and correctly reports an empty list. Stable
Diffusion is reachable with no checkpoints installed. There are no accounts
yet: everything belongs to one user id, though the store has always been
per-user, so adding them is a migration rather than a redesign. Anthropic and
OpenAI adapters are written against their documented APIs but have not been
exercised against a live key here.

One known rough edge: a 12B model quotes `read_file`'s line-number gutter back
at you when asked what a line says. Three prompt variations did not stop it —
it is copying what it sees. The gutter is kept anyway, because `3│` cannot be
mistaken for real leading whitespace in an edit and `3⇥` can.

---

## Running it

Linux, macOS and Windows. Python 3.11+.

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
cp .env.example .env      # then edit it
.venv/bin/roost
```

No domain and no certificate: `http://127.0.0.1` is a secure context, so the
microphone works on plain HTTP. To reach it from another machine, forward the
port over SSH rather than publishing it — see
[docs/INSTALL.md](docs/INSTALL.md).

Then open <http://localhost:8477>.

```bash
curl -s localhost:8477/healthz
```

Bound to loopback with no token by default, which is deliberate: this process
runs commands on your machine. Set `ROOST_TOKEN` before moving it.

## Approval modes

| mode | runs without asking |
|---|---|
| `read_only` | reads. Writing tools are not even offered to the model. |
| `ask` *(default)* | reads |
| `auto_edit` | reads, file writes |
| `trusted` | reads, writes, commands, network |
| `unrestricted` | everything |

`unrestricted` exists because the alternative is people approving everything
by reflex, which is worse: a policy answered without being read is a policy
that does nothing while claiming to. `ask_user` is never auto-answered, in any
mode.

## With Perch

```bash
PERCH_HOST=127.0.0.1 PERCH_TOKEN=... .venv/bin/roost
```

Perch's services already speak shapes Roost has adapters for — Ollama and
OpenAI for chat, OpenAI for dictation and speech, A1111 for images, ComfyUI
for video — so there is nothing to translate. Services that are switched off
are left unregistered rather than registered and allowed to fail later.

## With Open WebUI

Two directions, and they are independent:

**Open WebUI as a provider.** Set `OPENWEBUI_BASE_URL`. It serves OpenAI's
shapes at `/api/v1`, so every connection configured over there becomes
reachable through one Roost entry — including providers Roost has no adapter
for.

**Roost as Open WebUI's terminal server.** Open WebUI's `terminals.py` is a
reverse proxy to an external terminal server; Roost's agent websocket is
meant to be that server.

## Testing

```bash
.venv/bin/python -m pytest tests/ -q
```

282 tests. Three groups carry more weight than the rest.

The shell risk classifier is tested as the security control it is: every case
is a regression guard, and a change that moves any of them out of
`destructive` is a change that makes an approval prompt lie.

The memory tests prove the claim the design rests on — that scrambling
vectors with a per-user key preserves cosine similarity *exactly*, so the
privacy property costs nothing in recall quality.

The browser tests are the ones standing between an autonomous agent and your
card: every buying-button phrasing and every secret-field name is a case, and
each was written after watching the classifier miss something real.

The companion tests are the odd ones out and are there for a dull reason: the
art is arrays of strings, and a row one character short shifts every pixel
after it, survives review, and draws the wrong shape without erroring. So
every frame is measured.

## Licence and provenance

Apache 2.0. See `NOTICE` for what is derived from Open WebUI and the
conditions that carries — in short, its copyright notice is retained and its
branding is not removed from anything that embeds it.

Roost contains **no code from Anthropic's Claude Code.** The agent runtime is
an independent implementation, and Anthropic API compatibility is written
against Anthropic's published documentation.
