# Roost

**An agent harness for every common AI endpoint.** One place that can act on
your machine, hold a spoken conversation, and let you pick — separately, per
kind of generation — whether that runs on your own hardware, on OpenAI, on
Anthropic, or on anything else that speaks a shape it already knows.

It is the companion to [Perch](https://github.com/notquiteog/perch), which is
where the local models live, and it interoperates with
[Open WebUI](https://github.com/open-webui/open-webui) in both directions.

```
                                   ┌────────────────────────────────────────┐
   you ── websocket ──▶  Roost  ───┤  chat       Perch │ Anthropic │ OpenAI │
                          │        │  embedding  Perch │ Voyage    │ Gemini │
                          │        │  dictation  Perch │ OpenAI    │ Groq   │
                    ┌─────┴─────┐  │  speech     Perch │ OpenAI            │
                    │  agent    │  │  images     Perch │ OpenAI            │
                    │  voice    │  │  video      Perch                     │
                    └───────────┘  └────────────────────────────────────────┘
                     runs commands       one choice per row, made separately,
                     on this machine     enforced when the route resolves
```

Six ways to use it, one at a time, because typing at an agent, talking to
one, and watching one work are different kinds of attention:

| | |
|---|---|
| **Code** | type back and forth, with tool cards, diffs and approvals |
| **Talk** | one button, and then only your voice |
| **Live** | OpenAI's realtime API — speech to speech, no text in the middle |
| **Autopilot** | a goal and a live view of a screen it is driving. No chat |
| **Studio** | images and video, with every knob the backend actually has |
| **Search** | the agent's own search backend, without the agent |

---

## What works today

**An agent that acts on the machine.** Read, write, edit, multi-edit, patch,
outline, list, glob, grep, shell, plan and asking you a question — behind an
approval policy that grades each call by what it would actually do. `git
status` and `rm -rf /` are the same tool, so the risk is assessed from the
arguments, not the tool name. Everything is confined to one working root, and
the confinement survives `..`, symlinks and double-encoded paths.

Each of those exists because of a specific failure. `multi_edit` applies every
edit to a buffer and writes once, so a refactor cannot land halfway.
`read_files` because reading four files should not cost four model turns.
`outline` because reading two thousand lines to find one function wastes a
context window. `plan` because a long task with no visible plan cannot be
followed by the person watching it.

**And a way to offer fewer of them.** A long tool list is not free: gemma4:12b
given the full set opened a page and then reported that it had no way to
browse. The same model, the same task, the same prompt, with only the browser
tools — twenty-six seconds, correct, first try. So a session can be narrowed
to `browser`, `files`, `shell`, `web`, `desktop`, `media` or any combination,
and `ask_user` is always in whatever you pick, because a session that cannot
ask is a session that guesses.

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

**Chat and embedding are chosen separately**, and that is not a nicety. Your
questions go to one and your entire remembered history goes to the other; a
single "AI provider" setting is how someone ends up shipping their memory to a
vendor they picked to write code with.

**Connections are data, not start-up configuration.** Twenty presets — Groq,
OpenRouter, Fireworks, Together, NanoGPT, DeepInfra, Cerebras, Mistral, xAI,
Voyage, Google, LM Studio, vLLM, llama.cpp, Open WebUI and the rest — each a
starting point you can edit, added and tested while the daemon runs. **Model
lists are never remembered**: every picker asks the provider, every time,
because a local install has whatever happens to be pulled and a gateway's
catalogue changes weekly. The two services that publish no listing endpoint
say so on every row rather than implying the server confirmed a name it has
never seen.

**Embeddings that are actually different from each other.** OpenAI
`text-embedding-3-large`, Gemini Embedding 2, Voyage `voyage-3-large`, and
Qwen3-Embedding 8B and 4B on hardware you own. Several of these are
asymmetric — a passage and a question are embedded differently — and the
instruction goes on queries only, because doing it to both is the same as
doing it to neither.

**Tor, opt in per connection.** Never inferred from the proxy being up: that
would be a decision about where your prompts go, made on your behalf. Turning
it on with nothing behind it is refused at the point of use rather than
falling back to a direct connection, which is the failure the toggle exists to
prevent. Same rules as the sibling projects on this machine — Arti's 9150 by
default, and a probe that checks the C daemon's 9050 before concluding nothing
is listening.

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

The modes are described in [docs/MODES.md](docs/MODES.md) and the providers,
routing and Tor in [docs/PROVIDERS.md](docs/PROVIDERS.md).

**The web, and a real browser.** `web_search` and `web_fetch` for reading,
`research` for answering — several searches and the pages behind them in one
call, with every extract labelled inline with the URL it came from rather than
a bibliography at the end. Playwright for acting: navigate, read, click, type,
screenshot. The browser keeps a persistent profile, so you sign in to your
sites once by hand and the agent inherits the session without ever needing a
password — and a second session, which cannot have that profile because
Chromium locks it, gets a fresh browser and is *told* it is signed out, so it
reads a login wall as being logged out rather than as the site being broken.

The same search backend is reachable directly, without a model in the way, for
when you only wanted the links.

**The machine itself.** "Install Steam" is not one instruction — it is
`apt install steam-installer` on Debian, a flatpak where there is no native
package, already-done on a Steam Deck, `winget install Valve.Steam` on
Windows, and a cask on macOS. So the tools take a *name* and the platform
decides what it means: `system_info` reports what this is and how it can
elevate, and `package_install` routes accordingly. A derivative inherits its
parent's route through `ID_LIKE`, so Pop!_OS gets Ubuntu's without being
listed.

Some answers are not the package you asked for. "Install the Epic Games
Store" on Linux resolves to **Heroic**, because Epic ships no Linux client at
all — and the recipe carries the instruction to say so before installing it,
since someone who asked for Epic deserves to know they are getting a different
program and why.

Two failures it avoids by construction. A package manager asked to install
something interactively stops at *"Do you want to continue? [Y/n]"* in a
terminal nobody is reading, so every one of them is invoked non-interactively
with stdin closed. And `sudo` on a machine that wants a password does not
fail — it blocks forever on a prompt going nowhere — so the elevation route is
worked out first and a system-wide install is *refused*, naming the per-user
alternative, when there is no way to become root without a person.

On an immutable system — SteamOS, Silverblue, anything ostree — the system
package managers are present and will not work. The tool says so and points at
flatpak, and says plainly not to disable the read-only root, because an agent
that meets `apt` failing there will otherwise reach for `steamos-readonly
disable`.

**Display settings, honestly split.** `display_hdr` runs the command where the
desktop has one — KDE exposes HDR through `kscreen-doctor` — and where it does
not, which is most of them, it opens the right settings page and says what to
click, for the desktop tools to finish. COSMIC's own `cosmic-randr` has no HDR
subcommand; pretending there is a universal command is how an agent ends up
inventing one, running it, and inventing another.

**Images and video, tuned as far as you like.** The panel is drawn from what
the backend says it has: ask A1111 and its own samplers, schedulers,
upscalers and LoRA names come back as controls, with a sentence each. Nothing
in the client knows what a sampler is, so installing one on the diffusion box
adds it to the dropdown with no release in between. ComfyUI templates mark
their inputs with tokens and the form follows from which tokens are present —
so a new video model is one file, and a "Save (API format)" export can be
imported and tokenised for you. Every result is stored as a file plus its
recipe, including the seed the backend actually used, because a picture you
cannot reproduce is one you cannot iterate on.

**The desktop itself, without taking your mouse.** `ROOST_DESKTOP=true` adds
screenshots, clicks, typing, scrolling and key chords. The part that matters
is *where*: on a **virtual stage** the agent gets an X server of its own — its
own pointer, its own keyboard, its own windows — and applications are launched
onto it by environment. That is the only arrangement under which "it will not
take your mouse" is a guarantee rather than an intention, and it is checked
rather than asserted: the test drives a real Chromium through a booking form
and then reads the real display's pointer to prove it never moved.

On the screen you *are* looking at there is one pointer and it moves it, so
the promise is not available. What is available is that it gives it straight
back, and stops entirely the moment it finds the pointer somewhere it did not
leave it — because that means a person has a hand on the mouse. A stage also
carries a rectangle, which is what multi-monitor targeting is: everything
outside it is neither captured nor clickable, and a coordinate outside is
refused rather than clamped, since a clamped click lands on something real.

On Wayland capture goes through the compositor's own screenshot tool and input
through `/dev/uinput`, because X11 grabbers return a black image there.

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

## What this is built for

### The floor: a model *and* a tool budget

**Chat: `qwen3.5:9b` or `gemma4:12b`. Embedding: `Qwen3-Embedding-4B`.**

What a feature is allowed to depend on, not what will start. Below it the agent
loop does not slow down, it becomes unreliable in ways that look like the
harness misbehaving: prose where a tool call was needed, a plan the model just
wrote and has already lost.

The number does not travel alone. `gemma4:12b` did the five-step booking task
in twenty-six seconds, first try — **narrowed to about ten tools**. Given the
full set of thirty-odd it opened the page and announced it had no way to
browse. So at the
floor the **tool budget is part of the configuration**, which is what
`TOOLSETS` and the `tools` list on `POST /api/sessions` are for. Handing a
floor model everything is a way of putting it below the floor without changing
the model.

Honest about the evidence: those measurements are `gemma4:12b`. `qwen3.5:9b` is
named because the rest of this family tests against it, not because Roost has
measured it. And even narrowed, a 12B is not enough for `autopilot` — see
"What is not built yet".

**A warning, never a wall.** Nothing refuses a small model. `qwen3:1.7b` on a
4 GB box is yours to try, and narrowing the toolset is what will help most.
The floor governs what a **feature** may assume, not what an **operator** may run.

### The ceiling: frontier models with thinking

The end Roost was actually designed around — planning, calling tools and
recovering from its own mistakes is what reasoning models are best at.
Reasoning streams as its own channel and renders as working-out under the
reply; nothing is gated on model size; no prompt is shortened for the floor.
The single exception is a voice call, where reasoning is silence and `think` is
sent off deliberately.

### Where it runs: not a thin client

Roost installs like `claude-code` or `codex` — a daemon on the machine you want
worked on, reached from a browser. **The model may be remote. The agent is
not.** It runs shell with a risk classifier, edits files, drives a real
Chromium, and optionally sees the screen and moves the mouse — all on the host
it is installed on, whichever machine the tokens come from. `ROOST_WORKSPACE`,
the approval modes and `ROOST_UNCONFINED` are all about *that* box.

Worth being blunt about, because "points at a remote AI box" invites the
reading that the local install is just a UI. It is not, and somewhere you would
not put an agent is not somewhere to put this.

Two shapes follow:

**Models elsewhere.** A strong machine running Roost, models on a separate AI
box (Perch, an Ollama on the LAN) or a provider API. Best answers, and the
local box spends nothing on inference. This is the common case and it is what
`ROOST_LOCAL_ONLY` exists to *constrain* — set it and a hosted provider is an
error at route resolution, never a quiet fallback.

**Everything on one box, 16 GB+.** One machine, local models only, nothing
leaving the room: Ollama, whisper, Kokoro, the vector store and the daemon
together. Supported, and 16 GB is a real floor rather than a comfortable one —
a floor chat model is ~6 GB resident, `Qwen3-Embedding-4B` ~2.5 GB, whisper
`base` plus Kokoro ~1.9 GB, the daemon and store ~1 GB: about 11.5 GB with
everything warm. Headroom on 16 GB, none at all on 8. Voice is the first thing
to drop, and the browser and desktop extras are the next — they are separate
installs (`[browser]`, `[desktop]`) precisely so a small box need not carry
them.

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

And the newer half, on the same machine:

| | |
|---|---|
| booking | `gemma4:12b` opened a hotel site in a real Chromium, typed a city and two dates, submitted the form, read the results and reported three hotels with correct prices — **26 s**, first try, with the toolset narrowed to `browser` + `web`. Given all thirty-odd tools the same run failed outright, so the narrowing is part of the result rather than a detail of it |
| your mouse | throughout that, on a virtual stage, the real display's pointer did not move. The test asserts it |
| the real web | the same browser, on the same private display, loaded google.com and found its twenty interactive elements |
| autopilot | a goal, a live view at ~1.4 fps, 27 JPEG frames in 19 s, and a Stop button that stopped it |
| money | "Book this room — pay now" graded a purchase, and refused in **every** mode including `unrestricted` |
| studio | an A1111-shaped server's samplers, schedulers, upscalers and LoRAs came back as real controls; a generation went through with the settings the form had set, stored with the seed the backend used |
| connections | Groq-shaped connection added, tested and removed at runtime; the Tor probe found the C daemon on 9050 and said which implementation it was |
| model choice | an install with an embedding model pulled alongside a chat one picked the chat one — see below |

Three things that only a live model found, all now fixed and pinned by tests:

* **`<input type="date">` cannot be typed into.** Keystrokes go to whichever
  of its three segments has focus, and `2026-10-14` became `61014-02-02` — a
  date the form accepted and echoed back. Segmented inputs are set rather
  than typed; ordinary text still gets keystrokes, because that is what an
  autocomplete listens for.
* **The first model in a list is not necessarily one that can talk.**
  `/api/tags` listed `qwen3-embedding:4b` first, so every conversation came
  back as an HTTP 400 saying that model does not support chat — which reads as
  a broken daemon rather than a bad default. Models are now chosen by declared
  capability, and an agent session prefers one that can call tools.
* **A model that has lost the thread will call a harmless tool forever.**
  Thirty identical calls, 59,000 tokens of input, and an answer saying it
  could not browse. Identical calls are now counted per turn: the first few
  run, and a loop is stopped with a tool result that says so — visibly, so a
  turn that stops does not simply go quiet.

## What is not built yet

Video generation still needs a ComfyUI workflow template per model, and none
ship — but bringing one is now a command rather than an editing job: export
your graph with **Save (API format)**, import it, and the prompt, seed, size
and sampler settings are found and turned into controls, with a list of what
it found so you can see whether it worked. None ship because a template names
the checkpoints and custom nodes *your* ComfyUI has; one written here would
fail at the queue with an error about a missing key.

Anthropic, OpenAI, Voyage, Google and the OpenAI-compatible gateways are
written against their documented APIs and have not been exercised against a
live key here — Perch is what this machine has. The realtime adapter is in
the same position, and it is the one with the least margin for error, since
its event names moved between the beta and GA and only one of the two
spellings can be checked without a key.

There are no accounts yet: everything belongs to one user id, though the store
has always been per-user, so adding them is a migration rather than a
redesign.

**Small models are the real constraint on the autonomous modes.** gemma4:12b
does a five-step browser task reliably when the tool list is narrowed to what
it needs, and wanders when it is not — in autopilot, where nobody is answering
it, it wandered off to Google and then produced an unrelated refusal. The
harness held: the loop guard stopped it, the frames kept streaming, Stop
stopped it. But "fully autonomous" is a claim about the model as much as about
the harness, and on a 12B it is not yet true.

One known rough edge: a 12B model quotes `read_file`'s line-number gutter back
at you when asked what a line says. Three prompt variations did not stop it —
it is copying what it sees. The gutter is kept anyway, because `3│` cannot be
mistaken for real leading whitespace in an edit and `3⇥` can.

---

## Running it

Linux, macOS and Windows. Python 3.11+.

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
cp .env.example .env      # then edit it — it is read at start-up
.venv/bin/roost
```

The optional extras, none of which an install needs to be complete:

```bash
pip install -e '.[browser]'   # Playwright, then: playwright install chromium
pip install -e '.[desktop]'   # seeing and driving a screen
pip install -e '.[tor]'       # routing a connection through a SOCKS proxy
```

A screen of the agent's own additionally needs `Xvfb` from your package
manager — `apt install xvfb`. Without it, desktop control falls back to the
screen you are looking at and says so in the log, because an agent that cannot
see anything is not a safer agent, it is a useless one.

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

Note what is in none of those rows: **purchases and credentials**. They are
not on that ladder at all — they are a second axis, and both are *on*, because
a harness meant to finish a real task has to be able to reach the end of one.
It can buy things. It can type a password, a card number or a one-time code.

What it cannot do is either of them without you.

A purchase is confirmed **in every mode, including `unrestricted`, and
including an autopilot run nobody is watching**. There is no setting that
skips that prompt — deliberately, because a setting like that is one somebody
turns on during a demo and still has on six months later. Neither is ever
remembered either: "don't ask again" about spending money is the one answer
nobody should be able to give once. `ROOST_ALLOW_PURCHASES=false` turns a
purchase into a refusal rather than a prompt, for an install that should not
be able to buy anything at all.

And a secret never becomes readable by anything else. Not in the approval
prompt, not in the tool result, not in the transcript, and not in the page
read — `browser_read` redacts a field's value *and* its label when the field
holds a secret, which is how a typed password was coming back out. That was
found by the test that now proves it, against a real browser and a real form.
`browser_hand_over` is still there and is still better when you are at the
keyboard: a value the agent never receives cannot leak from anywhere.

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

394 tests. Some groups carry more weight than the rest.

The shell risk classifier is tested as the security control it is: every case
is a regression guard, and a change that moves any of them out of
`destructive` is a change that makes an approval prompt lie.

The memory tests prove the claim the design rests on — that scrambling
vectors with a per-user key preserves cosine similarity *exactly*, so the
privacy property costs nothing in recall quality.

The browser tests are the ones standing between an autonomous agent and your
card: every buying-button phrasing and every secret-field name is a case, and
each was written after watching the classifier miss something real.

`test_booking_walkthrough.py` is end to end and has no mocks below the
provider: a real Xvfb display, a real Chromium launched onto it, a real
booking form filled in, and then the two assertions the whole desktop design
exists for — that the real display's pointer did not move, and that "Book this
room — pay now" is refused in every approval mode. It skips where the machine
has no Xvfb or no Chromium rather than failing, because the suite has to pass
on a laptop with neither.

The Tor tests pin the proxy rules with no socket and no network: the
precedence, the Arti default, and the diagnosis that names the C daemon's port
before concluding nothing is listening. The failure they prevent is silent —
an operator who moved their proxy and finds half their traffic still dialling
the old port gets no error, only something that does not work.

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
