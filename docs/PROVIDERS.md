# Providers

Every kind of request is routed separately, and the route is resolved when it
is used rather than checked once at start-up. This page is what that means in
practice.

## The rule that shapes all of it

**A model list always comes from the provider.** Nothing here returns a
remembered list, a cached one, or a list assembled from what a vendor's
documentation said when this was written. Asking costs one request and buys
the only answer that is true: a local install has whatever happens to be
pulled, a gateway's catalogue changes weekly, and a key may not have access to
half of what the vendor publishes.

Two services have no listing endpoint at all. They say so, on every row, with
`source: "catalogue"` — so a picker can mark them rather than implying the
server confirmed a name it has never seen.

## Chat and embedding are not one setting

They are separate rows in the Connections screen, separate environment
variables, and separate routes in the registry. This is not tidiness.

Your questions go to the chat model. Your *entire remembered history* goes to
the embedding model, in bulk, every time anything is stored or recalled. Those
are different exposures, they often want different vendors, and a UI that
offers "your AI provider" as one choice is a UI that will eventually ship
someone's memory to a company they picked to write code with.

```bash
OPENMIRROR_CHAT_PROVIDER=anthropic
OPENMIRROR_EMBED_PROVIDER=perch:chat     # embeddings stay on your own hardware
```

The other four are pinned the same way, and only when they are named:
`OPENMIRROR_STT_PROVIDER`, `OPENMIRROR_TTS_PROVIDER`, `OPENMIRROR_IMAGE_PROVIDER`
and `OPENMIRROR_VIDEO_PROVIDER`, each taking its model from `OPENMIRROR_STT_MODEL`,
`OPENMIRROR_TTS_MODEL` (and `OPENMIRROR_TTS_VOICE`), `OPENMIRROR_IMAGE_MODEL` and
`OPENMIRROR_VIDEO_MODEL`. An unnamed modality goes to the first provider that can
serve it, local hardware first — so a machine running Perch with a hosted
gateway for everything else has to say which one speaks.

## What you can connect

Twenty presets ship, and each is a starting point rather than a constraint —
every field is editable, because these URLs change and an install pinned to
whatever was true when this file was written is worse than no preset at all.

| | |
|---|---|
| **Hosted chat** | OpenAI, Anthropic, Google (Gemini, through its OpenAI-compatible endpoint), Groq, OpenRouter, Fireworks, Together, SiliconFlow, Alibaba Cloud (Qwen), DeepSeek, NanoGPT, DeepInfra, Cerebras, Mistral, xAI |
| **Hosted embedding** | OpenAI, Voyage, Google, Alibaba Cloud (`text-embedding-v4`), SiliconFlow, Cohere, Jina, Mistral, and the gateways above |
| **Your hardware** | Ollama (local or remote), LM Studio, vLLM, llama.cpp, whisper.cpp, AUTOMATIC1111, ComfyUI |
| **Images** | AUTOMATIC1111, ComfyUI, Replicate, fal, OpenAI, Google (Imagen and the conversational image model), Black Forest Labs, Stability, Ideogram, Recraft, Runway, Luma, Kling, MiniMax, xAI, Together, Fireworks, DeepInfra, OpenRouter |
| **Video** | ComfyUI, Replicate, fal, OpenAI (Sora), Google (Veo), Runway, Luma, Kling, MiniMax, Stability, OpenRouter |
| **Both** | Open WebUI, which is itself a gateway to everything configured over there |
| **All five at once** | Perch — one host, one token |
| **All six on one key** | OpenRouter — chat, embeddings, dictation, speech, images and video |

**OpenRouter** is OpenAI's shapes for chat, embeddings and speech both ways,
and its own APIs for images and video; each modality lists its models at its
own address, so each picker shows only what can do that job. Two things were
found against the live API rather than in its documentation. Its model list
answers a wrong key with the full catalogue, so **Test** asks `/key`, which
does refuse. And speech comes back at the rate the voice was made at — 24 kHz
from Kokoro, 44.1 kHz from Fish Audio — which openmirror resamples rather than
playing nearly twice as slow.

Anything else that speaks OpenAI's shapes works too: pick any OpenAI-shaped
preset and change the address. That adapter is worth more than the rest put
together, because the shape won.

### "This runs on hardware I control"

A tick box, and it is asked rather than guessed because a remote Ollama looks
exactly like a local one from here. It decides whether `OPENMIRROR_LOCAL_ONLY` will
accept the connection — and `local_only` is enforced when a route resolves,
not in the UI, so a local backend being down is an error rather than a quiet
fallback to a hosted one.

## Embedding models

| model | dimensions | notes |
|---|---|---|
| `text-embedding-3-large` | 3072 | OpenAI. Truncatable — ask for 1024 and vectors are a third the size at close to the same recall |
| `gemini-embedding-2` | 3072 | Google. 8192 input tokens, truncatable from 128 up |
| `voyage-3-large` | 1024 | Voyage. 32k input tokens |
| `Qwen3-Embedding-8B` | 4096 | Open weights. The only entry here that can be both the best available and never leave the room |
| `Qwen3-Embedding-4B` | 2560 | **The floor.** Half the memory of the 8B for most of the quality; the one to start with on a single consumer GPU, and what memory and recall are tested against |

**Dimensions cannot change under an existing store.** Vectors of different
lengths are not comparable, so the store records what each was made with and
refuses to compare across them — which means a changed model or a changed
`OPENMIRROR_EMBED_DIMENSIONS` makes old memories unsearchable rather than wrong.
That is the right failure, and it is worth knowing before you change either.

It is worth knowing *twice* if you are moving up to the floor from something
smaller, because that is a migration and not a setting change: existing
memories do not convert, they stop being findable. Either accept that and let
the store refill, or re-embed deliberately before switching. A floor is aimed
precisely at installs sitting on an older default, so this is the likeliest
moment for anyone to hit it.

### The asymmetric ones

Several of these are trained so that a passage being stored and a question
being asked are embedded *differently*, and using one setting for both costs
recall in a way that reads as the model simply being worse than advertised.
openmirror handles it per model, because the mechanism differs:

* Voyage takes an `input_type` of `query` or `document`.
* `gemini-embedding-001` takes a `taskType`.
* `gemini-embedding-2` and the Qwen3 models take neither — the instruction
  goes in the text, on queries only.

The memory service knows whether it is storing or asking, so it is the layer
that applies this. A passage embedded with a query's instruction is embedded
wrongly, so doing it to both is the same as doing it to neither.

## Tor

Opt in per connection, from the Connections screen. **Never inferred from the
proxy being up** — an install may have Arti running for something else
entirely, and routing a model's traffic through it because it happened to be
there would be a decision about where your prompts go, made on your behalf.

```bash
pip install -e '.[tor]'
TOR_SOCKS_HOST=127.0.0.1
TOR_SOCKS_PORT=9150
```

Four rules, taken from the sibling projects on this machine so that an
operator who moves their proxy moves it once:

* **The environment wins**, then the default. An explicit address on one
  connection beats both, which is how two circuits run on one machine.
* **9150, not 9050.** Arti's default, not the C daemon's. A failed probe
  checks the other implementation's port before concluding nothing is
  listening, because `ECONNREFUSED 127.0.0.1` reads as "the model is down" and
  sends people to debug a machine that was never dialled.
* **`socks5h`, always.** The proxy resolves. Resolving a hostname here and
  dialling the result through Tor leaks the lookup to whatever DNS this
  machine uses, which is the thing the toggle was switched on to prevent.
* **Two minutes is a floor, not a default.** A request over Tor runs on two
  clocks — building the circuit, then waiting for bytes — and a caller asking
  for less gets the floor.

Turning it on with nothing behind it is **refused at the point of use**. It
never falls back to a direct connection, because a toggle that quietly sends
traffic the ordinary way when the proxy is missing is worse than one that
fails: the operator believes they are on Tor.

The probe is honest about its limit. It checks that something is listening on
the port. It does not build a circuit and cannot tell Tor from any other SOCKS
server — which is the truthful answer a cheap check can give, and enough to
tell you whether the box you ticked has anything behind it.

### What Tor does not cover

Two paths deliberately have no Tor option, and it is better to say so than to
let the tick box on another connection imply otherwise:

* **Perch.** A Perch host is its own shape rather than a stored connection —
  one host, one token, five services — and it is normally reached over an SSH
  tunnel to loopback, where there is nothing for Tor to hide. Adding the
  option means giving the probe *and* all five providers a transport, not one
  of them.
* **The realtime API.** One vendor's websocket, which cannot be pointed at
  local hardware in the first place. The Live pane says both things before you
  open it.

* **The open web.** `web_search` and `web_fetch` reach whatever page the
  agent was asked to read, which is a different destination on a different
  axis. Sending someone's browsing through the proxy their GPU happens to need
  would be applying one connection's rule to another's traffic. There is no
  "browse over Tor" option today; if one is added it belongs to the web tools
  as its own setting, not borrowed from a model connection.

Everything else goes through one transport, and a test greps the whole package
for anything that opens a connection without one — not just the adapters,
because the other shape of this bug is a call that never went near an adapter
at all: an evaluation helper, a health check, a token counter, reaching a model
server on a bare client.

That test exists because the failure was live. `OllamaProvider` accepted a
transport and used it nowhere, so a connection with Tor on registered, probed,
showed a "tor" tag — and sent every message straight out. A line or comment
block marked `# transport-exempt: <why>` is how a genuine exception says so,
and a second test refuses a reason too short to review: an exemption nobody
can check is one the next reader assumes was load-bearing.

It reads the syntax tree rather than grepping the text, because a regex also
matches the words in a comment explaining why something *cannot* be done, and
a check that cries wolf gets deleted rather than fixed. And a third test
asserts the guard can still see — that the package is where it thinks it is,
that it scanned a plausible number of files, and that it found the calls that
are certainly there. The failure mode of a check like this is not being wrong;
it is scanning nothing and reporting success, which is indistinguishable from
passing until the day it was meant to catch something.

## Choosing a model

### The floor, and why it is a model *and* a tool budget

**Chat: `qwen3.5:9b` or `gemma4:12b`. Embedding: `Qwen3-Embedding-4B`.**

That is the minimum openmirror's features are built and tested against — not the
smallest thing that runs. Below it the agent loop does not get slower, it gets
unreliable in ways that read as the harness being broken: a model that cannot
hold a tool schema answers in prose where a tool call was needed, and a model
that cannot hold the context loses the plan it just wrote.

**The qualifier is load-bearing, so it goes in the same sentence as the number.**
`gemma4:12b` did a five-step browser task — city, two dates, two guests,
submit, read the results, report three hotels with correct prices — in twenty-
six seconds, correct, first try. *With the tool list narrowed to `browser` and
`web`, about ten tools.* Given the full set of roughly thirty, the same model
on the same task with the same prompt opened the page and then reported that it
had no way to browse.

So "gemma4:12b is enough" is not a true statement on its own. **At the floor,
the tool budget is part of the configuration**, which is why `POST
/api/sessions` takes a `tools` list and `TOOLSETS` exists. A frontier model
does not need the narrowing; a floor model does, and giving it everything is a
way of putting it below the floor without changing the model.

Two honesty notes on the evidence:

* The measurements above are `gemma4:12b`. **There is no `qwen3.5:9b` data
  here** — it is named as the floor because the other projects in this family
  test against it, not because openmirror has measured it.
* Even narrowed, `gemma4:12b` is not enough for `autopilot`. Unattended it
  wandered off task; the harness held — the loop guard stopped it, Stop
  stopped it — but "fully autonomous" is a claim about the model as much as
  about the harness. See the README.

**The floor is a warning, never a wall.** Nothing here refuses a small model.
If you want `qwen3:1.7b` on a 4 GB box to see how far it gets, that is yours to
decide, and narrowing the toolset is the single thing that will help most. What
the floor governs is what a **feature** may assume, not what an **operator** may
run.

### The ceiling: frontier models with thinking

The other end is a first-class target rather than a happy accident, and it is
the end openmirror was designed around: an agent loop that plans, calls tools and
recovers from its own mistakes is exactly the workload reasoning models are
best at.

* Reasoning is **streamed as its own channel**, not folded into the answer, and
  the client renders it as working-out under the reply. A model that
  deliberates for a minute before its first token is a supported shape.
* **Nothing is gated on model size**, and no prompt is shortened for the floor.
  A frontier model gets the same tools through the same guards.
* The one place it is deliberately turned off is a **voice** call, where
  reasoning is silence: `gemma4:12b` emitted 1,534 characters of thinking
  before 276 of content, first content token at 7.8 seconds, so
  sentence-at-a-time synthesis bought nothing. The agent path leaves it on.

### How hard it thinks

One level per session — `off`, `low`, `medium`, `high`, `xhigh`, `max`, or
*default*, which leaves the model at its own. Set it from the **think** chip
beside the approval mode, with `/think high` in the composer, with `effort` on
`POST /api/sessions`, or with `{"type": "policy.set", "effort": "high"}` on the
socket. It changes from the next request on, subagents inherit it, and the
browser remembers the last one chosen for the next session.

The level is one dial and every host spells it differently, which is the whole
of `openmirror/providers/reasoning.py`:

| Host | What is sent |
|---|---|
| Anthropic | adaptive thinking + `output_config.effort` on 4.6 and later (no `xhigh` on 4.6); a `budget_tokens` on Haiku 4.5 and Opus 4.5, which refuse adaptive; the Models API's capability tree overrides the table |
| OpenAI | `reasoning_effort` from the model's own ladder — `none`…`max` on GPT-5.6, `low`…`max` on GPT-6 Astra (no `none`), `minimal` on the original gpt-5; nothing at all to a model that does not reason |
| OpenRouter | `reasoning: {effort}`, which OpenRouter maps per vendor |
| Groq | per model: `reasoning_effort` low–high on gpt-oss, `none`/`default` (plus levels on qwen3.8) on Qwen with `reasoning_format: parsed` |
| Fireworks | `reasoning_effort`, `none` for off |
| Together | `reasoning: {enabled}` on hybrids, `reasoning_effort` on gpt-oss |
| SiliconFlow, Alibaba Cloud | `enable_thinking` and a `thinking_budget`; Alibaba's open-source Qwen3 builds do not think on a non-streamed call |
| Gemini | `reasoning_effort`; only the 2.5 Flash models can be switched off |
| Ollama | `think`: `false` or low/medium/high — and `low` for off on gpt-oss, which ignores `false` |

Two rules hold across every row: a level a model lacks clamps **down** to one
it has, so `max` on a local model is its hardest setting rather than a 400;
and `off` means the least a model allows, never its default. Thinking is never
switched off on Claude with a disabled type — in an agent that makes the model
write tool calls into its visible text — so `off` there is low effort with the
working-out hidden. The same tables live in tern and cryptostore.

### How the default is picked

If you name one, it is used. If you do not, the provider is asked and the
first *usable* one is taken — which is not the same as the first one.

Ollama reports what each model can do, and an install with an embedding model
pulled alongside a chat one used to answer every conversation with an HTTP 400
saying that model does not support chat. So chat routes skip models that
declare only `embedding`, and an agent session prefers one that declares
`tools`: a model that connects, talks, and can call nothing is a more
confusing failure than one that does not start.

Providers that report no capabilities — which is every OpenAI-shaped server —
are not narrowed at all. "Did not say" is not "cannot".

## Keys

Environment variables are read at start-up. Connections added from the UI are
kept in `data/connections.json`, written `0600`, and never sent back to the
page — so editing a connection's Tor setting cannot blank its key, because the
client has nothing to send.

Nothing is encrypted. Anything that can read the file can read the key,
exactly as with `.env`, and pretending otherwise would be worse than saying
so.
