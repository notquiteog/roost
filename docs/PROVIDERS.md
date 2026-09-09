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
ROOST_CHAT_PROVIDER=anthropic
ROOST_EMBED_PROVIDER=perch:chat     # embeddings stay on your own hardware
```

## What you can connect

Twenty presets ship, and each is a starting point rather than a constraint —
every field is editable, because these URLs change and an install pinned to
whatever was true when this file was written is worse than no preset at all.

| | |
|---|---|
| **Hosted chat** | OpenAI, Anthropic, Groq, OpenRouter, Fireworks, Together, NanoGPT, DeepInfra, Cerebras, Mistral, xAI |
| **Hosted embedding** | OpenAI, Voyage, Google, and the gateways above |
| **Your hardware** | Ollama (local or remote), LM Studio, vLLM, llama.cpp, AUTOMATIC1111, ComfyUI |
| **Both** | Open WebUI, which is itself a gateway to everything configured over there |
| **All five at once** | Perch — one host, one token |

Anything else that speaks OpenAI's shapes works too: pick any OpenAI-shaped
preset and change the address. That adapter is worth more than the rest put
together, because the shape won.

### "This runs on hardware I control"

A tick box, and it is asked rather than guessed because a remote Ollama looks
exactly like a local one from here. It decides whether `ROOST_LOCAL_ONLY` will
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
| `Qwen3-Embedding-4B` | 2560 | Half the memory for most of the quality; the one to start with on a single consumer GPU |

**Dimensions cannot change under an existing store.** Vectors of different
lengths are not comparable, so the store records what each was made with and
refuses to compare across them — which means a changed model or a changed
`ROOST_EMBED_DIMENSIONS` makes old memories unsearchable rather than wrong.
That is the right failure, and it is worth knowing before you change either.

### The asymmetric ones

Several of these are trained so that a passage being stored and a question
being asked are embedded *differently*, and using one setting for both costs
recall in a way that reads as the model simply being worse than advertised.
Roost handles it per model, because the mechanism differs:

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

## Choosing a model

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
