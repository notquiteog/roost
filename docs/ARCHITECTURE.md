# Architecture

## Why this is a separate project rather than a fork

Open WebUI's licence permits modification and redistribution, but forbids
removing its branding in any deployment above fifty users. A rebranded fork
therefore breaks at exactly the point it succeeds. Roost is a distinct
application that interoperates with Open WebUI instead, and reuses its code
where useful under the terms in `NOTICE`.

Open WebUI also already does more than it is usually given credit for. Before
building anything, the following turned out to exist in 0.11.3:

* an Anthropic-compatible `/api/v1/messages` endpoint, so it is already a
  gateway anything speaking the Messages API can use;
* per-user vector memory in `user-memory-{id}` collections across fifteen
  vector stores, already behind a toggle and a permission;
* image generation against A1111, ComfyUI, OpenAI and Gemini;
* speech in and out against five providers each.

So Roost does not reimplement those. It builds the parts that were missing:
an agent runtime, duplex voice, and one place where the provider for each
modality is chosen and enforced.

## The layers

```
protocol/    wire types. Three processes may speak these, so none owns them.
providers/   one adapter per API shape; a registry that routes modality → provider.
agent/       tools, an approval policy, and the loop that runs them.
voice/       VAD, and the duplex conversation state machine.
routers/     two websockets and a small HTTP API.
```

Nothing in `agent/` or `voice/` knows which provider it is talking to. That is
what makes "chat on Anthropic, dictation on your own whisper" a configuration
question rather than a code path.

## Three decisions worth knowing about

**Risk is assessed from arguments, not from the tool.** `shell` is not
dangerous; `shell("rm -rf /")` is. If every command needed a human, the human
would stop reading within a day, and the one command that mattered would get
the same reflexive yes as the forty `git status` calls before it. The
classifier is deliberately biased: anything it cannot parse confidently is
escalated, never waved through.

**Approval suspends the loop.** It does not proceed optimistically and it does
not time out into a default. A denial is returned to the model as a tool
result explaining that a person said no, so it tries another route rather than
ending the turn and making them repeat themselves.

**Voice detection runs on the server.** The browser knows you started talking.
Only the server knows you started talking *while it was three words into a
reply*, and that the synthesis it has already queued must be discarded. Putting
the detector where that decision is made removes a round trip from the most
latency-sensitive moment in a conversation.

## Perch

Perch's five services each deliberately speak a shape something else already
speaks, so none needs an adapter:

| service | port | shape | Roost adapter |
|---|---|---|---|
| chat | 11434 | Ollama + OpenAI | `ollama.py` |
| dictation | 8080 | OpenAI transcriptions | `openai_compat.py` |
| images | 7860 | AUTOMATIC1111 | `a1111.py` |
| video | 8188 | ComfyUI | `comfyui.py` |
| speech | 8880 | OpenAI speech | `openai_compat.py` |

The probe checks `/healthz` and requires the body to say `service: perch`.
That guard is not theoretical: on the machine this was developed on, an
unrelated Go service answered on the port dictation was then using. Without
the check, transcription would have been posted to it — and now that these are
each backend's *conventional* port rather than a private block, the odds of
something else already holding one are considerably higher, not lower.

One constraint leaks into `comfyui.py`. Perch matches exact paths, so
`/history/{id}` is unreachable through it and only bare `/history` is — so the
provider polls that and looks its own job up, which works both behind Perch
and against a direct ComfyUI.


## Two things the live hardware taught us

Both of these were invisible against a mock and obvious within minutes of a
real local model.

**Reasoning is silence.** Synthesising a sentence at a time is supposed to
start the reply at the first sentence rather than the last. It cannot, if the
model spends the first eight seconds thinking: there is no sentence yet, and
the person just hears nothing and assumes it broke. Measured on gemma4:12b —
1534 characters of reasoning before 276 characters of answer, first content
token at 7.8 s. So the voice path asks for reasoning to be turned off
(`think: false`, ignored by servers that do not know it) and `VoiceConfig`
carries an override for models where thinking is cheap or wanted. The agent
path leaves reasoning on, because there nobody is waiting on a speaker.

**Chat templates leak their own control tokens.** gemma4:12b through Ollama
produced `<channel|>` at the head of an answer during a multi-step tool turn.
Intermittent, template-dependent, and not something a prompt can fix. It would
be a blemish in a transcript and a spoken word out of a speaker, so
`control_tokens.py` strips a narrow set of special-token *spellings* from
content. It requires at least one pipe: without that, the same filter eats
`<user>` and `<system>` out of anyone's XML.
