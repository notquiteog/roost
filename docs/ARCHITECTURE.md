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
net/         how traffic leaves this machine: one transport, and Tor.
providers/   one adapter per API shape; a registry that routes modality → provider.
media/       parameter schemas, ComfyUI templates, and where results are kept.
agent/       tools, an approval policy, the loop, and the stage its hands are on.
voice/       VAD, and the duplex conversation state machine.
routers/     the websockets and a small HTTP API.
```

Nothing in `agent/` or `voice/` knows which provider it is talking to. That is
what makes "chat on Anthropic, dictation on your own whisper" a configuration
question rather than a code path.

`net/` exists for a similar reason at a lower level. Six adapters each built
their own HTTP session, which was fine until two things had to be true of all
of them at once: that a connection can go through Tor, and that the choice is
per *connection* rather than per install. Six copies of that logic is six
places for it to be subtly different, and the one that gets it wrong is the
one that quietly connects directly.

## Where its hands are

`agent/stage.py` is the answer to a requirement that sounds like a preference
and is not: an agent booking a hotel must not take the mouse away from the
person watching it. An agent that owns the pointer owns the computer, and a
computer you cannot use while it works is a computer you have lent out.

There is exactly one honest way to get that, and it is not politeness about
where the cursor is put. A pointer is a property of a *display*, so the only
way for the agent to have its own is to give it its own display — an X server
started for the session, with its own pointer, keyboard and root window, and
applications launched onto it by environment. That is the `virtual` stage, and
under it the guarantee is structural rather than behavioural.

On the screen you *are* looking at there is one pointer and the agent moves
it. The promise is not available, so the code does not make it: it puts the
pointer back where it found it, and refuses to act at all when it finds the
pointer somewhere it did not leave it, because that means a person has a hand
on the mouse. That is cooperation, not isolation, and the module says so.

A stage also carries a rectangle, which is the whole of multi-monitor
targeting. The model is always shown a picture starting at (0, 0) and never
has to know its monitor begins 2560 pixels along; a coordinate outside the
rectangle is refused rather than clamped, because a clamped click lands on
something real and does something nobody asked for.

One thing worth knowing about launching onto a stage: setting `DISPLAY` is not
enough on a Wayland desktop. A toolkit that finds `WAYLAND_DISPLAY` set
prefers it, connects to the compositor the person is actually using, and puts
its window on their screen — while the agent screenshots its own empty display
and describes a black rectangle. So the stage's environment clears the Wayland
variables too, and an empty value in that dict means *remove*.

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

**One execution path for tools, however the model arrived.** The realtime
gateway has a model upstream that calls tools over its own protocol, and the
obvious shortcut is to give it a quick path — the model is waiting, a person
is listening, and an approval prompt in the middle of a spoken sentence is
awkward. That shortcut is the failure. `AgentSession.invoke` is public for
exactly this: same risk grading, same policy, same suspension on a human, same
events, so the tool cards and the approval bar appear while the conversation
is still going. A second path would be a second place for the policy to be
applied, and the one that got it wrong would be the one nobody was watching.

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


## Four things the live hardware taught us

Every one of these was invisible against a mock and obvious within minutes of
a real local model.

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

**The first model in a list is not necessarily one that can talk.** Ollama's
`/api/tags` lists everything that is pulled, and on an install with
`qwen3-embedding:4b` alongside a chat model the first entry was the embedding
one — so every conversation came back as an HTTP 400 saying that model does
not support chat. The error names the model, not the picker, so it reads as a
broken daemon. Models are now chosen by declared capability, and an agent
session prefers one that can call tools, because a model that connects, talks
and can do nothing is a more confusing failure than one that does not start.

**A model that has lost the thread will call a harmless tool forever.**
Measured: thirty identical calls to a tool reporting the screen size, 59,000
tokens of input, and an answer saying it could not browse. `MAX_STEPS` catches
that after sixty rounds, which is fifty-five too many. Identical calls are now
counted per turn — the first few run, because a screenshot taken twice is two
different pictures, and a loop is stopped with a tool result that says so. The
refusal is emitted as an event rather than returned silently, because a turn
that stops for no visible reason is the worst possible way to report it.

That episode also cost a tool. `desktop_stage` returned the same sentence
every time and was exactly what the confused model reached for; the screenshot
already reported both facts it carried. Fewer tools, same information — and
the same lesson applies to the tool list as a whole, which is why a session
can now be narrowed to the groups it actually needs.
