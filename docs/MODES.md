# The six modes

Not tabs over a shared workspace. Typing at an agent, talking to one, and
watching one work are different kinds of attention, and a layout that keeps a
text box on screen during a voice call is a layout that tells you to type.

One is on screen at a time, chosen from the rail on the left. Each hands back
what it holds when you leave it — a microphone left open because somebody
clicked away is a microphone left open.

## Code

The one everything else is measured against: a transcript, tool cards with the
risk grade on them, inline diffs, approve and deny, and a receipt at the end
of each turn saying what changed and offering to put it back.

The controls that qualify *the next thing you say* — approval mode, model,
working root, provider — sit under the box you type in rather than in a title
bar three hundred pixels away.

## Talk

One button, and then only your voice.

The button is in the bar and it does the whole thing: switches the app, opens
the microphone, starts listening. A "talk mode" that then needs a second
control to actually start talking is two buttons wearing one coat.

There is no composer here on purpose. Acting on the machine is a tick box, off
by default and decided per call, because consenting to be listened to is not
the same as consenting to have commands run — and the voice gateway enforces
that separately, so the box only decides whether an agent session is attached
at all.

Barge-in works the way it does everywhere else in Roost: talk over it and the
synthesis is killed server-side, the client drops what it has buffered, and
the history records only what you actually heard.

`Ctrl+Shift+T` from anywhere. `Escape` stops whatever is talking.

## Live

OpenAI's realtime API: speech to speech, one model, no text in the middle.

What you gain is latency and prosody — the model hears *how* something was
said and answers before a transcript would have existed. What you give up is
choice, and the pane says so: it is one vendor's model, it cannot be pointed
at your own hardware, and everything said in the room goes to them while it is
open. Ordinary Talk routes dictation, thinking and speech separately, so each
of those can stay on your machine.

Its tool calls go through the same approval path as the typed loop. If the
policy says a human decides, the model waits — and it is told, in the tool
result, that it is waiting on a person, so it says so out loud rather than
going quiet.

## Autopilot

A goal, and a window. No composer, no transcript, no tool cards: a live view
of a screen it is driving and one line of what it is doing.

It gets a screen of its own by default — see [AUTONOMY.md](AUTONOMY.md) — so
you keep using your computer while you watch. Running on *your* screen is
possible and has to be asked for in words; the refusal happens before anything
moves.

The picture and the narration come down separate channels, which is not an
accident. The narration is small, ordered and replayed to a client that
reconnects; the video is large, continuous and worthless five seconds later.
On one channel you would either store an hour of stale frames to replay, or
lose the record of what happened when someone reloads the page.

If it stops to ask something, the question appears in Code, with the full
context around it.

## Studio

Images and video, with every control the backend actually has.

The panel is drawn from what the server says: ask an A1111 and its own
samplers, schedulers, upscalers and LoRA names come back as real dropdowns,
each with a sentence explaining what it does. Nothing in the client knows what
a sampler is, so installing one on the diffusion box adds it here with no
release in between.

"Everything else" is a fold, not a lock. Every parameter is present and
settable; the ones you need most days are open and the other twenty are one
click away. Hiding one behind a preference would mean the person who wants it
has to find out it exists first.

Video is a job rather than a request — minutes, not seconds — so it starts,
reports how long it has been going, and can be stopped. There is no progress
bar because ComfyUI behind Perch exposes polling and nothing else, and a
percentage would be a fiction.

Every result is stored as a file plus a JSON sidecar holding the prompt, the
seed the backend *actually used*, and every setting. A picture you cannot
reproduce is a picture you cannot iterate on.

### Bringing a ComfyUI workflow

None ship, because a template names the checkpoints and custom nodes your
ComfyUI has. Bringing yours is one command — see
[roost/providers/workflows/README.md](../roost/providers/workflows/README.md).
The prompt, seed, size and sampler settings are found and turned into
controls, and what it found is reported back so you can see whether it worked.

## Search

The agent's search backend, without the agent.

Worth having for a reason that is not obvious: that backend is configured
once, with a key, and is usually better than a browser tab — a self-hosted
SearxNG, or a Brave key that does not track you. Looking something up should
not have to cost a model turn.

Results can be read in place, through the same fetch the agent uses — so the
private-address guard applies to a person clicking a link, not only to a model
following one.
