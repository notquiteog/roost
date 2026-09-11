"""The duplex voice protocol.

Turn-based voice — record, stop, transcribe, answer, speak — is a much
simpler thing to build and it is not what this is. Here both directions are
open at once, which buys three behaviours that turn-taking cannot have:

  * **Barge-in.** The microphone stays hot while the assistant is speaking,
    so interrupting it works the way interrupting a person works. This is the
    single largest difference in how a voice assistant feels, and it is why
    VAD runs on the server rather than in the browser: the server is the only
    place that knows both that you started talking and that it was mid-
    sentence, and it must cancel the synthesis it has already queued.
  * **Partial transcripts.** Text appears while you are still speaking.
  * **Speaking before the answer is finished.** Synthesis starts at the first
    sentence boundary rather than at the end of the completion, which takes
    most of a second out of the gap before the reply begins.

Audio moves as binary websocket frames and everything else as JSON, so the
two never have to be base64'd past each other.
"""

from __future__ import annotations

import time
from typing import Annotated, Literal

from pydantic import BaseModel, Field


def _now() -> float:
    return time.time()


# 16 kHz mono PCM16 in both directions unless a client negotiates otherwise.
# Chosen because every STT model in the pipeline wants 16 kHz, so resampling
# happens once in the browser rather than per hop on the server.
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_FRAME_MS = 20


class AudioFormat(BaseModel):
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: Literal[1] = 1
    encoding: Literal['pcm16', 'opus'] = 'pcm16'


# ---------------------------------------------------------------------------
# Client -> server
# ---------------------------------------------------------------------------


class VoiceStart(BaseModel):
    type: Literal['voice.start'] = 'voice.start'
    format: AudioFormat = Field(default_factory=AudioFormat)
    # Per-modality provider overrides for this call only, so one conversation
    # can dictate on a local model and answer on a hosted one.
    stt: str | None = None
    tts: str | None = None
    llm: str | None = None
    voice: str | None = None
    language: str | None = None
    # Off by default: a call that can run shell commands is a different thing
    # to consent to than a call that can answer questions.
    agent_session_id: str | None = None


class VoiceStop(BaseModel):
    type: Literal['voice.stop'] = 'voice.stop'


class VoiceInterrupt(BaseModel):
    """Explicit barge-in, for a push-to-interrupt button.

    Server-side VAD raises the same thing on its own; this exists because a
    noisy room is exactly where VAD is least trustworthy and a person still
    needs a way to cut the assistant off.
    """

    type: Literal['voice.interrupt'] = 'voice.interrupt'


class VoiceText(BaseModel):
    """Typed input during a call, so a name that will not transcribe can be spelled."""

    type: Literal['voice.text'] = 'voice.text'
    text: str


class VoiceMute(BaseModel):
    """Hold the microphone open but stop sending it upstream.

    Not the same as stopping the call: the assistant may still be speaking,
    and the session, its history and its agent binding all survive.
    """

    type: Literal['voice.mute'] = 'voice.mute'
    muted: bool = True


VoiceCommand = Annotated[
    VoiceStart | VoiceStop | VoiceInterrupt | VoiceText | VoiceMute,
    Field(discriminator='type'),
]


# ---------------------------------------------------------------------------
# Server -> client
# ---------------------------------------------------------------------------


class _VEvent(BaseModel):
    at: float = Field(default_factory=_now)


class VoiceReady(_VEvent):
    type: Literal['voice.ready'] = 'voice.ready'
    # What the client should send.
    format: AudioFormat
    # What the client will receive, which is usually a different rate: speech
    # synthesis returns what it returns.
    output_format: AudioFormat
    stt: str
    tts: str
    llm: str


class SpeechStarted(_VEvent):
    """The server's VAD believes the person started talking."""

    type: Literal['vad.speech_started'] = 'vad.speech_started'


class SpeechStopped(_VEvent):
    type: Literal['vad.speech_stopped'] = 'vad.speech_stopped'
    duration_ms: int = 0


class TranscriptPartial(_VEvent):
    type: Literal['transcript.partial'] = 'transcript.partial'
    text: str


class TranscriptFinal(_VEvent):
    type: Literal['transcript.final'] = 'transcript.final'
    text: str
    # Whether this went on to the model. A final transcript that is only
    # "mm" or silence is reported and dropped.
    submitted: bool = True


class AssistantTextDelta(_VEvent):
    type: Literal['assistant.text.delta'] = 'assistant.text.delta'
    text: str


class SynthesisStarted(_VEvent):
    type: Literal['speech.started'] = 'speech.started'
    # Audio for one utterance carries this id, so a client can throw away
    # buffered audio for a cancelled utterance without dropping the next one.
    utterance_id: str


class SynthesisStopped(_VEvent):
    type: Literal['speech.stopped'] = 'speech.stopped'
    utterance_id: str


class SynthesisCancelled(_VEvent):
    """The assistant was cut off. The client must drop any audio it has buffered."""

    type: Literal['speech.cancelled'] = 'speech.cancelled'
    utterance_id: str
    reason: Literal['barge_in', 'client', 'error'] = 'barge_in'


class VoiceWaiting(_VEvent):
    """The turn has stopped and is waiting on the person.

    Sent as well as spoken. The speech is what someone not looking at the
    screen needs; this is what a screen shows — and a client that reconnects
    mid-suspension has no other way to know the call is waiting rather than
    idle.
    """

    type: Literal['voice.waiting'] = 'voice.waiting'
    # 'question' — the agent asked something. 'approval' — a tool needs a yes.
    kind: str
    prompt: str = ''
    options: list[str] = Field(default_factory=list)
    # True for money and secrets, where the affirmative must be "confirm".
    strict: bool = False
    risk: str = ''


class VoiceAnswered(_VEvent):
    """What a spoken reply was taken to mean.

    Emitted for every attempt including the ones that were not understood,
    because "it did not hear you" and "it heard you and did nothing" look
    identical from outside and want very different reactions.
    """

    type: Literal['voice.answered'] = 'voice.answered'
    kind: str
    heard: str
    understood: str        # 'yes' | 'no' | 'unclear' | 'answer'


class VoiceError(_VEvent):
    type: Literal['voice.error'] = 'voice.error'
    message: str
    fatal: bool = False


VoiceEvent = Annotated[
    VoiceReady | VoiceWaiting | VoiceAnswered | SpeechStarted | SpeechStopped | TranscriptPartial | TranscriptFinal | AssistantTextDelta | SynthesisStarted | SynthesisStopped | SynthesisCancelled | VoiceError,
    Field(discriminator='type'),
]
