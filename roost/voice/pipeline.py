"""The duplex conversation loop.

Both directions are live at once, so the loop is not a sequence of steps but
a small state machine with two things happening at the same time: audio
arriving from the person, and audio being generated for them. Everything
interesting comes from the overlap.

**Barge-in.** While synthesis is playing, the microphone is still being fed
to the detector. When it opens, the current utterance is cancelled: the
synthesis task is killed, any queued audio is dropped, and the client is told
to discard what it has buffered. Without that last part the assistant keeps
talking for as long as the client had audio queued, which is exactly the
behaviour that makes voice assistants feel deaf.

**Speaking before the answer is finished.** Text is split into sentences as
it streams and each is synthesised as it completes, so the reply starts
around the first sentence rather than after the last. On a local model this
is most of the perceived latency.

**What was already said is what it remembers.** When an utterance is cut off
partway, the history records the part that was actually spoken aloud, not the
full text the model generated. Otherwise the assistant believes it told you
something you never heard, and the rest of the conversation is built on that.
"""

from __future__ import annotations

import asyncio
import logging
import re
import struct
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from roost.protocol.voice import (
    AssistantTextDelta,
    SpeechStarted,
    SpeechStopped,
    SynthesisCancelled,
    SynthesisStarted,
    SynthesisStopped,
    TranscriptFinal,
    VoiceError,
)
from roost.providers.base import ChatRequest, Message, StreamText, TextBlock
from roost.voice.vad import Ringbuffer, Vad, VadConfig, VadEvent

log = logging.getLogger(__name__)

# Sends one JSON event to the client.
EmitEvent = Callable[[Any], Awaitable[None]]
# Sends one chunk of PCM audio, tagged with the utterance it belongs to.
EmitAudio = Callable[[str, bytes], Awaitable[None]]

# A final transcript this short is almost always the detector opening on a
# cough or a chair. Reported to the client, never sent to the model.
MIN_UTTERANCE_CHARS = 2
FILLERS = {'uh', 'um', 'mm', 'hm', 'hmm', 'ah', 'er', 'eh', 'mhm', 'uh huh', 'you'}

_SENTENCE_END = re.compile(r'(?<=[.!?…])\s+|\n{2,}')
# Abbreviations that end in a full stop without ending a sentence. Small on
# purpose: a wrong split costs a slightly odd pause, a missed one costs a
# whole sentence of latency, so this errs toward splitting.
_ABBREV = re.compile(r'\b(?:[A-Z]|Mr|Mrs|Ms|Dr|Prof|St|vs|etc|e\.g|i\.e|approx|Fig|No)\.$', re.I)


def pcm_to_wav(pcm: bytes, sample_rate: int = 16_000, channels: int = 1) -> bytes:
    """Wrap raw PCM16 in a WAV container.

    Every transcription endpoint in use wants a container rather than raw
    samples, and a 44-byte header is cheaper than an encoder.
    """
    return (
        b'RIFF'
        + struct.pack('<I', 36 + len(pcm))
        + b'WAVEfmt '
        + struct.pack('<IHHIIHH', 16, 1, channels, sample_rate, sample_rate * channels * 2, channels * 2, 16)
        + b'data'
        + struct.pack('<I', len(pcm))
        + pcm
    )


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Take complete sentences off the front, leaving the rest.

    Returns (complete, remainder). The remainder is whatever has not yet been
    terminated, which is held until more text arrives or the stream ends.
    """
    out: list[str] = []
    rest = buffer

    while True:
        match = _SENTENCE_END.search(rest)
        if not match:
            break
        candidate = rest[: match.start() + 1]
        if _ABBREV.search(candidate.rstrip()):
            # An abbreviation, not a sentence end: look past it.
            following = _SENTENCE_END.search(rest, match.end())
            if not following:
                break
            candidate = rest[: following.start() + 1]
            rest = rest[following.end() :]
        else:
            rest = rest[match.end() :]
        if candidate.strip():
            out.append(candidate.strip())

    return out, rest


def is_meaningful(text: str) -> bool:
    """Whether a transcript is worth sending to the model."""
    cleaned = re.sub(r'[^\w\s]', '', text).strip().lower()
    if len(cleaned) < MIN_UTTERANCE_CHARS:
        return False
    words = cleaned.split()
    # All filler: whisper reliably hallucinates 'you' and 'thank you' on
    # silence, and answering those makes the assistant talk to itself.
    return bool(words) and not all(w in FILLERS for w in words)


@dataclass
class VoiceConfig:
    sample_rate: int = 16_000
    frame_ms: int = 20
    stt_model: str = 'whisper-1'
    tts_model: str = 'tts-1'
    tts_voice: str = 'alloy'
    llm_model: str = ''
    language: str | None = None
    # Passed through to the chat provider. The default turns reasoning off;
    # override it for a model where thinking is cheap or wanted.
    llm_extra: dict = field(default_factory=lambda: {})
    system_prompt: str = (
        'You are in a spoken conversation. Reply in one or two sentences unless asked for more. '
        'Write what should be said aloud: no markdown, no lists, no code blocks, no emoji. '
        'Expand numbers, symbols and abbreviations into words. If you did not understand, say so '
        'briefly and ask.'
    )
    vad: VadConfig = field(default_factory=VadConfig)


class VoiceSession:
    def __init__(
        self,
        *,
        stt: Any,
        tts: Any,
        llm: Any,
        config: VoiceConfig,
        emit_event: EmitEvent,
        emit_audio: EmitAudio,
        agent: Any = None,
    ) -> None:
        self.stt = stt
        self.tts = tts
        self.llm = llm
        self.config = config
        self.emit_event = emit_event
        self.emit_audio = emit_audio
        # When set, transcripts drive an agent session instead of a bare chat
        # model, so a voice call can act on the machine.
        self.agent = agent

        self.vad = Vad(config.vad)
        self.preroll = Ringbuffer(config.sample_rate, ms=400)
        self.history: list[Message] = []

        self._utterance = bytearray()
        self._capturing = False
        self._muted = False
        self._partial = bytearray()

        self._reply: asyncio.Task[None] | None = None
        self._utterance_id: str | None = None
        # What has actually been played to the person for the current
        # utterance, so an interruption can be recorded truthfully.
        self._spoken: list[str] = []

    @property
    def frame_bytes(self) -> int:
        return self.vad.frame_bytes

    @property
    def is_speaking(self) -> bool:
        return self._reply is not None and not self._reply.done()

    # -- audio in -----------------------------------------------------------

    async def feed(self, audio: bytes) -> None:
        """Take arbitrary-sized audio from the client and process whole frames.

        Browsers deliver whatever their audio worklet produces, which is not
        the detector's frame size, so a remainder is carried between calls.
        """
        if self._muted:
            return

        self._partial.extend(audio)
        size = self.frame_bytes

        while len(self._partial) >= size:
            frame = bytes(self._partial[:size])
            del self._partial[:size]
            await self._frame(frame)

    async def _frame(self, frame: bytes) -> None:
        if self._capturing:
            self._utterance.extend(frame)
        else:
            self.preroll.push(frame)

        event = self.vad.push(frame)

        if event is VadEvent.SPEECH_START:
            await self.emit_event(SpeechStarted())

            # The decisive moment: they started talking while it was talking.
            if self.is_speaking:
                await self._cancel_reply('barge_in')

            self._capturing = True
            # Pre-roll first, so the utterance starts before the detector did.
            self._utterance = bytearray(self.preroll.take())
            self._utterance.extend(frame)

        elif event is VadEvent.SPEECH_END and self._capturing:
            self._capturing = False
            captured = bytes(self._utterance)
            self._utterance = bytearray()
            self.preroll.clear()
            ms = int(len(captured) / 2 / self.config.sample_rate * 1000)
            await self.emit_event(SpeechStopped(duration_ms=ms))
            await self._handle_utterance(captured)

    async def mute(self, muted: bool) -> None:
        self._muted = muted
        if muted:
            # Drop a half-formed utterance rather than resuming it later with
            # a hole in the middle.
            self._capturing = False
            self._utterance = bytearray()
            self._partial = bytearray()
            self.vad.reset()
            self.preroll.clear()

    # -- transcription and reply -------------------------------------------

    async def _handle_utterance(self, pcm: bytes) -> None:
        if not pcm:
            return
        try:
            transcript = await self.stt.transcribe(
                pcm_to_wav(pcm, self.config.sample_rate),
                model=self.config.stt_model,
                language=self.config.language,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception('transcription failed')
            await self.emit_event(VoiceError(message=f'transcription failed: {exc}'))
            return

        text = (transcript.text or '').strip()
        if not is_meaningful(text):
            # Reported anyway: a client showing 'heard nothing' is far less
            # confusing than one that shows nothing at all.
            await self.emit_event(TranscriptFinal(text=text, submitted=False))
            return

        await self.emit_event(TranscriptFinal(text=text, submitted=True))
        await self.say(text)

    async def say(self, text: str) -> None:
        """Take a turn from text, as if it had been spoken."""
        if self.is_speaking:
            await self._cancel_reply('client')
        self.history.append(Message(role='user', content=[TextBlock(text=text)]))
        self._reply = asyncio.create_task(self._reply_turn())

    async def _reply_turn(self) -> None:
        utterance_id = uuid.uuid4().hex[:12]
        self._utterance_id = utterance_id
        self._spoken = []
        buffer = ''
        full: list[str] = []

        await self.emit_event(SynthesisStarted(utterance_id=utterance_id))

        try:
            async for event in self._generate():
                if not isinstance(event, StreamText):
                    continue
                await self.emit_event(AssistantTextDelta(text=event.text))
                buffer += event.text
                full.append(event.text)

                sentences, buffer = split_sentences(buffer)
                for sentence in sentences:
                    await self._speak(utterance_id, sentence)

            # Whatever is left when the model stops, terminated or not.
            if buffer.strip():
                await self._speak(utterance_id, buffer.strip())

            self.history.append(Message(role='assistant', content=[TextBlock(text=''.join(full))]))
            await self.emit_event(SynthesisStopped(utterance_id=utterance_id))

        except asyncio.CancelledError:
            # Record only what was actually heard. The rest never happened as
            # far as the person is concerned, and the model must not act later
            # as though it had been said.
            spoken = ' '.join(self._spoken).strip()
            self.history.append(
                Message(
                    role='assistant',
                    content=[TextBlock(text=(spoken + ' —' if spoken else '—') + ' [interrupted]')],
                )
            )
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception('reply failed')
            await self.emit_event(VoiceError(message=str(exc)))
            await self.emit_event(SynthesisStopped(utterance_id=utterance_id))

    async def _generate(self):
        """Stream the reply, from the agent if one is attached or the model if not."""
        if self.agent is not None:
            async for event in self._generate_via_agent():
                yield event
            return

        request = ChatRequest(
            model=self.config.llm_model,
            messages=self.history,
            system=self.config.system_prompt,
            # No ceiling. A spoken reply that stops mid-sentence is worse
            # here than anywhere else in Roost: there is no scrollback and no
            # stop reason to notice, just a voice that trails off. Length is
            # the prompt's job, and the prompt already asks for short answers.

            # Reasoning is silence. A model that thinks for eight seconds
            # before its first word of content cannot be rescued by
            # synthesising a sentence at a time, because there is no sentence
            # yet — the person just hears nothing and assumes it broke.
            # Measured on gemma4:12b: 1534 characters of thinking before 276
            # of answer. Servers that do not know the option ignore it.
            extra={'think': False, **self.config.llm_extra},
        )
        async for event in self.llm.stream(request):
            yield event

    async def _generate_via_agent(self):
        """Drive an agent session and speak its prose.

        Only assistant text is spoken. Reasoning, tool calls and tool output
        are not: reading a tool's output aloud is unbearable, and the approval
        prompts belong on screen where they can be read before being answered.
        """
        from roost.protocol.agent import TextDelta, TurnCompleted

        last = self.history[-1]
        text = ' '.join(b.text for b in last.content if isinstance(b, TextBlock))
        self.agent.submit(text)

        async for event in self.agent.events():
            if isinstance(event, TextDelta):
                yield StreamText(text=event.text)
            elif isinstance(event, TurnCompleted):
                break

    async def _speak(self, utterance_id: str, sentence: str) -> None:
        """Synthesise one sentence and stream it to the client."""
        try:
            async for chunk in self.tts.synthesize(
                sentence,
                model=self.config.tts_model,
                voice=self.config.tts_voice,
                fmt='pcm16',
                sample_rate=self.config.sample_rate,
            ):
                # Checked per chunk: cancellation can land between the model
                # producing a sentence and the client hearing it, and sending
                # audio for a cancelled utterance is what a barge-in is meant
                # to prevent.
                if self._utterance_id != utterance_id:
                    return
                await self.emit_audio(utterance_id, chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception('synthesis failed')
            await self.emit_event(VoiceError(message=f'speech synthesis failed: {exc}'))
            return

        self._spoken.append(sentence)

    # -- interruption -------------------------------------------------------

    async def _cancel_reply(self, reason: str) -> None:
        task, utterance_id = self._reply, self._utterance_id
        # Cleared before awaiting, so any synthesis still in flight sees the
        # id has moved on and stops sending.
        self._reply = None
        self._utterance_id = None

        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        if utterance_id:
            await self.emit_event(SynthesisCancelled(utterance_id=utterance_id, reason=reason))

    async def interrupt(self) -> None:
        if self.is_speaking:
            await self._cancel_reply('client')

    async def close(self) -> None:
        await self._cancel_reply('client')
        if self.agent is not None:
            await self.agent.close()
