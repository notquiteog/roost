"""Duplex behaviour: sentence-at-a-time synthesis, and barge-in."""

from __future__ import annotations

import asyncio
import struct

import pytest

from openmirror.protocol.voice import (
    SpeechStarted,
    SynthesisCancelled,
    SynthesisStarted,
    SynthesisStopped,
    TranscriptFinal,
)
from openmirror.providers.base import StreamDone, StreamText, Transcript
from openmirror.voice.pipeline import (
    VoiceConfig,
    VoiceSession,
    is_meaningful,
    pcm_to_wav,
    split_sentences,
)
from tests.test_vad import noise, tone


class FakeSTT:
    def __init__(self, text='delete the build directory'):
        self.text = text
        self.calls = 0
        self.seen: list[bytes] = []

    async def transcribe(self, audio, *, model, language=None):
        self.calls += 1
        self.seen.append(audio)
        return Transcript(text=self.text)


class SlowLLM:
    """Streams words with a delay, so a barge-in has something to land on."""

    def __init__(self, text: str, delay: float = 0.05):
        self.text = text
        self.delay = delay
        self.started = 0

    async def stream(self, req):
        self.started += 1
        for word in self.text.split(' '):
            await asyncio.sleep(self.delay)
            yield StreamText(text=word + ' ')
        yield StreamDone()


class FakeTTS:
    def __init__(self, delay: float = 0.02):
        self.delay = delay
        self.spoken: list[str] = []

    async def synthesize(self, text, *, model, voice, fmt='pcm16', sample_rate=16_000):
        self.spoken.append(text)
        for _ in range(3):
            await asyncio.sleep(self.delay)
            yield b'\x00\x01' * 160

    async def voices(self):
        return []


def build(llm, stt=None, tts=None):
    events: list = []
    audio: list[tuple[str, bytes]] = []

    async def emit_event(e):
        events.append(e)

    async def emit_audio(uid, chunk):
        audio.append((uid, chunk))

    session = VoiceSession(
        stt=stt or FakeSTT(),
        tts=tts or FakeTTS(),
        llm=llm,
        config=VoiceConfig(llm_model='test'),
        emit_event=emit_event,
        emit_audio=emit_audio,
    )
    return session, events, audio


# -- helpers ----------------------------------------------------------------


def test_wav_header_is_well_formed():
    pcm = b'\x00\x01' * 100
    wav = pcm_to_wav(pcm, 16_000)
    assert wav[:4] == b'RIFF' and wav[8:12] == b'WAVE'
    assert struct.unpack('<I', wav[4:8])[0] == 36 + len(pcm)
    assert struct.unpack('<I', wav[40:44])[0] == len(pcm)
    assert struct.unpack('<I', wav[24:28])[0] == 16_000


def test_sentence_splitting():
    done, rest = split_sentences('One. Two! Three? And a frag')
    assert done == ['One.', 'Two!', 'Three?']
    assert rest == 'And a frag'


def test_abbreviation_is_not_a_sentence_end():
    # 'Dr.' must not end a sentence, so the first split lands after 'it.'
    done, rest = split_sentences('See Dr. Smith about it. Then go. ')
    assert done == ['See Dr. Smith about it.', 'Then go.'], done
    assert rest.strip() == ''


def test_an_unterminated_tail_is_held_back():
    """Mid-stream, a sentence with nothing after it may not be finished yet.

    Holding it is what stops 'Delete the file.' being spoken as a complete
    thought when the next token turns out to be 'No, wait'. The caller flushes
    the remainder when the model stops.
    """
    done, rest = split_sentences('Done. Then go.')
    assert done == ['Done.']
    assert rest == 'Then go.'


def test_filler_is_not_worth_answering():
    assert is_meaningful('delete the branch')
    assert not is_meaningful('uh')
    assert not is_meaningful('  ')
    assert not is_meaningful('you')      # whisper's standard hallucination on silence


# -- the pipeline -----------------------------------------------------------


@pytest.mark.asyncio
async def test_speaks_each_sentence_as_it_completes():
    """Synthesis must not wait for the model to finish."""
    tts = FakeTTS()
    session, events, audio = build(SlowLLM('First sentence. Second sentence. Third.'), tts=tts)

    await session.say('hello')
    await asyncio.wait_for(session._reply, timeout=10)

    assert tts.spoken == ['First sentence.', 'Second sentence.', 'Third.']
    assert audio and all(uid == audio[0][0] for uid, _ in audio)
    assert any(isinstance(e, SynthesisStarted) for e in events)
    assert any(isinstance(e, SynthesisStopped) for e in events)


@pytest.mark.asyncio
async def test_audio_drives_a_full_turn():
    stt = FakeSTT('what is the time')
    session, events, audio = build(SlowLLM('It is four.', delay=0.01), stt=stt)

    # A pause to learn the room, then speech, then enough silence to close.
    await session.feed(noise(600, amplitude=0.002))
    await session.feed(tone(800, amplitude=0.3))
    await session.feed(noise(900, amplitude=0.002))

    assert any(isinstance(e, SpeechStarted) for e in events)
    final = [e for e in events if isinstance(e, TranscriptFinal)]
    assert final and final[0].submitted and final[0].text == 'what is the time'

    await asyncio.wait_for(session._reply, timeout=10)
    assert audio


@pytest.mark.asyncio
async def test_barge_in_cancels_the_reply():
    """Talking over the assistant must stop it and stop the audio."""
    llm = SlowLLM('One. Two. Three. Four. Five. Six. Seven. Eight.', delay=0.06)
    session, events, audio = build(llm)

    await session.feed(noise(600, amplitude=0.002))   # learn the room
    await session.say('go')
    await asyncio.sleep(0.35)                          # let it get going
    assert session.is_speaking

    before = len(audio)
    await session.feed(tone(200, amplitude=0.35))      # interrupt
    await asyncio.sleep(0.3)

    cancels = [e for e in events if isinstance(e, SynthesisCancelled)]
    assert cancels and cancels[0].reason == 'barge_in'
    assert not session.is_speaking
    # Nothing further was sent for the cancelled utterance.
    assert len(audio) == before or all(uid != cancels[0].utterance_id for uid, _ in audio[before:])


@pytest.mark.asyncio
async def test_interrupted_reply_records_only_what_was_heard():
    """History must not claim it said things that were cut off."""
    llm = SlowLLM('Alpha. Beta. Gamma. Delta. Epsilon. Zeta.', delay=0.06)
    session, events, audio = build(llm)

    await session.feed(noise(600, amplitude=0.002))
    await session.say('go')
    await asyncio.sleep(0.3)
    await session.interrupt()

    assistant = [m for m in session.history if m.role == 'assistant']
    assert assistant
    text = assistant[-1].content[0].text
    assert '[interrupted]' in text
    # Whatever it claims to have said must actually have been synthesised.
    for sentence in ('Alpha.', 'Beta.', 'Gamma.', 'Delta.', 'Epsilon.', 'Zeta.'):
        if sentence in text:
            assert sentence in session.tts.spoken, f'claimed {sentence!r} without speaking it'


@pytest.mark.asyncio
async def test_mute_stops_feeding_the_detector():
    session, events, audio = build(SlowLLM('hi'))
    await session.mute(True)
    await session.feed(tone(800, amplitude=0.4))
    assert not any(isinstance(e, SpeechStarted) for e in events)


@pytest.mark.asyncio
async def test_silence_transcribed_as_filler_is_not_answered():
    llm = SlowLLM('should never run')
    session, events, audio = build(llm, stt=FakeSTT('you'))

    await session.feed(noise(600, amplitude=0.002))
    await session.feed(tone(800, amplitude=0.3))
    await session.feed(noise(900, amplitude=0.002))

    final = [e for e in events if isinstance(e, TranscriptFinal)]
    assert final and not final[0].submitted
    assert llm.started == 0


@pytest.mark.asyncio
async def test_a_discarded_cough_stops_the_capture():
    """A burst too short to be speech must close the capture, not orphan it.

    The detector drops an utterance under `min_speech_ms` by reporting
    nothing at all, which used to leave the session capturing for the rest of
    the call: every later frame appended to an utterance nobody would send,
    and none of them reaching the pre-roll — so the next real sentence
    arrived with its first word missing, which is the one thing the pre-roll
    buffer exists to prevent.
    """
    stt = FakeSTT('what is the time')
    session, events, _ = build(SlowLLM('It is four.', delay=0.01), stt=stt)

    await session.feed(noise(600, amplitude=0.002))
    await session.feed(tone(120, amplitude=0.4))       # a cough, under min_speech_ms
    await session.feed(noise(900, amplitude=0.002))

    assert not [e for e in events if isinstance(e, TranscriptFinal)]
    assert not session._capturing, 'still capturing after a discarded burst'
    assert not session._utterance, 'audio accumulating into an utterance nobody will send'

    # And the next real sentence still gets its pre-roll.
    await session.feed(noise(600, amplitude=0.002))
    await session.feed(tone(800, amplitude=0.3))
    await session.feed(noise(900, amplitude=0.002))

    final = [e for e in events if isinstance(e, TranscriptFinal)]
    assert final and final[0].submitted
    # Pre-roll plus the 800 ms burst plus the hangover, not a whole call of noise.
    assert len(stt.seen[0]) < 2 * 16_000 * 3, 'utterance carries audio from before it began'
    await asyncio.wait_for(session._reply, timeout=10)
