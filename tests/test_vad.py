"""VAD behaviour against synthetic audio.

Synthetic rather than recorded, because what is being tested is the state
machine — when it opens, when it closes, what it ignores — and that should be
reproducible without a fixture file.
"""

from __future__ import annotations

import math
import struct

from openmirror.voice.vad import Vad, VadConfig, VadEvent, rms_dbfs


def tone(ms: int, *, amplitude: float, rate: int = 16_000, freq: float = 220.0) -> bytes:
    n = int(rate * ms / 1000)
    return b''.join(
        struct.pack('<h', int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / rate))) for i in range(n)
    )


def noise(ms: int, *, amplitude: float, rate: int = 16_000) -> bytes:
    """Deterministic pseudo-noise, so a failure is reproducible."""
    n = int(rate * ms / 1000)
    out = []
    state = 12345
    for _ in range(n):
        state = (1103515245 * state + 12345) % (2**31)
        out.append(struct.pack('<h', int((state / 2**31 - 0.5) * 2 * amplitude * 32767)))
    return b''.join(out)


def feed(vad: Vad, audio: bytes) -> list[tuple[int, VadEvent]]:
    """Push audio frame by frame, returning (ms offset, event) for transitions."""
    size = vad.frame_bytes
    events = []
    for i in range(0, len(audio) - size + 1, size):
        ev = vad.push(audio[i : i + size])
        if ev is not VadEvent.NONE:
            events.append((i // size * vad.config.frame_ms, ev))
    return events


def test_level_measurement():
    assert rms_dbfs(b'\x00\x00' * 160) < -90
    loud = rms_dbfs(tone(20, amplitude=0.5))
    quiet = rms_dbfs(tone(20, amplitude=0.05))
    assert -12 < loud < -6      # 0.5 amplitude sine is about -9 dBFS
    assert quiet < loud - 15


def test_detects_one_utterance():
    vad = Vad(VadConfig())
    audio = noise(600, amplitude=0.002) + tone(1200, amplitude=0.3) + noise(1000, amplitude=0.002)
    events = feed(vad, audio)

    kinds = [e for _, e in events]
    assert kinds == [VadEvent.SPEECH_START, VadEvent.SPEECH_END]

    start_ms = events[0][0]
    end_ms = events[1][0]
    # Opens shortly after the tone begins at 600 ms, and closes about a
    # hangover after it stops at 1800 ms.
    assert 600 <= start_ms <= 700, start_ms
    assert 1800 <= end_ms <= 2400, end_ms


def test_ignores_a_click():
    """A single short transient must not open the detector."""
    vad = Vad(VadConfig())
    audio = noise(600, amplitude=0.002) + tone(40, amplitude=0.4) + noise(800, amplitude=0.002)
    assert [e for _, e in feed(vad, audio)] == []


def test_hangover_bridges_a_pause_between_words():
    """A gap shorter than the hangover stays inside one utterance."""
    vad = Vad(VadConfig())
    audio = (
        noise(600, amplitude=0.002)
        + tone(500, amplitude=0.3)
        + noise(200, amplitude=0.002)     # pause between words, under the 500 ms hangover
        + tone(500, amplitude=0.3)
        + noise(1000, amplitude=0.002)
    )
    kinds = [e for _, e in feed(vad, audio)]
    assert kinds == [VadEvent.SPEECH_START, VadEvent.SPEECH_END], kinds


def test_adapts_to_a_noisy_room():
    """Speech is still found when the room is far louder than digital silence."""
    vad = Vad(VadConfig())
    room = 0.02
    audio = noise(1500, amplitude=room) + tone(1000, amplitude=0.35) + noise(1200, amplitude=room)
    kinds = [e for _, e in feed(vad, audio)]
    assert kinds == [VadEvent.SPEECH_START, VadEvent.SPEECH_END], kinds


def test_steady_noise_alone_never_opens_it():
    vad = Vad(VadConfig())
    assert [e for _, e in feed(vad, noise(4000, amplitude=0.02))] == []


def test_stuck_microphone_is_cut_off():
    vad = Vad(VadConfig(max_speech_ms=1000))
    audio = noise(300, amplitude=0.002) + tone(5000, amplitude=0.3)
    kinds = [e for _, e in feed(vad, audio)]
    assert VadEvent.SPEECH_END in kinds


SILENCE_FRAME = b'\x00\x00' * 320   # one 20 ms frame of digital silence at 16 kHz


def test_opening_frame_of_digital_silence_does_not_wedge_the_floor():
    """The mic stream is live before audio flows, and the room is not quiet.

    The first frames carry no signal at all, so nothing about them describes
    the room. If they set the floor, it pins to the absolute minimum, every
    later frame of room tone reads as speech, and the utterance never closes.
    """
    vad = Vad(VadConfig())
    audio = (
        SILENCE_FRAME * 5
        + noise(1500, amplitude=0.02)      # fan and mic hiss, well above -55 dBFS
        + tone(1000, amplitude=0.35)
        + noise(1500, amplitude=0.02)
    )
    events = feed(vad, audio)
    kinds = [e for _, e in events]

    assert VadEvent.SPEECH_END in kinds, f'utterance never closed: {events}'
    start_ms = next(ms for ms, e in events if e is VadEvent.SPEECH_START)
    end_ms = next(ms for ms, e in events if e is VadEvent.SPEECH_END)
    # The burst runs 1600..2600 ms; it must close within a hangover of that.
    assert 1600 <= start_ms <= 1800, start_ms
    assert 2600 <= end_ms <= 3200, end_ms


def test_room_tone_alone_never_opens_it_after_a_silent_start():
    """The inverse of the wedge: no burst at all, so nothing may be reported."""
    vad = Vad(VadConfig())
    audio = SILENCE_FRAME * 5 + noise(4000, amplitude=0.02)
    kinds = [e for _, e in feed(vad, audio)]
    assert VadEvent.SPEECH_START not in kinds, kinds


def test_stream_opening_mid_word_does_not_swallow_the_utterance():
    """The floor pinned too high is the same failure pointing the other way.

    A stream that opens partway through a word describes speech, not the
    room. Taken as the floor it sits above the talker, and an utterance with
    no pause in it for the floor to correct against is never reported.
    """
    vad = Vad(VadConfig())
    audio = tone(20, amplitude=0.9) + tone(3000, amplitude=0.2) + noise(1000, amplitude=0.002)
    events = feed(vad, audio)
    kinds = [e for _, e in events]

    assert kinds == [VadEvent.SPEECH_START, VadEvent.SPEECH_END], events
    # Opened near the top of the word rather than a second into it.
    assert events[0][0] <= 200, events[0][0]


def test_a_loud_first_frame_does_not_clip_the_first_utterance():
    """The gentler version: a quiet talker after an opening overload."""
    vad = Vad(VadConfig())
    audio = (
        tone(20, amplitude=0.9)
        + noise(400, amplitude=0.002)
        + tone(1000, amplitude=0.06)
        + noise(1500, amplitude=0.002)
    )
    events = feed(vad, audio)
    kinds = [e for _, e in events]
    assert kinds == [VadEvent.SPEECH_START, VadEvent.SPEECH_END], events
    # Speech starts at 420 ms. The pre-roll buffer covers 400 ms, so opening
    # later than that loses the first word outright.
    assert events[0][0] <= 520, events[0][0]


def test_floor_recovers_within_a_turn_not_within_a_call():
    """However the floor is wrong, it must correct in seconds, not stay wrong.

    This is the property the wedge violated: being wrong is survivable, being
    unable to become right is not.
    """
    vad = Vad(VadConfig())
    feed(vad, SILENCE_FRAME * 5 + noise(3000, amplitude=0.02))
    vad.reset()
    # Two utterances in the same room, after the bad start.
    audio = tone(800, amplitude=0.35) + noise(1200, amplitude=0.02)
    kinds = [e for _, e in feed(vad, audio)]
    assert kinds == [VadEvent.SPEECH_START, VadEvent.SPEECH_END], kinds
    kinds = [e for _, e in feed(vad, audio)]
    assert kinds == [VadEvent.SPEECH_START, VadEvent.SPEECH_END], kinds


def test_a_room_that_gets_louder_mid_call_does_not_wedge_it():
    """The floor is measured from a room; when the room changes, so must it.

    A fan or an air conditioner coming on partway through a call is a step
    change in the noise, and the frames from before it are no longer a
    description of anything. Held as the floor they read the new room as one
    endless utterance — the same wedge as a silent opening frame, arrived at
    from the other direction.
    """
    vad = Vad(VadConfig())
    audio = (
        noise(1000, amplitude=0.001)       # quiet room
        + noise(6000, amplitude=0.02)      # the fan comes on, and stays on
        + tone(800, amplitude=0.3)
        + noise(1500, amplitude=0.02)
    )
    events = feed(vad, audio)
    ends = [ms for ms, e in events if e is VadEvent.SPEECH_END]

    assert ends, f'never closed once the room changed: {events}'
    # Whatever it opened on when the fan started must close well inside the
    # 30 s stuck-microphone backstop, which is far too long to feel like a turn.
    assert ends[0] <= 8000, ends[0]
    # And the real burst at 7000..7800 ms is still found and closed.
    assert any(7000 <= ms <= 7300 for ms, e in events if e is VadEvent.SPEECH_START), events
    assert any(7800 <= ms <= 8600 for ms, e in events if e is VadEvent.SPEECH_END), events
