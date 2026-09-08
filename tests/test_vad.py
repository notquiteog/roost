"""VAD behaviour against synthetic audio.

Synthetic rather than recorded, because what is being tested is the state
machine — when it opens, when it closes, what it ignores — and that should be
reproducible without a fixture file.
"""

from __future__ import annotations

import math
import struct

from roost.voice.vad import Vad, VadConfig, VadEvent, rms_dbfs


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
