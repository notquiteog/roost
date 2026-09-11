"""Deciding when someone is talking.

This runs on the server, not in the browser, and that is the whole reason
barge-in works. The browser knows you started talking; only the server knows
you started talking *while it was three words into a reply* and that the
synthesis it has already queued must be thrown away. Putting the detector
where that decision is made removes a round trip from the most latency-
sensitive moment in the conversation.

The detector is adaptive energy with a hangover, which is a deliberately
modest choice. A neural VAD is more accurate and costs a model load, a
tensor library and a few milliseconds per frame on hardware that is already
running a language model; this costs a mean and a comparison. The accuracy
that matters here is not "is this human speech" but "has the level been above
the room for long enough to be worth acting on", and an adaptive floor
answers that well in a quiet room and acceptably in a noisy one.

The floor tracks the room rather than assuming one. A fixed threshold is
tuned for whoever set it: it clips the quiet talker and never closes for the
person with a fan running. Tracking the observed minimum handles both, and
adapting *upward* far more slowly than downward is what stops a long
utterance from being absorbed into the floor and cut off mid-sentence.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum


class VadEvent(StrEnum):
    NONE = 'none'
    SPEECH_START = 'speech_start'
    SPEECH_END = 'speech_end'


@dataclass(slots=True)
class VadConfig:
    sample_rate: int = 16_000
    frame_ms: int = 20
    # How far above the room's own level counts as speech. 9 dB is about the
    # gap between a fan and a person in the same room; below ~6 the detector
    # opens on the fan, above ~14 it clips the start of quiet speech.
    threshold_db: float = 9.0
    # Consecutive speech frames before speech is declared. Three frames is
    # 60 ms — long enough to reject a keyboard click, short enough that the
    # first phoneme is not lost.
    start_frames: int = 3
    # Silence before an utterance is considered finished. Well under a second,
    # because this is the pause the person actually feels; much above 600 ms
    # and the assistant seems slow to notice you stopped.
    hangover_ms: int = 500
    # An utterance shorter than this is a cough or a door.
    min_speech_ms: int = 200
    # Hard stop, so a stuck microphone cannot buffer for ever.
    max_speech_ms: int = 30_000
    # Absolute floor. Digital silence has no meaningful level, and without
    # this the adaptive floor chases it down to -infinity and then opens on
    # the first bit of line noise.
    silence_floor_db: float = -55.0
    # Ceiling on the floor, and the reason a stream that opens mid-word is
    # survivable. Energy alone cannot tell a loud room from a person talking,
    # so when the only audio so far is a person talking, the floor is measured
    # from their voice and sits above them — and a detector deaf to the talker
    # never recovers, because recovering needs a frame it can hear. A room
    # louder than this is rare; being wrong about one costs a spurious short
    # utterance, which `min_speech_ms` already discards.
    max_floor_db: float = -35.0
    # How much recent audio the floor is measured over. Long enough that a
    # talker's own pauses fall inside it — a window with no silence in it has
    # nothing to measure the room against — and short enough to follow a room
    # that actually changes.
    floor_window_ms: int = 3_000
    # How long the detector may stay open before the floor is assumed to be
    # describing a room that is no longer there. Longer than anyone speaks in
    # one breath, and far shorter than `max_speech_ms`, which is a backstop
    # against a stuck microphone rather than something a turn should wait for.
    stuck_speech_ms: int = 3_000
    # Below this a frame is not quiet, it is absent: a stream that has opened
    # but is not yet delivering audio, or a muted input. Nothing about it
    # describes the room, so it is not allowed to describe the floor.
    no_signal_db: float = -80.0


def rms_dbfs(frame: bytes) -> float:
    """Level of one PCM16 frame, in dBFS."""
    if len(frame) < 2:
        return -100.0

    total = 0
    count = len(frame) // 2
    for i in range(0, count * 2, 2):
        # int.from_bytes rather than numpy: this runs on every 20 ms frame of
        # every call, and importing numpy for a sum of squares is not worth
        # the dependency at this size.
        sample = int.from_bytes(frame[i : i + 2], 'little', signed=True)
        total += sample * sample

    if count == 0:
        return -100.0
    mean = total / count
    if mean <= 0:
        return -100.0
    return 20.0 * math.log10(math.sqrt(mean) / 32768.0)


@dataclass
class Vad:
    config: VadConfig = field(default_factory=VadConfig)

    _floor: float = field(default=-50.0, init=False)
    _speech_run: int = field(default=0, init=False)
    _silence_run: int = field(default=0, init=False)
    _in_speech: bool = field(default=False, init=False)
    _speech_ms: int = field(default=0, init=False)
    # Levels of the recent frames that carried signal. The minimum of these
    # is the room.
    _levels: deque[float] = field(default_factory=deque, init=False)
    _primed: bool = field(default=False, init=False)
    _remeasured: bool = field(default=False, init=False)

    @property
    def frame_bytes(self) -> int:
        return int(self.config.sample_rate * self.config.frame_ms / 1000) * 2

    @property
    def _prime_frames(self) -> int:
        """Frames of real audio before the floor stops being taken as read."""
        return max(1, 200 // self.config.frame_ms)

    @property
    def speaking(self) -> bool:
        return self._in_speech

    def reset(self) -> None:
        """Forget the utterance but keep the room.

        The noise floor is the expensive thing to learn and does not change
        between utterances; discarding it would mean re-learning the room
        after every sentence. Keeping it is only safe because it is now
        measured continuously — a floor carried across a reset corrects
        itself within a few seconds if it turns out to be wrong, where a
        latched one stayed wrong for the length of the call.
        """
        self._speech_run = 0
        self._silence_run = 0
        self._in_speech = False
        self._speech_ms = 0
        self._remeasured = False

    def _observe(self, level: float) -> None:
        """Move the floor toward the quietest thing the room has done lately.

        The target is the minimum over a window rather than the level of one
        frame, which is what keeps a long utterance out of the floor: however
        loud the last second was, the window still holds the pause before it.
        That is the job the old `is_speech` gate was doing, and doing it this
        way removes the gate's deadlock — a floor low enough to read the room
        as speech used to be a floor that could never be corrected, because
        correcting it required a frame of silence it could no longer see.
        """
        cfg = self.config
        if level > cfg.no_signal_db:
            self._levels.append(level)
        while len(self._levels) > max(1, cfg.floor_window_ms // cfg.frame_ms):
            self._levels.popleft()

        if not self._levels:
            # Nothing but digital silence so far. Say nothing about the room
            # rather than guessing, and leave the floor where it is.
            return

        target = min(max(min(self._levels), cfg.silence_floor_db), cfg.max_floor_db)

        if not self._primed:
            # The opening moments, where adapting from a default means
            # misjudging the first thing anyone says. Take the room as read —
            # bounded by `max_floor_db`, so an opening frame of speech cannot
            # put the floor above the talker.
            self._floor = target
            self._primed = len(self._levels) >= self._prime_frames
            return

        # Downward quickly, so walking into a quiet room is picked up in a
        # second; upward very slowly, so a sustained noise raises the floor
        # over several seconds rather than one loud frame doing it.
        rate = 0.15 if target < self._floor else 0.01
        moved = self._floor + rate * (target - self._floor)
        self._floor = min(max(moved, cfg.silence_floor_db), cfg.max_floor_db)

    def push(self, frame: bytes) -> VadEvent:
        """Feed one frame. Returns a transition, or NONE."""
        cfg = self.config
        level = rms_dbfs(frame)

        if self._in_speech and self._speech_ms >= cfg.stuck_speech_ms and not self._remeasured:
            # Nobody speaks this long without a pause quiet enough to close
            # on, so this is the room, not a person — and the window still
            # holds the quieter room it used to be, which is what is keeping
            # the floor underneath it. Drop that history and measure again
            # from what is actually arriving. Once per utterance: clearing it
            # every frame would leave a window of one, and a floor that chases
            # the level rather than the minimum swallows real speech.
            self._levels.clear()
            self._remeasured = True

        self._observe(level)
        is_speech = level > self._floor + cfg.threshold_db

        if is_speech:
            self._speech_run += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            self._speech_run = 0

        if self._in_speech:
            self._speech_ms += cfg.frame_ms

            if self._speech_ms >= cfg.max_speech_ms:
                self.reset()
                return VadEvent.SPEECH_END

            if self._silence_run * cfg.frame_ms >= cfg.hangover_ms:
                # The hangover itself is not speech, so it does not count
                # toward the minimum-length test.
                spoken = self._speech_ms - self._silence_run * cfg.frame_ms
                self.reset()
                return VadEvent.SPEECH_END if spoken >= cfg.min_speech_ms else VadEvent.NONE

        elif self._speech_run >= cfg.start_frames:
            self._in_speech = True
            # Backdated, so the frames that triggered the start are part of the
            # utterance rather than being counted as having happened before it.
            self._speech_ms = self._speech_run * cfg.frame_ms
            self._silence_run = 0
            return VadEvent.SPEECH_START

        return VadEvent.NONE


class Ringbuffer:
    """Keeps the last N milliseconds of audio.

    Needed because speech is declared 60 ms after it began: without a little
    history the utterance sent to transcription starts partway into the first
    word, and 'delete the branch' arrives as 'lete the branch'.
    """

    def __init__(self, sample_rate: int = 16_000, ms: int = 500) -> None:
        self.capacity = int(sample_rate * ms / 1000) * 2
        self._buf = bytearray()

    def push(self, frame: bytes) -> None:
        self._buf.extend(frame)
        if len(self._buf) > self.capacity:
            del self._buf[: len(self._buf) - self.capacity]

    def take(self) -> bytes:
        out = bytes(self._buf)
        self._buf.clear()
        return out

    def clear(self) -> None:
        self._buf.clear()
