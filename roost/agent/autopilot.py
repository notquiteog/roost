"""Watching it work, instead of talking to it.

The other modes are conversations: you say a thing, it does a thing, you read
what it did. This one is a *goal* and a window. You describe the outcome, and
then the only interface is a live view of a screen it is driving and a line of
text saying what it is doing — no composer, nothing to answer, nothing to
click unless it stops and asks.

That is a genuinely different thing to build, and two decisions make it
work rather than merely look impressive.

**It runs on a stage of its own by default.** The whole idea collapses if
watching it means giving up your computer: an agent booking a hotel over the
top of your cursor is not autonomy, it is being locked out of your own
machine. So autopilot asks for a virtual stage — its own X server, its own
pointer — and the live view is a video of that. You keep working. If no
virtual stage can be made, it says so *before* starting rather than quietly
taking the mouse.

**Autonomy is bounded by the same approval policy as everything else.** There
is no separate "autopilot mode" in the policy and there must not be: the
temptation to add one is exactly how an agent ends up with permissions nobody
granted it in a mode nobody was watching. What autopilot changes is the
*prompt* — it is told to finish rather than to check in — and the operator
chooses the mode as they always do. `trusted` is the honest default for it;
`unrestricted` is available and says what it is.

The frame stream is deliberately separate from the event log. Sessions replay
their events to a reattaching client, and a screenshot every second would fill
that log with megabytes of stale pictures — a client reattaching after lunch
wants to know what happened, not to watch an hour of video first.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Frames are for a person watching, not for the model — the model takes its own
# screenshots when it needs to look. So this is tuned for "smooth enough to
# follow" rather than for detail: a wide JPEG at a modest rate, which is a few
# hundred kilobytes a second and looks like a screen recording.
FRAME_WIDTH = 1280
FRAME_QUALITY = 55
FRAME_INTERVAL = 0.7


AUTOPILOT_PROMPT = """You are working on your own. Nobody is going to answer you between now \
and the end of this task, so plan for that.

Keep going until the goal is met or you are genuinely stuck. Do not stop to confirm a step you \
already know how to take, do not report progress and wait, and do not ask whether to continue — \
there is nobody at the keyboard. If you finish, say what you did and stop.

Someone is watching a live picture of the screen you are working on, and nothing else. They \
cannot see your reasoning and they are not reading a transcript. So before each action, say in \
one short sentence what you are about to do — that line is the whole of what they get.

Work on the screen you have been given. Take a screenshot, look at it, act, take another. Never \
work out a coordinate from memory or from an earlier picture: things move, and a click aimed at \
where a button used to be lands on whatever is there now.

If you become genuinely stuck — a login you cannot complete, a page that will not load, a choice \
only the person can make — stop and use ask_user. It suspends everything until they answer, which \
is correct: guessing at an irreversible step on someone's behalf is worse than waiting.

Anything that spends money stops for a human however this task was phrased, and being unattended \
does not change that. It makes it more important."""


@dataclass(slots=True)
class Run:
    """One autopilot task."""

    id: str
    goal: str
    session_id: str
    started_at: float = field(default_factory=time.time)
    # Set when the stage could not be a private one, so the UI can say — in
    # words, before anything moves — that this will be using the real mouse.
    shares_pointer: bool = False
    stage: str = 'virtual'
    watchers: int = 0

    def to_json(self) -> dict[str, object]:
        return {
            'id': self.id,
            'goal': self.goal,
            'session': self.session_id,
            'elapsed': int(time.time() - self.started_at),
            'stage': self.stage,
            'shares_pointer': self.shares_pointer,
            'watchers': self.watchers,
        }


def encode_frame(png: bytes, *, width: int = FRAME_WIDTH, quality: int = FRAME_QUALITY) -> bytes:
    """A screen capture, small enough to send several times a second.

    JPEG rather than PNG, and scaled down: a 2560-wide PNG screenshot is about
    two megabytes and a 1280-wide JPEG of the same thing is about eighty
    kilobytes. For reading a screen the difference is invisible; for streaming
    it is the difference between working and not.
    """
    from PIL import Image

    with Image.open(io.BytesIO(png)) as image:
        image = image.convert('RGB')
        if image.width > width:
            height = round(image.height * width / image.width)
            image = image.resize((width, height), Image.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format='JPEG', quality=quality, optimize=True)
        return buf.getvalue()


async def frames(stage, *, interval: float = FRAME_INTERVAL) -> AsyncIterator[bytes]:
    """A stream of JPEG frames from a stage, for as long as anyone is watching.

    Captures are taken off the event loop — grabbing a screen shells out on
    Wayland and blocks on X — and a failed capture is skipped rather than
    ending the stream. A dropped frame is a flicker; an ended stream is a
    person who thinks the agent has stopped.
    """
    consecutive_failures = 0
    while True:
        started = time.monotonic()
        try:
            png = await asyncio.to_thread(stage.capture)
            yield await asyncio.to_thread(encode_frame, png)
            consecutive_failures = 0
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            consecutive_failures += 1
            log.debug('frame capture failed: %s', exc)
            if consecutive_failures >= 10:
                # Ten in a row is not a flicker: the display has gone. Ending
                # the stream lets the client say so instead of showing a frozen
                # picture that looks like an agent thinking hard.
                log.warning('giving up on the frame stream after %d failures', consecutive_failures)
                return

        # Measured from the start of the capture, so a slow grab does not add
        # to the interval and turn 1.4 frames a second into 0.7.
        elapsed = time.monotonic() - started
        await asyncio.sleep(max(0.05, interval - elapsed))


class Autopilot:
    """The live runs, by id."""

    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}

    def start(self, run: Run) -> Run:
        self.runs[run.id] = run
        return run

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)

    def list(self) -> list[dict[str, object]]:
        return [r.to_json() for r in self.runs.values()]

    def forget(self, run_id: str) -> None:
        self.runs.pop(run_id, None)


autopilot = Autopilot()
