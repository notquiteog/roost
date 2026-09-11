"""The frame path: what capture hands over, and what the stream does with it.

The point of the contract under test is that nothing on the hot path encodes
a picture it is going to throw away. A stage that can hand over pixels does;
`encode_frame` produces exactly one JPEG from them; and the PNG that other
callers — the screenshot tool, the desktop tools — rely on is still there,
unchanged, for the callers that actually want a file.
"""

from __future__ import annotations

import asyncio
import io

import pytest

from openmirror.agent.autopilot import FRAME_WIDTH, _grab, encode_frame, frames
from openmirror.agent.stage import RawFrame, Rect, Stage

Image = pytest.importorskip('PIL.Image', reason='the frame path needs pillow')


def a_picture(width: int, height: int):
    """Something with structure in it, so a resize has something to lose."""
    image = Image.new('RGB', (width, height))
    px = image.load()
    for y in range(height):
        for x in range(width):
            px[x, y] = ((x * 7) % 256, (y * 11) % 256, ((x ^ y) * 3) % 256)
    return image


def png_of(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


class PngOnlyStage(Stage):
    """A stage from before `capture_raw` existed: it can only make a PNG."""

    kind = 'virtual'
    shares_pointer = False

    def __init__(self, width=640, height=400):
        self.size = (width, height)
        self.captures = 0

    def rect(self):
        return Rect(0, 0, *self.size)

    def capture(self) -> bytes:
        self.captures += 1
        return png_of(a_picture(*self.size))

    def move(self, x, y): ...
    def click(self, button='left', double=False): ...
    def type_text(self, text): ...
    def key(self, combo): ...
    def scroll(self, amount): ...


class RawStage(PngOnlyStage):
    """A stage whose grabber returns pixels, as `mss` does."""

    def __init__(self, width=640, height=400):
        super().__init__(width, height)
        self.raw_captures = 0

    def capture_raw(self) -> RawFrame:
        self.raw_captures += 1
        image = a_picture(*self.size)
        return RawFrame(image.tobytes(), image.width, image.height)


# -- what a RawFrame is -----------------------------------------------------


def test_a_raw_frame_is_what_pillow_expects_frombytes_to_be():
    """The whole saving rests on this: the buffer needs no reinterpreting."""
    original = a_picture(64, 48)
    frame = RawFrame(original.tobytes(), 64, 48)

    assert frame.size == (64, 48)
    assert frame.mode == 'RGB'
    assert len(frame.data) == 64 * 48 * 3
    restored = Image.frombytes(frame.mode, frame.size, frame.data)
    assert restored.tobytes() == original.tobytes()


# -- encode_frame takes either -----------------------------------------------


def test_encode_frame_takes_raw_pixels():
    jpeg = encode_frame(RawFrame(a_picture(80, 60).tobytes(), 80, 60))
    with Image.open(io.BytesIO(jpeg)) as out:
        assert out.format == 'JPEG'
        assert out.size == (80, 60)


def test_encode_frame_still_takes_png_bytes():
    """Kept deliberately: a caller holding a screenshot should not have to
    decode it itself just to hand it over."""
    jpeg = encode_frame(png_of(a_picture(80, 60)))
    with Image.open(io.BytesIO(jpeg)) as out:
        assert out.format == 'JPEG'
        assert out.size == (80, 60)


def test_both_routes_produce_the_same_frame():
    """Raw is a shortcut, not a different picture."""
    image = a_picture(300, 200)
    from_raw = encode_frame(RawFrame(image.tobytes(), 300, 200))
    from_png = encode_frame(png_of(image))
    assert from_raw == from_png


def test_a_wide_frame_is_scaled_down_and_a_narrow_one_is_left_alone():
    image = a_picture(FRAME_WIDTH * 2, FRAME_WIDTH)
    with Image.open(io.BytesIO(encode_frame(RawFrame(image.tobytes(), *image.size)))) as out:
        # Aspect kept, so a watcher sees the shape of the screen it is watching.
        assert out.size == (FRAME_WIDTH, FRAME_WIDTH // 2)

    small = a_picture(320, 240)
    with Image.open(io.BytesIO(encode_frame(RawFrame(small.tobytes(), 320, 240)))) as out:
        assert out.size == (320, 240)


def test_quality_is_a_lever_the_caller_holds():
    raw = RawFrame(a_picture(400, 300).tobytes(), 400, 300)
    assert len(encode_frame(raw, quality=20)) < len(encode_frame(raw, quality=90))


# -- the default capture_raw -------------------------------------------------


def test_a_stage_that_only_makes_pngs_still_yields_raw_pixels():
    """The base class decodes its own PNG rather than refusing. Slow, but a
    stage without a fast path must still stream."""
    stage = PngOnlyStage(120, 90)
    frame = stage.capture_raw()

    assert isinstance(frame, RawFrame)
    assert frame.size == (120, 90)
    assert frame.data == a_picture(120, 90).tobytes()


def test_a_raw_stage_does_not_go_near_png():
    """The bug this change exists for: a capture that compresses a frame the
    encoder is about to decompress again."""
    stage = RawStage(120, 90)
    frame = stage.capture_raw()

    assert stage.captures == 0, 'capture_raw fell back to the PNG path'
    assert stage.raw_captures == 1
    assert frame.data == a_picture(120, 90).tobytes()


def test_capture_still_returns_a_png_for_the_callers_that_want_one():
    """The desktop screenshot tool and the Wayland crop both hold PNG bytes,
    so adding a raw path must not have taken theirs away."""
    for stage in (PngOnlyStage(64, 64), RawStage(64, 64)):
        png = stage.capture()
        assert png[:8] == b'\x89PNG\r\n\x1a\n'
        with Image.open(io.BytesIO(png)) as im:
            assert im.size == (64, 64)


# -- what the stream actually calls ------------------------------------------


def test_the_stream_asks_for_raw_pixels():
    """Every real stage answers with pixels — either its grabber's, or the
    base class's own decode. Either way the encoder never sees a PNG."""
    raw = RawStage()
    assert isinstance(_grab(raw), RawFrame)
    assert raw.captures == 0

    png_only = PngOnlyStage()
    assert isinstance(_grab(png_only), RawFrame)
    assert png_only.captures == 1


def test_a_duck_typed_stage_with_no_capture_raw_still_streams():
    """Fakes in other tests implement `capture` and nothing else, and so might
    a stage written elsewhere. They stream, slowly, rather than erroring."""

    class JustCapture:
        def capture(self):
            return png_of(a_picture(32, 32))

    assert isinstance(_grab(JustCapture()), bytes)


@pytest.mark.asyncio
async def test_frames_yields_jpegs_from_a_raw_stage():
    stage = RawStage(200, 100)
    out = []

    async def drain():
        async for frame in frames(stage, interval=0.0):
            out.append(frame)
            if len(out) == 3:
                return

    await asyncio.wait_for(drain(), timeout=10)

    assert len(out) == 3
    assert stage.captures == 0, 'the hot path encoded a PNG per frame'
    assert stage.raw_captures == 3
    for frame in out:
        with Image.open(io.BytesIO(frame)) as im:
            assert im.format == 'JPEG'
            assert im.size == (200, 100)


@pytest.mark.asyncio
async def test_a_failing_capture_is_a_dropped_frame_not_a_dead_stream():
    """A flicker is survivable; a stream that ends looks like an agent that
    has stopped."""

    class Flaky(RawStage):
        def capture_raw(self):
            self.raw_captures += 1
            if self.raw_captures % 2:
                raise RuntimeError('the display blinked')
            return super().capture_raw()

    stage = Flaky(64, 64)
    stage.raw_captures = 0
    out = []

    async def drain():
        async for frame in frames(stage, interval=0.0):
            out.append(frame)
            if len(out) == 2:
                return

    await asyncio.wait_for(drain(), timeout=10)
    assert len(out) == 2
