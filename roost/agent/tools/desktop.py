"""Controlling the desktop itself.

This is the weakest part of the whole safety story and it is worth being
plain about why.

In the browser, a click is graded by *reading the thing being clicked* — its
label, its name, its href. "Place your order" is recognisable as a purchase
before the click happens, and that check is what lets the agent be trusted
with a checkout page. On a raw desktop there is no such thing to read. There
is a bitmap and a pair of coordinates, and no amount of care turns those into
"this button charges your card".

Three things compensate, none of them as good:

**A click must state what it believes it is clicking.** The model passes a
label, and that label is graded exactly as a browser element would be. A model
has no incentive to misdescribe its own action, and the human sees both the
label and the coordinates in the prompt.

**Coordinates go stale, so a screenshot must be recent.** Clicking a position
worked out from a screenshot taken two minutes ago is how you click whatever
has since moved there. A click without a fresh screenshot is refused.

**No desktop click is ever a read.** The browser can safely auto-run clicks
under a permissive policy because it graded them first. Here the floor is
`execute`, always.

The honest summary: desktop autopilot is meaningfully less safe than browser
autopilot, and if the task can be done in the browser it should be.
"""

from __future__ import annotations

import base64
import time
from typing import Any

from roost.agent.tools.base import Assessment, Output, Tool, ToolContext, ToolError
from roost.agent.tools.browser import classify_click, classify_field
from roost.protocol.agent import Risk

# How old a screenshot may be and still be trusted for aiming a click. Short,
# because a notification sliding in is enough to move what is under a point.
STALE_AFTER = 45.0


class _DesktopTool(Tool):
    def __init__(self, state: dict[str, Any]) -> None:
        # Shared between the tools so a click can tell whether a screenshot
        # has been taken since the last thing that changed the screen.
        self.state = state


class DesktopScreenshotTool(_DesktopTool):
    name = 'desktop_screenshot'
    description = (
        'Take a picture of the whole screen. Do this before every click or drag: '
        'coordinates are worked out from what you see, and anything you saw more than '
        'a few seconds ago may have moved.'
    )
    input_schema = {'type': 'object', 'properties': {}}

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        return Assessment(risk=Risk.READ, summary='screenshot the desktop')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        import asyncio

        from roost.agent import desktop

        try:
            # Capture shells out and blocks; off the loop so a voice call in
            # the same process does not stutter.
            png = await asyncio.to_thread(desktop.capture)
            size = await asyncio.to_thread(desktop.screen_size)
        except desktop.DesktopUnavailable as exc:
            raise ToolError(str(exc)) from exc

        self.state['last_shot'] = time.monotonic()
        self.state['size'] = size

        encoded = base64.b64encode(png).decode()
        return Output(
            content=f'Screenshot taken: {size.width}x{size.height}. '
                    'Coordinates are in pixels from the top-left of the image below.',
            display={
                'image': encoded,
                'media_type': 'image/png',
                'width': size.width,
                'height': size.height,
            },
            # The same image, to the model. Without this it receives only the
            # sentence above and will describe a screen it has never seen.
            images=[(encoded, 'image/png')],
        )


class DesktopClickTool(_DesktopTool):
    name = 'desktop_click'
    description = (
        'Click at a point on the screen. You must take a screenshot first and work the '
        'coordinates out from it. Describe what you are clicking in `label` — that is what '
        'the person is shown when they are asked to approve it, so make it accurate.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'x': {'type': 'integer'},
            'y': {'type': 'integer'},
            'label': {
                'type': 'string',
                'description': 'What is at that point, as you read it on screen. Required.',
            },
            'button': {'type': 'string', 'description': "'left', 'right' or 'middle'. Default left."},
            'double': {'type': 'boolean'},
        },
        'required': ['x', 'y', 'label'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        x, y, label = args.get('x'), args.get('y'), (args.get('label') or '').strip()
        if not isinstance(x, int) or not isinstance(y, int):
            return Assessment(risk=Risk.EXECUTE, summary='', invalid='x and y must be integers')
        if not label:
            return Assessment(
                risk=Risk.EXECUTE, summary='',
                invalid='label is required — say what you are clicking, it is what the person is shown',
            )

        last = self.state.get('last_shot')
        if last is None:
            return Assessment(
                risk=Risk.EXECUTE, summary='',
                invalid='take a screenshot first; there is nothing to aim at',
            )
        age = time.monotonic() - last
        if age > STALE_AFTER:
            return Assessment(
                risk=Risk.EXECUTE, summary='',
                invalid=f'the last screenshot is {int(age)}s old. Take another before clicking.',
            )

        # Graded from what the model says is there, exactly as a browser
        # element would be. Weaker, and the best available.
        risk, why = classify_click({'label': label})
        # Never below execute: unlike the browser, nothing here was verified.
        if risk is Risk.WRITE:
            risk = Risk.EXECUTE
        return Assessment(risk=risk, summary=f'click {label!r} at ({x}, {y}) — {why}')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        import asyncio

        from roost.agent import desktop

        try:
            size = self.state.get('size') or await asyncio.to_thread(desktop.screen_size)
            pointer = await asyncio.to_thread(desktop.make_input, size)
        except desktop.DesktopUnavailable as exc:
            raise ToolError(str(exc)) from exc

        try:
            await asyncio.to_thread(pointer.move, args['x'], args['y'])
            await asyncio.sleep(0.08)
            await asyncio.to_thread(
                pointer.click, args.get('button', 'left'), bool(args.get('double'))
            )
        finally:
            await asyncio.to_thread(pointer.close)

        # The screen has almost certainly changed, so the old screenshot must
        # not be reused to aim the next click.
        self.state['last_shot'] = None
        return Output(
            content=f'Clicked ({args["x"]}, {args["y"]}). Take another screenshot to see what happened.',
            display={'x': args['x'], 'y': args['y'], 'label': args['label']},
        )


class DesktopTypeTool(_DesktopTool):
    name = 'desktop_type'
    description = (
        'Type text wherever the keyboard focus currently is. Click the field first. '
        'Say what the field is in `field`. You cannot type passwords, card numbers or '
        'one-time codes — ask the person to type those themselves.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'text': {'type': 'string'},
            'field': {'type': 'string', 'description': 'What you are typing into, as labelled on screen.'},
        },
        'required': ['text', 'field'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        if not args.get('text'):
            return Assessment(risk=Risk.WRITE, summary='', invalid='text is required')
        field = (args.get('field') or '').strip()
        if not field:
            return Assessment(risk=Risk.WRITE, summary='', invalid='field is required — say what you are typing into')

        risk, why = classify_field({'label': field, 'name': field})
        if risk is Risk.CREDENTIAL:
            # The value is kept out of the summary deliberately.
            return Assessment(risk=risk, summary=f'type into {field!r} — {why}')
        shown = args['text'] if len(args['text']) <= 50 else args['text'][:47] + '...'
        return Assessment(risk=risk, summary=f'type {shown!r} into {field!r}')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        import asyncio

        from roost.agent import desktop

        try:
            size = self.state.get('size') or await asyncio.to_thread(desktop.screen_size)
            pointer = await asyncio.to_thread(desktop.make_input, size)
        except desktop.DesktopUnavailable as exc:
            raise ToolError(str(exc)) from exc

        try:
            await asyncio.to_thread(pointer.type_text, args['text'])
        finally:
            await asyncio.to_thread(pointer.close)

        self.state['last_shot'] = None
        return Output(content=f'Typed {len(args["text"])} characters into {args["field"]!r}.')


class DesktopKeyTool(_DesktopTool):
    name = 'desktop_key'
    description = (
        "Press a key or a chord, such as 'enter', 'ctrl+s' or 'alt+tab'. "
        'Applies to whatever currently has focus.'
    )
    input_schema = {
        'type': 'object',
        'properties': {'combo': {'type': 'string', 'description': "e.g. 'ctrl+shift+t'"}},
        'required': ['combo'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        combo = (args.get('combo') or '').strip()
        if not combo:
            return Assessment(risk=Risk.EXECUTE, summary='', invalid='combo is required')
        return Assessment(risk=Risk.EXECUTE, summary=f'press {combo}')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        import asyncio

        from roost.agent import desktop

        try:
            size = self.state.get('size') or await asyncio.to_thread(desktop.screen_size)
            pointer = await asyncio.to_thread(desktop.make_input, size)
        except desktop.DesktopUnavailable as exc:
            raise ToolError(str(exc)) from exc

        try:
            await asyncio.to_thread(pointer.key, args['combo'])
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        finally:
            await asyncio.to_thread(pointer.close)

        self.state['last_shot'] = None
        return Output(content=f'Pressed {args["combo"]}.')


def desktop_tools() -> list[Tool]:
    state: dict[str, Any] = {}
    return [
        DesktopScreenshotTool(state),
        DesktopClickTool(state),
        DesktopTypeTool(state),
        DesktopKeyTool(state),
    ]
