"""End-to-end exercise of the agent loop against a scripted provider.

A fake provider rather than a live model: the loop's behaviour under denial,
interruption and a suspended question is exactly what must not depend on
whether a model happened to feel cooperative that morning.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from openmirror.agent.approval import Mode
from openmirror.agent.runtime import build_session
from openmirror.protocol.agent import (
    QuestionAsked,
    Risk,
    ToolCompleted,
    ToolDenied,
    ToolProposed,
    TurnCompleted,
)
from openmirror.providers.base import ChatRequest, StreamDone, StreamText, StreamToolUse


class ScriptedProvider:
    """Replays a list of exchanges, one per model round trip."""

    def __init__(self, script: list[list]) -> None:
        self.script = script
        self.calls = 0
        self.seen: list[ChatRequest] = []

    async def stream(self, req: ChatRequest):
        self.seen.append(req)
        events = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        for event in events:
            yield event

    async def models(self):
        return [{'id': 'scripted'}]


async def drain(session, until=TurnCompleted, timeout=10):  # noqa: ASYNC109 - test helper
    """Collect events until the turn ends."""
    out = []

    async def pump():
        async for event in session.events():
            out.append(event)
            if isinstance(event, until):
                return

    await asyncio.wait_for(pump(), timeout=timeout)
    return out


def make(tmp: Path, script, mode=Mode.ASK):
    return build_session(root=tmp, provider=ScriptedProvider(script), model='scripted', mode=mode)


@pytest.mark.asyncio
async def test_writes_a_file_after_approval():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        session = make(
            root,
            [
                [
                    StreamText(text='Creating it now.'),
                    StreamToolUse(id='c1', name='write_file', input={'path': 'hello.txt', 'content': 'hi\n'}),
                    StreamDone(stop_reason='tool_use'),
                ],
                [StreamText(text='Done.'), StreamDone()],
            ],
        )
        await session.start()
        session.submit('create hello.txt')

        events = []

        async def pump():
            async for event in session.events():
                events.append(event)
                # A new file is a WRITE, which `ask` mode confirms.
                if isinstance(event, ToolProposed) and event.needs_approval:
                    session.approve(event.call.id)
                if isinstance(event, TurnCompleted):
                    return

        await asyncio.wait_for(pump(), timeout=10)

        assert (root / 'hello.txt').read_text() == 'hi\n'
        assert any(isinstance(e, ToolCompleted) and e.result.ok for e in events)
        assert [e for e in events if isinstance(e, TurnCompleted)][0].stop_reason == 'end_turn'


@pytest.mark.asyncio
async def test_denial_reaches_the_model_as_a_result():
    with tempfile.TemporaryDirectory() as tmp:
        session = make(
            Path(tmp),
            [
                [
                    StreamToolUse(id='c1', name='shell', input={'command': 'rm -rf /'}),
                    StreamDone(stop_reason='tool_use'),
                ],
                [StreamText(text='Understood, I will not.'), StreamDone()],
            ],
        )
        await session.start()
        session.submit('clean up')

        events = []

        async def pump():
            async for event in session.events():
                events.append(event)
                if isinstance(event, ToolProposed) and event.needs_approval:
                    session.deny(event.call.id, 'absolutely not')
                if isinstance(event, TurnCompleted):
                    return

        await asyncio.wait_for(pump(), timeout=10)

        assert any(isinstance(e, ToolDenied) for e in events)
        # The denial must have gone back to the model, and the turn must have
        # continued rather than ending on it.
        last = session.provider.seen[-1]
        flat = str([b for m in last.messages for b in m.content])
        assert 'declined' in flat
        assert session.provider.calls == 2


@pytest.mark.asyncio
async def test_read_only_mode_refuses_without_asking():
    with tempfile.TemporaryDirectory() as tmp:
        session = make(
            Path(tmp),
            [
                [
                    StreamToolUse(id='c1', name='shell', input={'command': 'touch x'}),
                    StreamDone(stop_reason='tool_use'),
                ],
                [StreamText(text='ok'), StreamDone()],
            ],
            mode=Mode.READ_ONLY,
        )
        await session.start()
        session.submit('make a file')
        events = await drain(session)

        assert any(isinstance(e, ToolDenied) for e in events)
        # No human was asked: read-only refuses outright.
        assert not any(isinstance(e, ToolProposed) and e.needs_approval for e in events)


@pytest.mark.asyncio
async def test_path_escape_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        session = make(
            Path(tmp),
            [
                [
                    StreamToolUse(id='c1', name='read_file', input={'path': '../../../../etc/passwd'}),
                    StreamDone(stop_reason='tool_use'),
                ],
                [StreamText(text='blocked'), StreamDone()],
            ],
            mode=Mode.TRUSTED,
        )
        await session.start()
        session.submit('read the password file')
        events = await drain(session)

        done = [e for e in events if isinstance(e, ToolCompleted)]
        assert done and not done[0].result.ok
        assert 'outside the session root' in done[0].result.content


@pytest.mark.asyncio
async def test_ask_user_suspends_until_answered():
    with tempfile.TemporaryDirectory() as tmp:
        session = make(
            Path(tmp),
            [
                [
                    StreamToolUse(id='c1', name='ask_user', input={'question': 'Which one?', 'options': ['a', 'b']}),
                    StreamDone(stop_reason='tool_use'),
                ],
                [StreamText(text='Going with b.'), StreamDone()],
            ],
            mode=Mode.UNRESTRICTED,
        )
        await session.start()
        session.submit('do the thing')

        events = []

        async def pump():
            async for event in session.events():
                events.append(event)
                if isinstance(event, QuestionAsked):
                    # Even in unrestricted mode the question is asked, not answered.
                    session.answer(event.question_id, 'b')
                if isinstance(event, TurnCompleted):
                    return

        await asyncio.wait_for(pump(), timeout=10)

        assert any(isinstance(e, QuestionAsked) for e in events)
        result = [e for e in events if isinstance(e, ToolCompleted)][0]
        assert result.result.content == 'b'


@pytest.mark.asyncio
async def test_edit_requires_a_read_first():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / 'a.txt').write_text('one\ntwo\n')
        session = make(
            root,
            [
                [
                    StreamToolUse(
                        id='c1', name='edit_file',
                        input={'path': 'a.txt', 'old_string': 'two', 'new_string': 'three'},
                    ),
                    StreamDone(stop_reason='tool_use'),
                ],
                [StreamText(text='ok'), StreamDone()],
            ],
            mode=Mode.UNRESTRICTED,
        )
        await session.start()
        session.submit('change two to three')
        events = await drain(session)

        result = [e for e in events if isinstance(e, ToolCompleted)][0]
        assert not result.result.ok
        assert 'read it before editing' in result.result.content
        # Unchanged on disk, which is the point.
        assert (root / 'a.txt').read_text() == 'one\ntwo\n'


@pytest.mark.asyncio
async def test_interrupt_cancels_the_turn():
    with tempfile.TemporaryDirectory() as tmp:
        session = make(
            Path(tmp),
            [
                [
                    StreamToolUse(id='c1', name='shell', input={'command': 'sleep 30'}),
                    StreamDone(stop_reason='tool_use'),
                ],
            ],
            mode=Mode.UNRESTRICTED,
        )
        await session.start()
        session.submit('sleep')

        events = []

        async def pump():
            async for event in session.events():
                events.append(event)
                if isinstance(event, TurnCompleted):
                    return

        task = asyncio.create_task(pump())
        # Let the command actually start before pulling the rug.
        await asyncio.sleep(0.5)
        assert session.interrupt()
        await asyncio.wait_for(task, timeout=10)

        assert [e for e in events if isinstance(e, TurnCompleted)][0].stop_reason == 'interrupted'


@pytest.mark.asyncio
async def test_a_refused_purchase_is_terminal_not_a_detour():
    """Observed for real: told to place an order and refused, a local model
    clicked a different button and then went for the card fields. It was
    following the ordinary denial message, which says to find another route.
    For money and secrets that instruction is exactly wrong."""
    from openmirror.agent.session import _denial_text
    from openmirror.protocol.agent import ToolCall

    ordinary = _denial_text(ToolCall(id='c', name='shell', risk=Risk.EXECUTE), 'no')
    assert 'another way' in ordinary

    for risk in (Risk.PURCHASE, Risk.CREDENTIAL):
        text = _denial_text(ToolCall(id='c', name='browser_click', risk=risk), 'no')
        assert 'final' in text.lower()
        assert 'do not attempt it another way' in text.lower()
        assert 'another way, or ask' not in text.lower()


@pytest.mark.asyncio
async def test_a_purchase_is_confirmed_even_in_unrestricted_mode():
    """`unrestricted` means stop asking about this machine. It has never meant
    consent to spend money, and there is no setting that makes it mean that.

    Purchases are *possible* by default now — a harness meant to finish a real
    task has to be able to reach the end of one — but `allow_purchases`
    decides whether spending can happen at all, not whether it happens
    unasked. A flag that skipped the prompt is one somebody sets during a demo
    and still has set six months later.
    """
    from openmirror.agent.approval import ApprovalPolicy, Decision, Mode
    from openmirror.protocol.agent import ToolCall

    call = ToolCall(id='c', name='browser_click', risk=Risk.PURCHASE, summary="click 'Place your order'")

    yolo = ApprovalPolicy(mode=Mode.UNRESTRICTED)
    assert yolo.decide(call)[0] is Decision.ASK

    explicitly_on = ApprovalPolicy(mode=Mode.UNRESTRICTED, allow_purchases=True)
    assert explicitly_on.decide(call)[0] is Decision.ASK, 'no setting skips the prompt'

    off = ApprovalPolicy(mode=Mode.UNRESTRICTED, allow_purchases=False)
    assert off.decide(call)[0] is Decision.DENY, 'off should refuse, not prompt'

    # And a one-off approval is never remembered for a purchase.
    yolo.remember(call)
    assert yolo.decide(call)[0] is Decision.ASK


@pytest.mark.asyncio
async def test_a_secret_is_confirmed_and_never_printed():
    """Typing a password is allowed now, and asked about every time.

    This used to be a flat refusal, on the reasoning that an approval prompt
    still meant the agent held the secret. The capability was wanted, so the
    protection moved rather than disappearing: the value stays out of the
    prompt, out of the tool result and out of the page read — see
    tests/test_money_and_secrets.py, which proves that against a real browser.
    """
    from openmirror.agent.approval import ApprovalPolicy, Decision, Mode
    from openmirror.protocol.agent import ToolCall

    call = ToolCall(id='c', name='browser_type', risk=Risk.CREDENTIAL, summary='type into a password field')
    assert ApprovalPolicy(mode=Mode.UNRESTRICTED).decide(call)[0] is Decision.ASK
    assert ApprovalPolicy(mode=Mode.UNRESTRICTED, allow_credentials=False).decide(call)[0] is Decision.DENY

    # Never remembered: "don't ask again" about a secret is not an answer.
    policy = ApprovalPolicy(mode=Mode.TRUSTED)
    policy.remember(call)
    assert policy.decide(call)[0] is Decision.ASK


@pytest.mark.asyncio
async def test_a_tool_image_actually_reaches_the_model():
    """Caught by watching a real run: the screenshot was rendered for the human
    but never sent to the model, which then described a screen it had never
    seen — fluently, and wrongly. `display` is for the UI; `images` is for the
    model, and they are separate for exactly this reason."""
    import base64

    from openmirror.agent.tools.base import Assessment, Output, Tool
    from openmirror.providers.base import ImageBlock

    pixel = base64.b64encode(b'\x89PNG\r\n\x1a\n fake').decode()

    class ShotTool(Tool):
        name = 'shot'
        description = 'take a picture'
        input_schema = {'type': 'object', 'properties': {}}

        def assess(self, args, ctx):
            return Assessment(risk=Risk.READ, summary='shot')

        async def run(self, args, ctx):
            return Output(
                content='Screenshot taken: 100x100.',
                display={'image': pixel, 'media_type': 'image/png'},
                images=[(pixel, 'image/png')],
            )

    with tempfile.TemporaryDirectory() as tmp:
        provider = ScriptedProvider([
            [StreamToolUse(id='c1', name='shot', input={}), StreamDone(stop_reason='tool_use')],
            [StreamText(text='I can see it.'), StreamDone()],
        ])
        session = build_session(
            root=Path(tmp), provider=provider, model='x', mode=Mode.UNRESTRICTED,
            tools=[ShotTool()],
        )
        await session.start()
        session.submit('what is on screen')
        await asyncio.wait_for(session._turn, timeout=10)

        # The second request is the one that should carry the picture.
        blocks = [b for m in provider.seen[-1].messages for b in m.content]
        images = [b for b in blocks if isinstance(b, ImageBlock)]
        assert images, 'the screenshot never reached the model'
        assert images[0].data == pixel
        assert images[0].media_type == 'image/png'


@pytest.mark.asyncio
async def test_display_alone_does_not_reach_the_model():
    """The inverse: a tool that only fills `display` must not leak it into the
    conversation, or every diff and file listing would be sent twice."""
    from openmirror.agent.tools.base import Assessment, Output, Tool
    from openmirror.providers.base import ImageBlock

    class DisplayOnly(Tool):
        name = 'shot'
        description = 'take a picture'
        input_schema = {'type': 'object', 'properties': {}}

        def assess(self, args, ctx):
            return Assessment(risk=Risk.READ, summary='shot')

        async def run(self, args, ctx):
            return Output(content='done', display={'image': 'zzz', 'media_type': 'image/png'})

    with tempfile.TemporaryDirectory() as tmp:
        provider = ScriptedProvider([
            [StreamToolUse(id='c1', name='shot', input={}), StreamDone(stop_reason='tool_use')],
            [StreamText(text='ok'), StreamDone()],
        ])
        session = build_session(
            root=Path(tmp), provider=provider, model='x', mode=Mode.UNRESTRICTED, tools=[DisplayOnly()]
        )
        await session.start()
        session.submit('go')
        await asyncio.wait_for(session._turn, timeout=10)

        blocks = [b for m in provider.seen[-1].messages for b in m.content]
        assert not [b for b in blocks if isinstance(b, ImageBlock)]
