"""The realtime conversation, as one websocket to the browser.

The shape mirrors the voice socket deliberately — binary frames are audio,
text frames are JSON — so a client that can already do one can do the other
with the audio rate changed. What is different is where the intelligence
lives: the voice gateway owns the turn-taking and this one relays it, because
the model upstream is doing that work itself.

The part worth reading is what happens when the model calls a tool.

An autopilot that can run commands on someone's machine while they are talking
to it is the most dangerous thing in this codebase, and the temptation is to
give it its own quick path — the model is waiting, a person is listening, and
an approval prompt in the middle of a spoken sentence is awkward. That
temptation is exactly the failure. So tool calls go through
`AgentSession.invoke`, the same method the typed loop uses: same risk grading,
same policy, same suspension on a human, same events, so the tool cards appear
in the transcript and the approval bar appears in the UI while the person is
still holding the conversation. If the policy says a human decides, the model
waits — and it is told, in the tool result, that it is waiting on a person, so
it can say so out loud instead of going quiet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from openmirror.agent.manager import manager
from openmirror.config import config
from openmirror.protocol.agent import ToolCall
from openmirror.providers.realtime import (
    SAMPLE_RATE,
    RealtimeConfig,
    RealtimeSession,
    RealtimeUnavailable,
    tool_specs,
)

log = logging.getLogger(__name__)

router = APIRouter()

# The realtime model talks. It should not narrate a build log, and it must not
# read a tool's output aloud — so it is told what to say about its own work.
AUTOPILOT_PROMPT = """You are openmirror, in a spoken conversation with someone at their computer. \
You can act on the machine through the tools you have.

Speak the way a person would. Short sentences. No lists read aloud, no file paths spelled out \
character by character, no reading command output back — say what it means instead. If \
something takes a while, say what you are doing before you start rather than going silent.

You have real tools that change real files and run real commands. Before you do anything that \
changes something, say what you are about to do, in one sentence. Some actions need the person \
to approve them on screen: when that happens the tool result will tell you so — say out loud \
that you are waiting on them rather than falling silent, because they may not be looking at the \
screen.

If a tool result says a person declined something, do not try it another way. Ask them what \
they would rather do.

You are hearing a live microphone. If what you heard makes no sense, say so and ask, rather \
than acting on a guess — acting on a misheard instruction is much worse here than asking twice."""


@router.websocket('/ws/realtime')
async def realtime_socket(
    ws: WebSocket,
    session: str | None = Query(None, description='Agent session whose tools it may use.'),
    voice: str | None = Query(None),
    model: str | None = Query(None),
    token: str | None = Query(None),
) -> None:
    if config.auth_token and token != config.auth_token:
        await ws.close(code=4401, reason='unauthorised')
        return

    await ws.accept()

    # An agent session is optional. Without one this is a conversation and
    # nothing more, which is a perfectly reasonable thing to want and is also
    # the only form that is safe to leave running unattended.
    agent = manager.get(session) if session else None
    tools = tool_specs(agent.tools) if agent is not None else []

    instructions = AUTOPILOT_PROMPT if agent is not None else (
        'You are openmirror, having a spoken conversation. Speak the way a person would: short '
        'sentences, no lists read aloud. You have no tools in this conversation — if you are '
        'asked to do something on the computer, say that this mode is talk only and offer to '
        'do it in a session that has tools.'
    )
    if agent is not None:
        instructions += f'\n\nWorking directory: {agent.root}\nApproval policy: {agent.policy.describe()}'

    live = RealtimeSession(
        config.openai_key,
        RealtimeConfig(
            model=model or config.realtime_model,
            voice=voice or config.realtime_voice,
            instructions=instructions,
            tools=tools,
            url=config.realtime_url,
        ),
    )

    try:
        await live.connect()
    except RealtimeUnavailable as exc:
        await ws.send_json({'type': 'realtime.error', 'message': str(exc), 'fatal': True})
        await ws.close(code=4404)
        return

    await ws.send_json(
        {
            'type': 'realtime.ready',
            'sample_rate': SAMPLE_RATE,
            'model': live.config.model,
            'voice': live.config.voice,
            'session': agent.id if agent is not None else None,
            'tools': sorted(agent.tools) if agent is not None else [],
            'policy': agent.policy.describe() if agent is not None else '',
        }
    )

    # What the model is currently saying, so a barge-in can tell the server how
    # much of it was actually heard.
    speaking = {'item_id': '', 'samples': 0}
    turn_id = uuid.uuid4().hex[:16]

    async def run_tool(call_id: str, name: str, arguments: dict) -> None:
        """Run one tool the model asked for, and hand the result back.

        In its own task: a tool that suspends on an approval would otherwise
        stop this coroutine reading from upstream, and the conversation would
        freeze — including the person's ability to say "no, stop".
        """
        if agent is None:
            await live.send_tool_result(call_id, 'This conversation has no tools attached.')
            return

        call = ToolCall(id=call_id, name=name, arguments=arguments)
        try:
            result = await agent.invoke(call, turn_id)
            payload = result.content
            if result.is_error:
                payload = f'FAILED: {payload}'
        except Exception as exc:  # noqa: BLE001
            log.exception('realtime tool %s failed', name)
            payload = f'FAILED: {exc}'

        try:
            await live.send_tool_result(call_id, payload)
        except RealtimeUnavailable:
            pass

    async def pump_upstream() -> None:
        async for event in live.events():
            kind = event['kind']

            if kind == 'audio':
                speaking['item_id'] = event['item_id'] or speaking['item_id']
                speaking['samples'] += len(event['pcm']) // 2
                try:
                    await ws.send_bytes(event['pcm'])
                except (WebSocketDisconnect, RuntimeError):
                    return
                continue

            if kind == 'tool':
                # Not awaited: see run_tool.
                asyncio.create_task(run_tool(event['call_id'], event['name'], event['arguments']))
                await _say(ws, {'type': 'realtime.tool', 'name': event['name'], 'arguments': event['arguments']})
                continue

            if kind == 'speech.started':
                # The person started talking over it. Everything already sent
                # has to be dropped by the client, and the server has to be
                # told where the person actually stopped hearing.
                played_ms = int(speaking['samples'] / SAMPLE_RATE * 1000)
                try:
                    await live.interrupt(played_ms=played_ms, item_id=speaking['item_id'])
                except RealtimeUnavailable:
                    pass
                speaking['samples'] = 0
                await _say(ws, {'type': 'realtime.interrupted'})
                continue

            if kind == 'audio.done':
                speaking['samples'] = 0

            await _say(ws, {'type': f'realtime.{kind}', **{k: v for k, v in event.items() if k != 'kind'}})

    upstream = asyncio.create_task(pump_upstream())

    try:
        while True:
            message = await ws.receive()
            if message.get('type') == 'websocket.disconnect':
                break

            if (audio := message.get('bytes')) is not None:
                await live.send_audio(audio)
                continue

            if (text := message.get('text')) is None:
                continue
            try:
                command = json.loads(text)
            except json.JSONDecodeError:
                continue

            kind = command.get('type')
            if kind == 'realtime.text':
                await live.send_text(command.get('text', ''))
            elif kind == 'realtime.interrupt':
                await live.interrupt(
                    played_ms=int(command.get('played_ms') or 0), item_id=speaking['item_id']
                )
                speaking['samples'] = 0
            elif kind == 'realtime.stop':
                break
            elif kind == 'ping':
                await _say(ws, {'type': 'pong'})

    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.exception('realtime socket failed')
    finally:
        upstream.cancel()
        await live.close()


async def _say(ws: WebSocket, payload: dict) -> None:
    """Send, tolerating a client that has already gone.

    The usual way one of these ends is the tab closing, and the teardown that
    follows still emits — so a send on a dead socket is expected rather than
    exceptional. Letting it raise turns an ordinary hang-up into a traceback
    and skips the rest of the cleanup.
    """
    try:
        await ws.send_json(payload)
    except (WebSocketDisconnect, RuntimeError):
        pass
