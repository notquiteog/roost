"""OpenAI's realtime API: speech in, speech out, one socket.

This is not a modality the registry can route, and pretending otherwise would
be the wrong shape. Everywhere else here, a conversation is three services
bolted together — transcribe, think, synthesise — and the gateway owns the
seams: it decides when an utterance ended, when to interrupt, what history to
keep. The realtime API is one service that owns all of that itself, speech to
speech, with no text in the middle. So it gets its own adapter, and the voice
pipeline is left alone.

What you gain is the latency and the prosody: the model hears *how* something
was said and answers before a transcript would have existed. What you give up
is choice — it is one vendor's model, it cannot be pointed at your own
hardware, and everything said in the room goes to it. Both halves of that are
worth stating in the UI, and the UI does.

Three details that are easy to get wrong:

**The event names moved between the beta and GA.** Audio deltas arrive as
`response.audio.delta` on the beta protocol and `response.output_audio.delta`
on GA, and a client that knows only one of them sits in silence against the
other while everything looks fine. Both are accepted, on the way in and out.

**Interruption has to be told to the server.** Barge-in is not just dropping
what is buffered locally: the model believes it has already said everything
it sent, so the conversation has to be truncated at the point the person
actually stopped hearing it, or the model's memory of the exchange includes
three sentences nobody heard.

**Audio is 24 kHz PCM16, base64, and the rate is not negotiable.** A client
sending 16 kHz gets a model that hears everything a third too slow, which
presents as poor transcription rather than as a format error.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000

# The same event, under both spellings. Kept as tuples rather than normalised
# at the edges because the GA name is what a reader should see in the code and
# the beta name is what a five-year-old deployment still sends.
AUDIO_DELTA = ('response.output_audio.delta', 'response.audio.delta')
AUDIO_DONE = ('response.output_audio.done', 'response.audio.done')
TRANSCRIPT_DELTA = ('response.output_audio_transcript.delta', 'response.audio_transcript.delta')
TEXT_DELTA = ('response.output_text.delta', 'response.text.delta')


@dataclass(slots=True)
class RealtimeConfig:
    model: str = 'gpt-realtime'
    voice: str = 'marin'
    instructions: str = ''
    # Server-side turn detection. The alternative is the client deciding when
    # someone has stopped talking, which is a round trip's worth of latency in
    # the one place a conversation cannot afford it.
    turn_detection: str = 'server_vad'
    # How long a silence ends a turn, in milliseconds. Short feels sharp and
    # interrupts people who pause to think; long feels sluggish. 500 is the
    # documented default and a reasonable place to start arguing from.
    silence_ms: int = 500
    temperature: float | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    url: str = 'wss://api.openai.com/v1/realtime'


class RealtimeUnavailable(RuntimeError):
    """The realtime API cannot be reached, or is not configured."""


class RealtimeSession:
    """One live conversation with the realtime model."""

    def __init__(self, api_key: str, config: RealtimeConfig | None = None) -> None:
        if not api_key:
            raise RealtimeUnavailable(
                'the realtime API needs an OpenAI key (OPENAI_API_KEY). It is one vendor\'s '
                'service and cannot be pointed at local hardware — for a local voice call, '
                'use the ordinary voice mode, which routes dictation, thinking and speech '
                'separately.'
            )
        self.api_key = api_key
        self.config = config or RealtimeConfig()
        self._ws: Any = None
        self._session: Any = None

    async def connect(self) -> None:
        import aiohttp

        url = f'{self.config.url}?model={self.config.model}'
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            # Harmless on GA and required on the beta protocol. Sending it is
            # the cheaper mistake: without it, an older deployment refuses the
            # upgrade with a message about the model rather than the header.
            'OpenAI-Beta': 'realtime=v1',
        }
        # transport-exempt: the realtime API has no Tor option, and saying so
        # is better than appearing to. It is one vendor's websocket, it cannot
        # be pointed at local hardware, and a SOCKS-proxied websocket upgrade
        # is a different code path from the request-shaped calls the transport
        # covers. The Live pane tells the person both facts before they open it.
        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(url, headers=headers, heartbeat=20)
        except Exception as exc:
            await self._session.close()
            self._session = None
            raise RealtimeUnavailable(f'could not open the realtime socket: {exc}') from exc

        await self.configure()

    async def configure(self) -> None:
        """Tell the session what it is and what it may call."""
        session: dict[str, Any] = {
            'modalities': ['audio', 'text'],
            'voice': self.config.voice,
            'input_audio_format': 'pcm16',
            'output_audio_format': 'pcm16',
            # Asked for explicitly: without it there is no record of what the
            # person said, and a conversation with no transcript is one nobody
            # can check afterwards.
            'input_audio_transcription': {'model': 'whisper-1'},
        }
        if self.config.instructions:
            session['instructions'] = self.config.instructions
        if self.config.temperature is not None:
            session['temperature'] = self.config.temperature
        if self.config.turn_detection == 'none':
            session['turn_detection'] = None
        else:
            session['turn_detection'] = {
                'type': self.config.turn_detection,
                'silence_duration_ms': self.config.silence_ms,
                # The model answers by itself when someone stops talking. That
                # is what makes this a conversation rather than a walkie-talkie.
                'create_response': True,
                'interrupt_response': True,
            }
        if self.config.tools:
            session['tools'] = self.config.tools
            session['tool_choice'] = 'auto'

        await self.send({'type': 'session.update', 'session': session})

    # -- sending ------------------------------------------------------------

    async def send(self, event: dict[str, Any]) -> None:
        if self._ws is None or self._ws.closed:
            raise RealtimeUnavailable('the realtime socket is closed')
        await self._ws.send_str(json.dumps(event))

    async def send_audio(self, pcm: bytes) -> None:
        """Append microphone audio. 24 kHz, mono, signed 16-bit little-endian."""
        await self.send(
            {'type': 'input_audio_buffer.append', 'audio': base64.b64encode(pcm).decode()}
        )

    async def send_text(self, text: str) -> None:
        """Say something as though it had been spoken.

        Useful for more than typing: it is how a tool result that needs the
        model to react — "the page you asked about has loaded" — gets into a
        conversation that is otherwise entirely audio.
        """
        await self.send(
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_text', 'text': text}],
                },
            }
        )
        await self.send({'type': 'response.create'})

    async def send_tool_result(self, call_id: str, output: str) -> None:
        """Hand back what a tool returned, and let the model carry on talking.

        The `response.create` afterwards is not optional. Without it the model
        has the result and no reason to speak, and the person is left listening
        to silence after watching a tool run.
        """
        await self.send(
            {
                'type': 'conversation.item.create',
                'item': {
                    'type': 'function_call_output',
                    'call_id': call_id,
                    'output': output[:10_000],
                },
            }
        )
        await self.send({'type': 'response.create'})

    async def interrupt(self, *, played_ms: int = 0, item_id: str = '') -> None:
        """Stop the model talking, and tell it how much was actually heard.

        The truncation is the part that matters. The server believes the whole
        reply was delivered; if it is not told where the person stopped hearing
        it, the model's memory of the conversation contains sentences nobody
        heard and it will refer back to them.
        """
        await self.send({'type': 'response.cancel'})
        if item_id:
            await self.send(
                {
                    'type': 'conversation.item.truncate',
                    'item_id': item_id,
                    'content_index': 0,
                    'audio_end_ms': max(0, int(played_ms)),
                }
            )

    # -- receiving ----------------------------------------------------------

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        """Upstream events, with audio decoded and the two spellings unified.

        Yields `{'kind': ..., ...}` rather than the raw event: a caller should
        not have to know which protocol version answered, and the mapping is
        the one thing this adapter exists to own.
        """
        import aiohttp

        if self._ws is None:
            raise RealtimeUnavailable('connect() first')

        async for message in self._ws:
            if message.type == aiohttp.WSMsgType.ERROR:
                yield {'kind': 'error', 'message': str(self._ws.exception())}
                break
            if message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                break
            if message.type != aiohttp.WSMsgType.TEXT:
                continue

            try:
                event = json.loads(message.data)
            except json.JSONDecodeError:
                continue

            kind = event.get('type', '')

            if kind in AUDIO_DELTA:
                yield {
                    'kind': 'audio',
                    'pcm': base64.b64decode(event.get('delta') or ''),
                    'item_id': event.get('item_id', ''),
                }
            elif kind in AUDIO_DONE:
                yield {'kind': 'audio.done', 'item_id': event.get('item_id', '')}
            elif kind in TRANSCRIPT_DELTA:
                yield {'kind': 'said', 'text': event.get('delta') or ''}
            elif kind in TEXT_DELTA:
                yield {'kind': 'text', 'text': event.get('delta') or ''}
            elif kind == 'conversation.item.input_audio_transcription.completed':
                yield {'kind': 'heard', 'text': event.get('transcript') or ''}
            elif kind == 'input_audio_buffer.speech_started':
                yield {'kind': 'speech.started'}
            elif kind == 'input_audio_buffer.speech_stopped':
                yield {'kind': 'speech.stopped'}
            elif kind == 'response.function_call_arguments.done':
                # Arguments arrive as partial JSON and are only useful whole.
                try:
                    arguments = json.loads(event.get('arguments') or '{}')
                except json.JSONDecodeError:
                    arguments = {}
                yield {
                    'kind': 'tool',
                    'call_id': event.get('call_id', ''),
                    'name': event.get('name', ''),
                    'arguments': arguments,
                }
            elif kind == 'response.done':
                response = event.get('response') or {}
                yield {
                    'kind': 'done',
                    'status': response.get('status', ''),
                    'usage': response.get('usage') or {},
                }
            elif kind == 'session.created':
                yield {'kind': 'ready', 'session': event.get('session') or {}}
            elif kind == 'error':
                error = event.get('error') or {}
                yield {'kind': 'error', 'message': error.get('message') or json.dumps(error)[:300]}

    async def close(self) -> None:
        try:
            if self._ws is not None and not self._ws.closed:
                await self._ws.close()
        finally:
            if self._session is not None:
                await self._session.close()
            self._ws = self._session = None


def tool_specs(tools: dict[str, Any]) -> list[dict[str, Any]]:
    """Agent tools, in the shape the realtime session wants them declared.

    Flat rather than nested under `function`, which is how this API differs
    from chat completions — and getting it wrong produces a session that
    connects, works, and never calls anything.
    """
    return [
        {
            'type': 'function',
            'name': tool.name,
            'description': tool.description,
            'parameters': tool.input_schema,
        }
        for tool in tools.values()
    ]


async def probe(api_key: str, config: RealtimeConfig | None = None, timeout: float = 10.0) -> tuple[bool, str]:  # noqa: ASYNC109 - a connect deadline
    """Whether a realtime session can actually be opened with this key."""
    session = RealtimeSession(api_key, config)
    try:
        await asyncio.wait_for(session.connect(), timeout=timeout)
    except (RealtimeUnavailable, TimeoutError) as exc:
        return False, str(exc)
    finally:
        await session.close()
    return True, 'the realtime API accepted the connection'
