"""Anything speaking OpenAI's shapes.

This one adapter is worth more than the rest put together, because the shape
won. OpenAI itself, Perch's chat and speech services, vLLM, llama.cpp's
server, LM Studio, Groq, Together, OpenRouter and Open WebUI's own
`/api/v1/chat/completions` are all reachable through it with nothing changed
but a base URL and a key.

The conversions are the interesting part. Our message format keeps content as
typed blocks and OpenAI's does not, so tool calls have to be flattened on the
way out (a tool result becomes a `role: "tool"` message of its own) and
reassembled on the way back (arguments arrive as partial JSON strings spread
over many deltas, and are useless until the last one lands).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import aiohttp

from roost.media.params import Param, common_image
from roost.net.transport import Transport
from roost.providers.base import (
    ChatProvider,
    ChatRequest,
    EmbeddingProvider,
    GeneratedMedia,
    ImageBlock,
    ImageProvider,
    Message,
    StreamDone,
    StreamEvent,
    StreamText,
    StreamThinking,
    StreamToolUse,
    STTProvider,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Transcript,
    TTSProvider,
)
from roost.providers.control_tokens import strip_control_tokens

log = logging.getLogger(__name__)


def _to_openai_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
    """Flatten our blocks into OpenAI's message list.

    One assistant turn can hold text and several tool calls, and each tool
    result becomes its own message afterwards — so the count of messages going
    out is not the count coming in.
    """
    out: list[dict[str, Any]] = []
    if system:
        out.append({'role': 'system', 'content': system})

    for msg in messages:
        # Tool results are their own messages in this dialect and cannot be
        # carried inside the user turn they arrived with, so they are pulled
        # out first and appended before it.
        tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
        for tr in tool_results:
            out.append(
                {
                    'role': 'tool',
                    'tool_call_id': tr.tool_use_id,
                    # OpenAI has no error flag on a tool message; the model can
                    # only be told in words, so say so unambiguously.
                    'content': f'ERROR: {tr.content}' if tr.is_error else tr.content,
                }
            )

        parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                parts.append({'type': 'text', 'text': block.text})
            elif isinstance(block, ImageBlock):
                parts.append(
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:{block.media_type};base64,{block.data}'},
                    }
                )
            elif isinstance(block, ThinkingBlock):
                # Not replayed. No OpenAI-shaped endpoint accepts reasoning back
                # as input, and several 400 on an unknown content part.
                continue
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(
                    {
                        'id': block.id,
                        'type': 'function',
                        'function': {'name': block.name, 'arguments': json.dumps(block.input)},
                    }
                )

        if not parts and not tool_calls:
            continue

        entry: dict[str, Any] = {'role': msg.role}
        if parts:
            # A lone text part is sent as a bare string: some servers that claim
            # compatibility reject the array form on a text-only message.
            entry['content'] = parts[0]['text'] if len(parts) == 1 and parts[0]['type'] == 'text' else parts
        elif tool_calls:
            entry['content'] = None
        if tool_calls:
            entry['tool_calls'] = tool_calls
        out.append(entry)

    return out


class OpenAICompatProvider(ChatProvider, EmbeddingProvider, STTProvider, TTSProvider, ImageProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str = '',
        *,
        provider_id: str = 'openai',
        timeout: int = 600,
        headers: dict[str, str] | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.provider_id = provider_id
        self.timeout = timeout
        self._extra_headers = headers or {}
        self.transport = transport or Transport(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        h = {'Content-Type': 'application/json', **self._extra_headers}
        if self.api_key:
            h['Authorization'] = f'Bearer {self.api_key}'
        return h

    def _session(self, timeout: float | None = None) -> aiohttp.ClientSession:
        # Through the transport rather than built here, so that turning Tor on
        # for this connection reaches chat, embeddings, speech and images at
        # once instead of whichever of them was remembered.
        return self.transport.session(timeout)

    # -- chat ---------------------------------------------------------------

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        payload: dict[str, Any] = {
            'model': req.model,
            'messages': _to_openai_messages(req.messages, req.system),
            'stream': True,
            # Ask for usage on the final chunk. Servers that do not know the
            # option ignore it; none of the ones tested reject it.
            'stream_options': {'include_usage': True},
        }
        # Omitted entirely when uncapped, which is this API's own way of saying
        # "up to the model's limit". A large stand-in would be a ceiling by
        # another name, and the wrong one on most models.
        if req.max_tokens:
            payload['max_tokens'] = req.max_tokens
        if req.temperature is not None:
            payload['temperature'] = req.temperature
        if req.stop:
            payload['stop'] = req.stop
        if req.tools:
            payload['tools'] = [
                {
                    'type': 'function',
                    'function': {'name': t.name, 'description': t.description, 'parameters': t.input_schema},
                }
                for t in req.tools
            ]
        payload.update(req.extra)

        # Tool arguments arrive as partial JSON keyed by index, not by id, so
        # they are accumulated here and parsed once the stream ends.
        pending: dict[int, dict[str, Any]] = {}
        usage: dict[str, int] = {}
        stop_reason = 'end_turn'

        async with self._session() as session:
            async with session.post(
                f'{self.base_url}/chat/completions', json=payload, headers=self._headers()
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f'{self.provider_id}: HTTP {resp.status}: {body[:500]}')

                async for raw in resp.content:
                    line = raw.decode('utf-8', 'replace').strip()
                    if not line or not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get('usage'):
                        u = chunk['usage']
                        usage = {
                            'input_tokens': u.get('prompt_tokens', 0),
                            'output_tokens': u.get('completion_tokens', 0),
                        }

                    for choice in chunk.get('choices') or []:
                        delta = choice.get('delta') or {}

                        if delta.get('content'):
                            # A local server behind an OpenAI-shaped endpoint leaks
                            # template markers the same way a native one does.
                            cleaned = strip_control_tokens(delta['content'])
                            if cleaned:
                                yield StreamText(text=cleaned)

                        # Where a server exposes reasoning it uses one of these
                        # two names; neither is standard.
                        reasoning = delta.get('reasoning_content') or delta.get('reasoning')
                        if reasoning:
                            yield StreamThinking(text=reasoning)

                        for tc in delta.get('tool_calls') or []:
                            idx = tc.get('index', 0)
                            slot = pending.setdefault(idx, {'id': '', 'name': '', 'args': ''})
                            if tc.get('id'):
                                slot['id'] = tc['id']
                            fn = tc.get('function') or {}
                            if fn.get('name'):
                                slot['name'] = fn['name']
                            if fn.get('arguments'):
                                slot['args'] += fn['arguments']

                        if choice.get('finish_reason'):
                            stop_reason = {
                                'stop': 'end_turn',
                                'length': 'max_tokens',
                                'tool_calls': 'tool_use',
                                'function_call': 'tool_use',
                            }.get(choice['finish_reason'], choice['finish_reason'])

        for idx in sorted(pending):
            slot = pending[idx]
            if not slot['name']:
                continue
            try:
                args = json.loads(slot['args']) if slot['args'].strip() else {}
            except json.JSONDecodeError:
                # A truncated argument string means the model was cut off
                # mid-call. Passing {} would run the tool with defaults, which
                # is worse than failing, so the call is dropped.
                log.warning('%s: unparseable tool arguments for %s', self.provider_id, slot['name'])
                continue
            yield StreamToolUse(id=slot['id'] or f'call_{idx}', name=slot['name'], input=args)

        yield StreamDone(stop_reason=stop_reason, usage=usage)

    async def models(self) -> list[dict[str, Any]]:
        async with self._session() as session:
            async with session.get(f'{self.base_url}/models', headers=self._headers()) as resp:
                if resp.status != 200:
                    return []
                body = await resp.json()
        return [{'id': m.get('id'), 'provider': self.provider_id} for m in body.get('data', []) if m.get('id')]

    # -- embeddings ---------------------------------------------------------

    async def embed(
        self,
        texts: list[str],
        model: str,
        *,
        input_type: str = 'document',
        dimensions: int | None = None,
    ) -> list[list[float]]:
        # `input_type` is not part of OpenAI's shape and is deliberately
        # dropped rather than passed through: several servers claiming
        # compatibility 400 on an unknown field, and losing a quality nicety
        # is better than losing the call.
        payload: dict[str, Any] = {'model': model, 'input': texts}
        if dimensions:
            payload['dimensions'] = dimensions
        async with self._session() as session:
            async with session.post(
                f'{self.base_url}/embeddings', json=payload, headers=self._headers()
            ) as resp:
                resp.raise_for_status()
                body = await resp.json()
        return [d['embedding'] for d in sorted(body['data'], key=lambda d: d.get('index', 0))]

    # -- speech to text -----------------------------------------------------

    def supports_streaming(self) -> bool:
        # The OpenAI transcription endpoint is request/response. whisper.cpp
        # behind Perch is the same. The gateway cuts on VAD instead.
        return False

    async def transcribe(self, audio: bytes, *, model: str, language: str | None = None) -> Transcript:
        form = aiohttp.FormData()
        form.add_field('file', audio, filename='audio.wav', content_type='audio/wav')
        form.add_field('model', model)
        if language:
            form.add_field('language', language)
        form.add_field('response_format', 'json')

        headers = {k: v for k, v in self._headers().items() if k != 'Content-Type'}
        async with self._session() as session:
            async with session.post(f'{self.base_url}/audio/transcriptions', data=form, headers=headers) as resp:
                if resp.status != 200:
                    raise RuntimeError(f'{self.provider_id} stt: HTTP {resp.status}: {(await resp.text())[:300]}')
                body = await resp.json()
        return Transcript(text=(body.get('text') or '').strip(), partial=False, language=language)

    # -- text to speech -----------------------------------------------------

    async def synthesize(
        self, text: str, *, model: str, voice: str, fmt: str = 'pcm16', sample_rate: int = 16_000
    ) -> AsyncIterator[bytes]:
        # 'pcm' is raw 24 kHz signed 16-bit little-endian on OpenAI and on
        # Kokoro. Asking for it rather than mp3 keeps a decoder out of the
        # voice path, which is worth a few hundred milliseconds per utterance.
        payload = {
            'model': model,
            'input': text,
            'voice': voice,
            'response_format': 'pcm' if fmt == 'pcm16' else fmt,
        }
        async with self._session() as session:
            async with session.post(f'{self.base_url}/audio/speech', json=payload, headers=self._headers()) as resp:
                if resp.status != 200:
                    raise RuntimeError(f'{self.provider_id} tts: HTTP {resp.status}: {(await resp.text())[:300]}')
                # Yielded as it arrives. Closing this iterator mid-utterance is
                # what cancels a synthesis on barge-in.
                async for chunk in resp.content.iter_chunked(4096):
                    yield chunk

    async def voices(self) -> list[dict[str, Any]]:
        async with self._session() as session:
            try:
                async with session.get(f'{self.base_url}/audio/voices', headers=self._headers()) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        items = body.get('voices', body) if isinstance(body, dict) else body
                        return [{'id': v} if isinstance(v, str) else v for v in items]
            except aiohttp.ClientError:
                pass
        # OpenAI has no voices endpoint; these are its documented six.
        return [{'id': v} for v in ('alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer')]

    # -- images -------------------------------------------------------------

    async def generate(
        self, prompt: str, *, model: str, n: int = 1, size: str = '1024x1024', **kw: Any
    ) -> list[GeneratedMedia]:
        import base64

        payload: dict[str, Any] = {'model': model, 'prompt': prompt, 'n': n, 'size': size}

        # `response_format` is a dall-e parameter and gpt-image-1 rejects it:
        # that model always returns base64 and 400s on being told to. Sending
        # it unconditionally is the single most common way this call fails.
        if not model.startswith('gpt-image'):
            payload['response_format'] = 'b64_json'

        # Only the ones that were actually set. An explicit `style: null` is a
        # 400 on several gateways that otherwise accept the request.
        for key in ('quality', 'style', 'background', 'output_format',
                    'output_compression', 'moderation'):
            value = kw.pop(key, None)
            if value not in (None, '', 'auto'):
                payload[key] = value
        payload.update(kw)

        async with self._session() as session:
            async with session.post(
                f'{self.base_url}/images/generations', json=payload, headers=self._headers()
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(
                        f'{self.provider_id} image: HTTP {resp.status}: {(await resp.text())[:300]}'
                    )
                body = await resp.json()

        fmt = payload.get('output_format', 'png')
        out: list[GeneratedMedia] = []
        for row in body.get('data', []):
            if row.get('b64_json'):
                out.append(
                    GeneratedMedia(
                        data=base64.b64decode(row['b64_json']),
                        media_type=f'image/{fmt}',
                        meta={'revised_prompt': row.get('revised_prompt', '')},
                    )
                )
            elif row.get('url'):
                # Some gateways only return a URL. Fetched here rather than
                # handed onward, so that every caller gets bytes and none of
                # them has to know which backend it was talking to.
                async with self._session(120) as session:
                    async with session.get(row['url']) as media:
                        media.raise_for_status()
                        out.append(
                            GeneratedMedia(data=await media.read(), media_type=f'image/{fmt}')
                        )
        return out

    async def describe(self, model: str = '') -> list[Param]:
        """The knobs this model actually has.

        Branching on the model name rather than offering the union: `style` is
        a dall-e-3 parameter and `background` is a gpt-image-1 one, and a form
        showing both teaches people to set things that are being ignored.
        """
        params = common_image()
        # No negative conditioning anywhere in this API. Left out rather than
        # shown and dropped — a field that does nothing is worse than none.
        params = [p for p in params if p.name != 'negative_prompt']
        params = [p for p in params if p.name != 'seed']

        if model.startswith('gpt-image'):
            return params + [
                Param('size', 'Size', 'enum', default='1024x1024',
                      options=['1024x1024', '1536x1024', '1024x1536', 'auto'], group='shape'),
                Param('quality', 'Quality', 'enum', default='auto',
                      options=['auto', 'low', 'medium', 'high'], group='quality',
                      help='Higher costs more and takes longer. "auto" lets the model pick.'),
                Param('background', 'Background', 'enum', default='auto',
                      options=['auto', 'transparent', 'opaque'], group='output',
                      help='Transparent needs png or webp output.'),
                Param('output_format', 'File format', 'enum', default='png',
                      options=['png', 'jpeg', 'webp'], group='output', advanced=True),
                Param('output_compression', 'Compression', 'int', default=100, minimum=0,
                      maximum=100, step=1, group='output', advanced=True,
                      help='For jpeg and webp only.'),
                Param('moderation', 'Moderation', 'enum', default='auto', options=['auto', 'low'],
                      group='output', advanced=True),
            ]

        if model.startswith('dall-e-3'):
            return params + [
                Param('size', 'Size', 'enum', default='1024x1024',
                      options=['1024x1024', '1792x1024', '1024x1792'], group='shape'),
                Param('quality', 'Quality', 'enum', default='standard',
                      options=['standard', 'hd'], group='quality'),
                Param('style', 'Style', 'enum', default='vivid', options=['vivid', 'natural'],
                      group='quality',
                      help='"vivid" pushes toward dramatic and saturated; "natural" does not.'),
            ]

        # An OpenAI-shaped server this build has never heard of. The three
        # parameters every one of them takes, and nothing invented on top.
        return params + [
            Param('size', 'Size', 'string', default='1024x1024', group='shape',
                  help='Whatever sizes this server accepts — it is not one this build knows.'),
        ]
