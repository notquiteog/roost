"""Anthropic's Messages API, natively.

Going through an OpenAI-shaped translation layer would work and would cost
three things worth having: extended thinking, prompt caching, and tool
results that keep their structure. Since our internal message format is
already block-shaped, this adapter is mostly a rename.

Written against Anthropic's published API documentation. No Anthropic source
is used or reproduced here.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from roost.net.transport import Transport
from roost.providers.base import (
    ChatProvider,
    ChatRequest,
    ImageBlock,
    StreamDone,
    StreamEvent,
    StreamText,
    StreamThinking,
    StreamToolUse,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

API_VERSION = '2023-06-01'


def _to_anthropic_content(block: Any) -> dict[str, Any] | None:
    if isinstance(block, TextBlock):
        return {'type': 'text', 'text': block.text}
    if isinstance(block, ImageBlock):
        return {
            'type': 'image',
            'source': {'type': 'base64', 'media_type': block.media_type, 'data': block.data},
        }
    if isinstance(block, ThinkingBlock):
        # Replayed only with its signature. An unsigned thinking block is
        # rejected, so one that lost its signature is dropped rather than
        # failing the whole request.
        if not block.signature:
            return None
        return {'type': 'thinking', 'thinking': block.text, 'signature': block.signature}
    if isinstance(block, ToolUseBlock):
        return {'type': 'tool_use', 'id': block.id, 'name': block.name, 'input': block.input}
    if isinstance(block, ToolResultBlock):
        return {
            'type': 'tool_result',
            'tool_use_id': block.tool_use_id,
            'content': block.content,
            'is_error': block.is_error,
        }
    return None


class AnthropicProvider(ChatProvider):
    def __init__(
        self,
        base_url: str = 'https://api.anthropic.com/v1',
        api_key: str = '',
        *,
        provider_id: str = 'anthropic',
        timeout: int = 600,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.provider_id = provider_id
        self.timeout = timeout
        self.transport = transport or Transport(timeout=timeout)
        # Output ceilings, per model. See _output_limit.
        self._output_limits: dict[str, int] = {}

    def _headers(self) -> dict[str, str]:
        return {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': API_VERSION,
        }

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        messages: list[dict[str, Any]] = []
        for msg in req.messages:
            blocks = [b for b in (_to_anthropic_content(b) for b in msg.content) if b]
            if blocks:
                messages.append({'role': msg.role, 'content': blocks})

        payload: dict[str, Any] = {
            'model': req.model,
            'messages': messages,
            # Required by this API -- there is no omitting it, so this is the
            # one adapter that cannot express "no ceiling" by absence. It sends
            # the model's own maximum, asked for once and remembered.
            'max_tokens': req.max_tokens or await self._output_limit(req.model),
            'stream': True,
        }
        if req.system:
            payload['system'] = req.system
        if req.temperature is not None:
            payload['temperature'] = req.temperature
        if req.stop:
            payload['stop_sequences'] = req.stop
        if req.tools:
            payload['tools'] = [
                {'name': t.name, 'description': t.description, 'input_schema': t.input_schema} for t in req.tools
            ]
        payload.update(req.extra)

        # Blocks arrive indexed, and a tool_use block's input is streamed as
        # partial JSON across many deltas, so state is kept per index until
        # content_block_stop says the block is whole.
        blocks: dict[int, dict[str, Any]] = {}
        usage: dict[str, int] = {}
        stop_reason = 'end_turn'

        async with self.transport.session() as session:
            async with session.post(f'{self.base_url}/messages', json=payload, headers=self._headers()) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f'anthropic: HTTP {resp.status}: {body[:500]}')

                async for raw in resp.content:
                    line = raw.decode('utf-8', 'replace').strip()
                    if not line.startswith('data:'):
                        continue
                    try:
                        ev = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue

                    kind = ev.get('type')

                    if kind == 'message_start':
                        u = (ev.get('message') or {}).get('usage') or {}
                        usage['input_tokens'] = u.get('input_tokens', 0)

                    elif kind == 'content_block_start':
                        cb = ev.get('content_block') or {}
                        blocks[ev.get('index', 0)] = {
                            'type': cb.get('type'),
                            'id': cb.get('id', ''),
                            'name': cb.get('name', ''),
                            'json': '',
                        }

                    elif kind == 'content_block_delta':
                        d = ev.get('delta') or {}
                        dt = d.get('type')
                        if dt == 'text_delta':
                            yield StreamText(text=d.get('text', ''))
                        elif dt == 'thinking_delta':
                            yield StreamThinking(text=d.get('thinking', ''))
                        elif dt == 'input_json_delta':
                            slot = blocks.get(ev.get('index', 0))
                            if slot is not None:
                                slot['json'] += d.get('partial_json', '')

                    elif kind == 'content_block_stop':
                        slot = blocks.pop(ev.get('index', 0), None)
                        if slot and slot.get('type') == 'tool_use':
                            try:
                                args = json.loads(slot['json']) if slot['json'].strip() else {}
                            except json.JSONDecodeError:
                                continue
                            yield StreamToolUse(id=slot['id'], name=slot['name'], input=args)

                    elif kind == 'message_delta':
                        stop_reason = (ev.get('delta') or {}).get('stop_reason') or stop_reason
                        u = ev.get('usage') or {}
                        if u.get('output_tokens'):
                            usage['output_tokens'] = u['output_tokens']

                    elif kind == 'error':
                        raise RuntimeError(f'anthropic: {(ev.get("error") or {}).get("message", "stream error")}')

        yield StreamDone(stop_reason=stop_reason, usage=usage)

    #: Only used when the API will not say what a model's ceiling is.
    #: Conservative on purpose: every current model accepts at least this,
    #: so being wrong costs a shorter answer rather than a refused request.
    #: A number ABOVE the real ceiling is a 400, which is a broken adapter
    #: rather than a short answer -- so the fallback errs downwards.
    _FALLBACK_MAX_OUTPUT = 8192

    async def _output_limit(self, model: str) -> int:
        """The model's own output ceiling, from ``GET /v1/models/{id}``.

        ``max_tokens`` on that response is the OUTPUT cap and
        ``max_input_tokens`` is the context window -- two different fields,
        and reading the wrong one would ask for an output the size of the
        whole window.

        Remembered per model: a model's ceiling does not change, and a
        lookup on every turn would put a round trip in front of every
        reply.
        """
        cached = self._output_limits.get(model)
        if cached is not None:
            return cached
        limit = self._FALLBACK_MAX_OUTPUT
        try:
            async with self.transport.session(10) as session:
                async with session.get(
                    f'{self.base_url}/models/{model}', headers=self._headers()
                ) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        reported = body.get('max_tokens')
                        if isinstance(reported, int) and reported > 0:
                            limit = reported
        except Exception:
            # A capability lookup that fails is a reason to be conservative
            # about length, never a reason to fail somebody's turn.
            pass
        self._output_limits[model] = limit
        return limit

    async def models(self) -> list[dict[str, Any]]:
        async with self.transport.session(30) as session:
            async with session.get(f'{self.base_url}/models', headers=self._headers()) as resp:
                if resp.status != 200:
                    return []
                body = await resp.json()
        return [{'id': m['id'], 'provider': self.provider_id} for m in body.get('data', []) if m.get('id')]
