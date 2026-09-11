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
import re
from collections.abc import AsyncIterator
from typing import Any

from openmirror.net.transport import Transport
from openmirror.providers import reasoning
from openmirror.providers.base import (
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
    refused,
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



#: Keys of `ChatRequest.extra` this adapter converts itself, and so must not
#: also splat into the payload verbatim.
_TRANSLATED_EXTRAS = reasoning.TRANSLATED_EXTRAS

#: Models that still accept temperature/top_p/top_k. The current generation
#: does not: those parameters were removed and are a 400, not an ignore.
_SAMPLING_MODELS = re.compile(r'^claude-(3|opus-4-[0-6]|sonnet-4|haiku-4)', re.I)


def _takes_sampling(model: str) -> bool:
    return bool(_SAMPLING_MODELS.match(model or ''))


def reasoning_params(req: ChatRequest, *, capabilities: Any = None, max_tokens: int = 0) -> dict[str, Any]:
    """The reasoning and sampling half of the request body.

    Its own function so it can be checked without an endpoint. Every rule here
    is one that fails silently or hard, and none of them is visible in a
    response body — which is the argument for testing it directly rather than
    trusting a live call to have exercised it.

    What moved under the older shape most code was written to, and is now
    decided per model family in `openmirror.providers.reasoning`:

    * ``{'type': 'enabled', 'budget_tokens': N}`` is a 400 on the current
      models — and REQUIRED on Haiku 4.5, which rejects adaptive thinking
      outright. Sending adaptive to every model made openmirror unusable on it.
    * ``display`` defaults to ``omitted``, a silent change from Opus 4.6.
      Without asking for ``summarized``, thinking blocks still arrive and still
      bill — with empty text, and the working-out panel sits blank.
    * A disabled thinking type is the wrong way to turn reasoning off in an
      agent: the model sometimes writes a tool call into its VISIBLE TEXT
      instead of a tool_use block, and the turn succeeds with nothing run. So
      ``off`` — which the voice path asks for, because reasoning is silence in
      a call — is low effort with the working-out hidden.
    * Opus 4.6 has no ``xhigh``; the level clamps down to one it has.

    ``capabilities`` is the Models API's tree for this model, when known; it
    overrides the id-based table, so a model released later is still asked in
    a form it accepts.
    """
    level = reasoning.request_level(req.effort, req.extra)
    r = reasoning.anthropic_reasoning(
        req.model, level, capabilities=capabilities, max_tokens=max_tokens,
        takes_sampling=_takes_sampling(req.model),
    )
    out = dict(r['body'])
    # Sampling was REMOVED on the current models — temperature, top_p and top_k
    # are each a 400 rather than being ignored — and on the older ones it is a
    # 400 while the model thinks. Sent only where both allow it.
    if req.temperature is not None and r['sampling']:
        out['temperature'] = req.temperature
    return out


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
        # Output ceiling and capability tree, per model. See _model_info.
        self._model_infos: dict[str, tuple[int, Any]] = {}

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

        limit, capabilities = await self._model_info(req.model)
        params = reasoning_params(req, capabilities=capabilities, max_tokens=req.max_tokens or limit)
        # A budget-only model (Haiku 4.5) refuses a thinking budget that does
        # not fit under max_tokens with room left for the answer.
        budget = (params.get('thinking') or {}).get('budget_tokens') or 0
        payload: dict[str, Any] = {
            'model': req.model,
            'messages': messages,
            # Required by this API -- there is no omitting it, so this is the
            # one adapter that cannot express "no ceiling" by absence. It sends
            # the model's own maximum, asked for once and remembered.
            'max_tokens': max(req.max_tokens or limit, budget + 1024 if budget else 0),
            'stream': True,
        }
        if req.system:
            payload['system'] = req.system

        payload.update(params)
        if req.stop:
            payload['stop_sequences'] = req.stop
        if req.tools:
            payload['tools'] = [
                {'name': t.name, 'description': t.description, 'input_schema': t.input_schema} for t in req.tools
            ]
        # The escape hatch, minus the keys this adapter has already translated.
        # `think` is Ollama's spelling and has no meaning here: passed through
        # it becomes an unknown top-level parameter, which this API answers
        # with a 400 -- so the voice path talking to Anthropic failed outright.
        payload.update({k: v for k, v in req.extra.items() if k not in _TRANSLATED_EXTRAS})

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

    async def _model_info(self, model: str) -> tuple[int, Any]:
        """The model's own output ceiling and capability tree, from ``GET /v1/models/{id}``.

        ``max_tokens`` on that response is the OUTPUT cap and
        ``max_input_tokens`` is the context window -- two different fields,
        and reading the wrong one would ask for an output the size of the
        whole window. ``capabilities`` says which thinking shapes and effort
        levels the model takes.

        Remembered per model once it has answered: a model's ceiling does not
        change, and a lookup on every turn would put a round trip in front of
        every reply. A failed lookup is not remembered, so one timeout does
        not pin the conservative fallback for the life of the process.
        """
        cached = self._model_infos.get(model)
        if cached is not None:
            return cached
        limit, capabilities, answered = self._FALLBACK_MAX_OUTPUT, None, False
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
                        if isinstance(body.get('capabilities'), dict):
                            capabilities = body['capabilities']
                        answered = True
        except Exception:
            # A capability lookup that fails is a reason to be conservative
            # about length, never a reason to fail somebody's turn.
            pass
        if answered:
            self._model_infos[model] = (limit, capabilities)
        return limit, capabilities

    async def models(self) -> list[dict[str, Any]]:
        async with self.transport.session(30) as session:
            async with session.get(f'{self.base_url}/models', headers=self._headers()) as resp:
                if resp.status in (401, 403):
                    # Refused, and said so. Returned as [] this read as "nothing
                    # pulled yet" and the Test button reported it as a success.
                    raise refused(self.provider_id, self.base_url, resp.status)
                if resp.status != 200:
                    return []
                body = await resp.json()
        return [{'id': m['id'], 'provider': self.provider_id} for m in body.get('data', []) if m.get('id')]
