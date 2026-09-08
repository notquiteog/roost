"""Ollama's native API.

Ollama also speaks OpenAI's shape, so `OpenAICompatProvider` would work
against it. This exists anyway because the native API reports things the
compatibility layer drops — which models are pulled, which are resident, and
how long each load took — and because `/api/chat` returns newline-delimited
JSON rather than SSE, which is a different parser however you dress it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import aiohttp

from roost.providers.base import (
    ChatProvider,
    ChatRequest,
    EmbeddingProvider,
    ImageBlock,
    Message,
    StreamDone,
    StreamEvent,
    StreamText,
    StreamThinking,
    StreamToolUse,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from roost.providers.control_tokens import strip_control_tokens


def _to_ollama_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({'role': 'system', 'content': system})

    for msg in messages:
        for block in msg.content:
            if isinstance(block, ToolResultBlock):
                out.append({'role': 'tool', 'content': block.content})

        text = ' '.join(b.text for b in msg.content if isinstance(b, TextBlock))
        images = [b.data for b in msg.content if isinstance(b, ImageBlock)]
        calls = [
            {'function': {'name': b.name, 'arguments': b.input}}
            for b in msg.content
            if isinstance(b, ToolUseBlock)
        ]
        if not text and not images and not calls:
            continue

        entry: dict[str, Any] = {'role': msg.role, 'content': text}
        if images:
            entry['images'] = images
        if calls:
            entry['tool_calls'] = calls
        out.append(entry)

    return out


class OllamaProvider(ChatProvider, EmbeddingProvider):
    def __init__(
        self, base_url: str = 'http://localhost:11434', api_key: str = '', *, provider_id: str = 'ollama', timeout: int = 600
    ) -> None:
        # Tolerates a base URL given with or without the /api suffix, because
        # both forms are in every guide on the internet.
        self.base_url = base_url.rstrip('/').removesuffix('/api')
        self.api_key = api_key
        self.provider_id = provider_id
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {'Content-Type': 'application/json'}
        # Vanilla Ollama has no auth. Behind Perch it is a bearer token, which
        # plain Ollama ignores — so this is always safe to send.
        if self.api_key:
            h['Authorization'] = f'Bearer {self.api_key}'
        return h

    async def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]:
        options: dict[str, Any] = {}
        if req.temperature is not None:
            options['temperature'] = req.temperature
        if req.stop:
            options['stop'] = req.stop
        if req.max_tokens:
            options['num_predict'] = req.max_tokens

        payload: dict[str, Any] = {
            'model': req.model,
            'messages': _to_ollama_messages(req.messages, req.system),
            'stream': True,
        }
        if options:
            payload['options'] = options
        if req.tools:
            payload['tools'] = [
                {
                    'type': 'function',
                    'function': {'name': t.name, 'description': t.description, 'parameters': t.input_schema},
                }
                for t in req.tools
            ]
        payload.update(req.extra)

        usage: dict[str, int] = {}
        stop_reason = 'end_turn'
        counter = 0

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f'{self.base_url}/api/chat', json=payload, headers=self._headers()) as resp:
                if resp.status != 200:
                    raise RuntimeError(f'ollama: HTTP {resp.status}: {(await resp.text())[:500]}')

                # NDJSON, one object per line, rather than SSE.
                async for raw in resp.content:
                    line = raw.decode('utf-8', 'replace').strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg = chunk.get('message') or {}
                    if msg.get('content'):
                        # Templates leak their own control tokens into content;
                        # this text is also what gets spoken aloud.
                        cleaned = strip_control_tokens(msg['content'])
                        if cleaned:
                            yield StreamText(text=cleaned)
                    if msg.get('thinking'):
                        yield StreamThinking(text=msg['thinking'])

                    for tc in msg.get('tool_calls') or []:
                        fn = tc.get('function') or {}
                        args = fn.get('arguments') or {}
                        # Ollama returns arguments already parsed, except when
                        # the model emitted a JSON string, which some do.
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        counter += 1
                        stop_reason = 'tool_use'
                        yield StreamToolUse(id=f'call_{counter}', name=fn.get('name', ''), input=args)

                    if chunk.get('done'):
                        usage = {
                            'input_tokens': chunk.get('prompt_eval_count', 0),
                            'output_tokens': chunk.get('eval_count', 0),
                        }
                        if chunk.get('done_reason') == 'length':
                            stop_reason = 'max_tokens'
                        break

        yield StreamDone(stop_reason=stop_reason, usage=usage)

    async def models(self) -> list[dict[str, Any]]:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f'{self.base_url}/api/tags', headers=self._headers()) as resp:
                if resp.status != 200:
                    return []
                body = await resp.json()
        return [
            {
                'id': m['name'],
                'provider': self.provider_id,
                'size': m.get('size'),
                'family': (m.get('details') or {}).get('family'),
            }
            for m in body.get('models', [])
            if m.get('name')
        ]

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f'{self.base_url}/api/embed', json={'model': model, 'input': texts}, headers=self._headers()
            ) as resp:
                resp.raise_for_status()
                body = await resp.json()
        return body.get('embeddings', [])
