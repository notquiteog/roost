"""Ollama's native API.

Ollama also speaks OpenAI's shape, so `OpenAICompatProvider` would work
against it. This exists anyway because the native API reports things the
compatibility layer drops — which models are pulled, which are resident, and
how long each load took — and because `/api/chat` returns newline-delimited
JSON rather than SSE, which is a different parser however you dress it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from roost.net.transport import Transport
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

log = logging.getLogger(__name__)


def _capabilities(model: dict[str, Any]) -> list[str]:
    """What a `/api/tags` row says the model can do, wherever it says it.

    Two places, because builds disagree: this machine's Ollama reports at the
    top level and others nest it under `details`. Reading only one returns an
    empty list from the other, and an empty list means "no idea" — which is
    indistinguishable from a model that genuinely declares nothing.
    """
    top = model.get('capabilities')
    if top:
        return list(top)
    nested = (model.get('details') or {}).get('capabilities')
    return list(nested) if nested else []


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
        self,
        base_url: str = 'http://localhost:11434',
        api_key: str = '',
        *,
        provider_id: str = 'ollama',
        timeout: int = 600,
        transport: Transport | None = None,
    ) -> None:
        # Tolerates a base URL given with or without the /api suffix, because
        # both forms are in every guide on the internet.
        self.base_url = base_url.rstrip('/').removesuffix('/api')
        self.api_key = api_key
        self.provider_id = provider_id
        self.timeout = timeout
        self.transport = transport or Transport(timeout=timeout)
        # model -> capabilities, from `/api/show`. Per instance, which is per
        # connection, so two servers with the same model name never share an
        # answer about it.
        self._capability_cache: dict[str, list[str]] = {}

    def _session(self, timeout: float | None = None):
        """Every request goes through the transport, without exception.

        Ollama took a transport and then built its own session in all three of
        its methods, which meant a connection with the Tor toggle on
        registered, probed, showed a "tor" tag in the UI — and sent every
        message, every model listing and every embedding straight out. That is
        the exact failure the toggle exists to prevent, and it is worse than
        not having the feature: the operator believes they are on Tor.

        Anything added here uses this. There is no cheaper path.
        """
        return self.transport.session(timeout)

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

        async with self._session() as session:
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
        """What is pulled, and what each one can be asked to do.

        The capability field is the interesting part, and where it lives is
        not settled. This Ollama reports it at the top level of `/api/tags`;
        others put it under `details`; the documentation says `/api/tags`
        carries none at all and that `/api/show` is the source. All three are
        handled, because the consequence of finding nothing is not "no
        information" — it is `can_serve` assuming a model can do anything,
        which is how an embedding model gets chosen for chat and every
        conversation comes back as an HTTP 400 naming the model rather than
        the picker.
        """
        async with self._session(30) as session:
            async with session.get(f'{self.base_url}/api/tags', headers=self._headers()) as resp:
                if resp.status != 200:
                    return []
                body = await resp.json()
        listed = [
            {
                'id': m['name'],
                'provider': self.provider_id,
                'size': m.get('size'),
                'family': (m.get('details') or {}).get('family'),
                # What this model can be asked to do, as Ollama reports it:
                # 'completion', 'tools', 'vision', 'thinking', 'embedding'.
                # Carried because it is the only reliable way to tell an
                # embedding model from a chat one — the names do not, and
                # picking wrong produces a 400 that reads as a broken install.
                'capabilities': _capabilities(m),
            }
            for m in body.get('models', [])
            if m.get('name')
        ]

        # Anything the listing did not classify is asked about directly. Cached
        # for the life of the process: capabilities cannot change without the
        # model being pulled again, and this is otherwise a request per model
        # every time a picker opens. Bounded, so a machine with forty models
        # pulled does not open forty sockets at once.
        unknown = [row for row in listed if not row['capabilities']]
        if unknown:
            gate = asyncio.Semaphore(5)

            async def fill(row: dict[str, Any]) -> None:
                async with gate:
                    row['capabilities'] = await self._show_capabilities(row['id'])

            await asyncio.gather(*(fill(row) for row in unknown), return_exceptions=True)

        return listed

    async def _show_capabilities(self, model: str) -> list[str]:
        """`/api/show` for one model, remembered.

        Failures are cached as "nothing known" rather than retried on every
        call: a server that does not implement this endpoint would otherwise
        be asked about every model, every time, forever.
        """
        if model in self._capability_cache:
            return self._capability_cache[model]

        found: list[str] = []
        try:
            async with self._session(30) as session:
                async with session.post(
                    f'{self.base_url}/api/show', json={'model': model}, headers=self._headers()
                ) as resp:
                    if resp.status == 200:
                        found = (await resp.json()).get('capabilities') or []
        except Exception as exc:  # noqa: BLE001 - an endpoint that is not there is an answer
            log.debug('%s: /api/show for %s did not answer: %s', self.provider_id, model, exc)

        self._capability_cache[model] = found
        return found

    async def embed(
        self,
        texts: list[str],
        model: str,
        *,
        input_type: str = 'document',
        dimensions: int | None = None,
    ) -> list[list[float]]:
        # Ollama takes neither an input type nor an output dimension. Both are
        # accepted and ignored so that swapping an embedding provider is a
        # configuration change rather than a code change at the call site.
        async with self._session() as session:
            async with session.post(
                f'{self.base_url}/api/embed', json={'model': model, 'input': texts}, headers=self._headers()
            ) as resp:
                resp.raise_for_status()
                body = await resp.json()
        return body.get('embeddings', [])
