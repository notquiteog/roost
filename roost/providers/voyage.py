"""Voyage AI embeddings.

Small adapter, one endpoint, and one detail that is easy to get wrong and
expensive to notice: Voyage's models are trained asymmetrically. A passage
being stored and a question being asked are embedded with different
`input_type` values, and using one for both costs recall quality in a way that
looks like the model simply being worse than advertised. So the caller says
which it is doing, and the default is `document` — the storing case, which is
the one that happens in bulk.

There is no model-listing endpoint, so `models()` returns what
`catalog.Host.fallback_models` documents and marks it as not having come from
the service. That distinction is carried through to the UI rather than being
smoothed over: a list the server did not confirm can contain a name the server
will reject.
"""

from __future__ import annotations

from typing import Any

from roost.net.transport import Transport
from roost.providers.base import EmbeddingProvider


class VoyageProvider(EmbeddingProvider):
    def __init__(
        self,
        base_url: str = 'https://api.voyageai.com/v1',
        api_key: str = '',
        *,
        provider_id: str = 'voyage',
        timeout: int = 120,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.provider_id = provider_id
        self.timeout = timeout
        self.transport = transport or Transport(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        h = {'Content-Type': 'application/json'}
        if self.api_key:
            h['Authorization'] = f'Bearer {self.api_key}'
        return h

    async def embed(
        self,
        texts: list[str],
        model: str,
        *,
        input_type: str = 'document',
        dimensions: int | None = None,
    ) -> list[list[float]]:
        payload: dict[str, Any] = {'model': model, 'input': texts, 'input_type': input_type}
        if dimensions:
            payload['output_dimension'] = dimensions

        async with self.transport.session() as session:
            async with session.post(
                f'{self.base_url}/embeddings', json=payload, headers=self._headers()
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f'voyage: HTTP {resp.status}: {(await resp.text())[:300]}')
                body = await resp.json()

        # Sorted by index rather than trusted in order: the API documents the
        # index field, and a reordered batch pairs every vector with the wrong
        # text — which produces a memory store that is wrong rather than empty.
        return [d['embedding'] for d in sorted(body['data'], key=lambda d: d.get('index', 0))]

    async def models(self) -> list[dict[str, Any]]:
        from roost.providers.catalog import HOSTS_BY_ID

        host = HOSTS_BY_ID.get('voyage')
        return [
            {'id': m, 'provider': self.provider_id, 'source': 'catalogue'}
            for m in (host.fallback_models if host else ())
        ]
