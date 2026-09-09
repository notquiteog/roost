"""Google's generative language API, for the Gemini embedding models.

Only embeddings. Gemini *chat* is reachable through the generic OpenAI adapter
against `/v1beta/openai`, which Google serves and keeps current, so a second
chat translation here would be a maintenance burden bought with nothing.
Embeddings are the exception: the OpenAI-shaped path exposes neither
`outputDimensionality` nor the query/document distinction, and both matter —
the first is what makes a 3072-dimension model storable at 768, and the second
is worth real recall.

Two details are easy to get wrong and both were, here, before being checked
against the current documentation:

**The key goes in a header.** `x-goog-api-key`, not `?key=`. A credential in a
query string is a credential in every access log, proxy log and error body on
the path — including the ones this adapter would otherwise have had to
truncate to avoid echoing it back.

**`taskType` is not a universal parameter.** `gemini-embedding-001` takes it;
`gemini-embedding-2` does not, and expresses the same asymmetry as an
instruction inside the text. Sending it anyway to a model that does not take
it is how a working configuration turns into a 400 on upgrade, so it is sent
only where the catalogue says it belongs — and the prefix, for models that
want one, is applied by the caller that knows whether this is a query or a
passage. See `roost.providers.catalog.query_prefix`.
"""

from __future__ import annotations

from typing import Any

from roost.net.transport import Transport
from roost.providers.base import EmbeddingProvider

# Google names a task rather than an input type. The two that matter map onto
# the same distinction Voyage draws.
TASK_FOR = {'document': 'RETRIEVAL_DOCUMENT', 'query': 'RETRIEVAL_QUERY'}

# The models documented to accept `taskType`. A prefix match, because Google
# publishes dated and `-latest` aliases of the same weights and all of them
# take it. Anything not listed is sent without it, which is the safe direction:
# a missing task hint costs a little recall, an unknown field costs the call.
TASK_TYPE_MODELS = ('gemini-embedding-001', 'text-embedding-004', 'embedding-001')


class GoogleProvider(EmbeddingProvider):
    def __init__(
        self,
        base_url: str = 'https://generativelanguage.googleapis.com/v1beta',
        api_key: str = '',
        *,
        provider_id: str = 'google',
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
            h['x-goog-api-key'] = self.api_key
        return h

    @staticmethod
    def _qualify(model: str) -> str:
        """`gemini-embedding-2` and `models/gemini-embedding-2` both work.

        Both spellings are in circulation — the listing endpoint returns the
        long one and every example uses the short one — and a caller should
        not have to know which this is.
        """
        return model if model.startswith('models/') else f'models/{model}'

    @staticmethod
    def _takes_task_type(model: str) -> bool:
        bare = model.removeprefix('models/')
        return any(bare.startswith(known) for known in TASK_TYPE_MODELS)

    async def embed(
        self,
        texts: list[str],
        model: str,
        *,
        input_type: str = 'document',
        dimensions: int | None = None,
    ) -> list[list[float]]:
        name = self._qualify(model)
        common: dict[str, Any] = {'model': name}
        if dimensions:
            common['outputDimensionality'] = dimensions
        if self._takes_task_type(name):
            common['taskType'] = TASK_FOR.get(input_type, 'RETRIEVAL_DOCUMENT')

        payload = {
            'requests': [{**common, 'content': {'parts': [{'text': text}]}} for text in texts]
        }

        async with self.transport.session() as session:
            async with session.post(
                f'{self.base_url}/{name}:batchEmbedContents',
                json=payload,
                headers=self._headers(),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f'google: HTTP {resp.status}: {(await resp.text())[:400]}')
                body = await resp.json()

        return [e['values'] for e in body.get('embeddings', [])]

    async def models(self) -> list[dict[str, Any]]:
        """What this key can actually reach, asked of Google.

        Filtered to models that declare `embedContent`, because the same
        listing carries every chat model too, and offering those as embedding
        options produces a failure at the first store rather than at the pick.
        """
        async with self.transport.session(30) as session:
            async with session.get(f'{self.base_url}/models', headers=self._headers()) as resp:
                if resp.status != 200:
                    return []
                body = await resp.json()

        out: list[dict[str, Any]] = []
        for model in body.get('models', []):
            methods = model.get('supportedGenerationMethods') or []
            if 'embedContent' not in methods and 'batchEmbedContents' not in methods:
                continue
            name = str(model.get('name', ''))
            if not name:
                continue
            out.append(
                {
                    'id': name.removeprefix('models/'),
                    'provider': self.provider_id,
                    'label': model.get('displayName') or '',
                    'dimensions': model.get('outputDimensions'),
                }
            )
        return out
