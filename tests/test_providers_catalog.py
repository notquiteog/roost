"""The catalogue, the connections, and how a model gets chosen.

Three things that are cheap to get wrong and expensive to notice: an embedding
model whose dimensions are recorded incorrectly (the store silently stops
matching), a key written back out to a client that should never see it, and a
chat route that lands on a model which cannot hold a conversation.
"""

from __future__ import annotations

import json

import pytest

from openmirror.providers import catalog, connections
from openmirror.providers.base import Modality
from openmirror.providers.connections import Connection, ConnectionStore
from openmirror.providers.registry import can_serve, pick_model

# -- the catalogue ----------------------------------------------------------


def test_every_preset_has_somewhere_to_go():
    for host in catalog.HOSTS:
        assert host.base_url.startswith(('http://', 'https://')), host.id
        assert host.adapter, host.id
        assert host.modalities, host.id


def test_the_five_named_embedding_models_are_there():
    """The ones this was asked for by name. If one is renamed, say so loudly."""
    for wanted in (
        'text-embedding-3-large',
        'gemini-embedding-2',
        'voyage-3-large',
        'qwen3-embedding-8b',
        'qwen3-embedding-4b',
    ):
        assert wanted in catalog.EMBEDDING_BY_ID, wanted


def test_every_embedding_model_declares_its_dimensions():
    """The one property that cannot change under an existing store.

    Vectors of different lengths are not comparable, so a wrong number here is
    a memory that silently stops matching rather than one that errors.
    """
    for model in catalog.EMBEDDING_MODELS:
        assert model.dimensions > 0, model.id
        assert model.served_by, model.id


def test_a_model_can_be_found_by_the_name_its_host_uses():
    """`qwen3-embedding:8b` on Ollama is `Qwen/Qwen3-Embedding-8B` on vLLM."""
    found = catalog.describe_embedding('qwen3-embedding:8b')
    assert found is not None and found.id == 'qwen3-embedding-8b'
    assert catalog.describe_embedding('Qwen/Qwen3-Embedding-8B') is found


def test_the_query_prefix_goes_only_where_it_belongs():
    """Asymmetric models want an instruction on queries and not on passages.

    Doing it to both is the same as doing it to neither, and doing it to a
    model that does not want it is worse than not doing it at all.
    """
    assert catalog.query_prefix('gemini-embedding-2')
    assert catalog.query_prefix('qwen3-embedding:4b')
    # OpenAI's are symmetric, and Gemini's 001 takes a taskType parameter
    # instead — neither should get the prefix.
    assert catalog.query_prefix('text-embedding-3-large') == ''
    assert catalog.query_prefix('gemini-embedding-001') == ''
    assert catalog.query_prefix('something-nobody-has-heard-of') == ''


def test_a_service_with_no_model_list_says_so():
    voyage = catalog.HOSTS_BY_ID['voyage']
    assert voyage.lists_models is False
    assert voyage.fallback_models, 'it has to offer something'


# -- connections ------------------------------------------------------------


def test_a_connection_never_hands_back_its_key():
    conn = Connection(id='x', label='X', adapter='openai', base_url='http://x', api_key='sk-secret')
    shown = conn.redacted()
    assert 'api_key' not in shown
    assert shown['has_key'] is True
    assert 'sk-secret' not in json.dumps(shown)


def test_a_connection_survives_being_written_and_read(tmp_path):
    store = ConnectionStore(tmp_path / 'connections.json')
    store.put(Connection(id='groq', label='Groq', adapter='openai',
                         base_url='https://api.groq.com/openai/v1', api_key='k',
                         modalities=['chat'], tor=True))

    back = store.get('groq')
    assert back is not None
    assert back.api_key == 'k'
    assert back.tor is True
    assert back.transport().proxy is not None


def test_the_connections_file_is_not_world_readable(tmp_path):
    """It holds keys. Nothing is encrypted and pretending otherwise would be
    worse than saying so, but the file mode is free."""
    path = tmp_path / 'connections.json'
    ConnectionStore(path).put(Connection(id='x', label='X', adapter='openai', base_url='http://x'))
    assert path.stat().st_mode & 0o077 == 0


def test_an_unreadable_connections_file_is_not_fatal(tmp_path):
    """A daemon whose environment-configured providers are fine must still start."""
    path = tmp_path / 'connections.json'
    path.write_text('{ this is not json')
    assert ConnectionStore(path).load() == []


def test_a_preset_becomes_a_connection():
    conn = connections.from_host('groq', api_key='k')
    assert conn.adapter == 'openai'
    assert conn.base_url.startswith('https://api.groq.com')
    assert 'chat' in conn.modalities


def test_a_connection_is_narrowed_to_what_its_adapter_can_do():
    """Ticking a modality an adapter has no implementation for must not register
    a provider that fails on its first call."""
    conn = Connection(id='x', label='X', adapter='anthropic', base_url='http://x',
                      modalities=['chat', 'image', 'video'])
    info, impls = connections.build(conn)
    assert set(impls) == {Modality.CHAT}
    assert info.id == 'x'


def test_a_second_api_behind_the_same_key_gets_its_own_adapter():
    """OpenAI's video endpoint is not OpenAI-shaped chat, so the video slot on
    an OpenAI connection is Sora rather than the chat adapter with a different
    method called on it."""
    from openmirror.providers.openai_compat import OpenAICompatProvider
    from openmirror.providers.sora import SoraProvider

    _, impls = connections.build(connections.from_host('openai'))
    assert isinstance(impls[Modality.CHAT], OpenAICompatProvider)
    assert isinstance(impls[Modality.IMAGE], OpenAICompatProvider)
    assert isinstance(impls[Modality.VIDEO], SoraProvider)
    # And the same object serves every modality the one adapter does handle,
    # which is what makes one Tor toggle cover all of them.
    assert impls[Modality.CHAT] is impls[Modality.IMAGE]


def test_voyage_is_embedding_only():
    _, impls = connections.build(connections.from_host('voyage'))
    assert set(impls) == {Modality.EMBEDDING}


def test_google_serves_media_from_a_different_adapter_than_embeddings():
    """One key, three request shapes: `:embedContent`, `:predict` and
    `:predictLongRunning` are not the same API and must not be one object."""
    from openmirror.providers.google import GoogleProvider
    from openmirror.providers.google_media import GoogleMediaProvider
    from openmirror.providers.openai_compat import OpenAICompatProvider

    _, impls = connections.build(connections.from_host('google'))
    # Chat too, which the catalogue always offered for this host and `build`
    # used to narrow away in silence — a Gemini connection could embed and
    # draw but not be talked to. It goes through Google's OpenAI-compatible
    # endpoint under the same base URL, a fourth shape on the same key.
    assert set(impls) == {Modality.CHAT, Modality.EMBEDDING, Modality.IMAGE, Modality.VIDEO}
    assert isinstance(impls[Modality.CHAT], OpenAICompatProvider)
    assert impls[Modality.CHAT].base_url.endswith('/v1beta/openai')
    assert isinstance(impls[Modality.EMBEDDING], GoogleProvider)
    assert isinstance(impls[Modality.IMAGE], GoogleMediaProvider)
    # Image and video are two instances, not one: they hit different endpoints
    # and the kind cannot be recovered from the model id.
    assert impls[Modality.IMAGE] is not impls[Modality.VIDEO]
    assert impls[Modality.IMAGE].kind == 'image'
    assert impls[Modality.VIDEO].kind == 'video'


def test_every_catalogued_media_host_actually_builds():
    """A preset that names an adapter with no implementation behind it is a
    connect button that fails after the key has been typed in."""
    from openmirror.providers.catalog import HOSTS

    for host in HOSTS:
        wanted = host.modalities & {Modality.IMAGE, Modality.VIDEO}
        if not wanted:
            continue
        _, impls = connections.build(connections.from_host(host.id, api_key='k:s'))
        assert wanted <= set(impls), f'{host.id} presents {wanted} and builds {set(impls)}'


# -- picking a model --------------------------------------------------------


def test_an_embedding_model_is_never_chosen_for_chat():
    """Seen on a real install: `/api/tags` listed the embedding model first, so
    every conversation came back as an HTTP 400 saying that model does not
    support chat — which reads as a broken daemon rather than a bad default."""
    models = [
        {'id': 'qwen3-embedding:4b', 'capabilities': ['embedding']},
        {'id': 'gemma4:12b', 'capabilities': ['completion', 'tools', 'vision']},
    ]
    assert pick_model(models, Modality.CHAT) == 'gemma4:12b'
    assert pick_model(models, Modality.EMBEDDING) == 'qwen3-embedding:4b'


def test_a_model_that_cannot_call_tools_is_not_chosen_for_an_agent():
    models = [
        {'id': 'plain:7b', 'capabilities': ['completion']},
        {'id': 'toolful:7b', 'capabilities': ['completion', 'tools']},
    ]
    assert pick_model(models, Modality.CHAT, need_tools=True) == 'toolful:7b'
    # Without the requirement, first usable wins.
    assert pick_model(models, Modality.CHAT) == 'plain:7b'


def test_a_provider_that_declares_nothing_is_not_narrowed_to_nothing():
    """Only Ollama reports capabilities. Every OpenAI-shaped server returns bare
    ids, and treating "did not say" as "cannot" would leave those installs
    unable to pick anything at all."""
    models = [{'id': 'gpt-5'}, {'id': 'gpt-5-mini'}]
    assert can_serve(models[0], Modality.CHAT) is True
    assert pick_model(models, Modality.CHAT, need_tools=True) == 'gpt-5'


def test_no_usable_model_is_an_empty_answer_not_a_wrong_one():
    assert pick_model([{'id': 'e', 'capabilities': ['embedding']}], Modality.CHAT) == ''
    assert pick_model([], Modality.CHAT) == ''


# -- embeddings, end to end -------------------------------------------------


class Recorder:
    """Remembers how it was called, which is the whole of what is under test."""

    def __init__(self):
        self.calls = []

    async def embed(self, texts, model, *, input_type='document', dimensions=None):
        self.calls.append({'texts': texts, 'model': model,
                           'input_type': input_type, 'dimensions': dimensions})
        return [[0.0] * 8 for _ in texts]


@pytest.mark.asyncio
async def test_a_query_is_embedded_as_a_query(tmp_path):
    """Storing and asking are different operations for these models, and the
    memory service is the only layer that knows which is happening."""
    import tempfile

    from openmirror.memory.service import MemoryService
    from openmirror.memory.store import MemoryStore
    from openmirror.providers.base import ProviderInfo
    from openmirror.providers.registry import ProviderRegistry, Route, RouteSet

    registry = ProviderRegistry()
    recorder = Recorder()
    registry.register(
        ProviderInfo(id='r', label='R', modalities={Modality.EMBEDDING}, local=True),
        {Modality.EMBEDDING: recorder},
    )
    registry.set_defaults(RouteSet(routes={
        Modality.EMBEDDING: Route(provider='r', model='qwen3-embedding:8b',
                                  options={'dimensions': 1024}),
    }))

    with tempfile.TemporaryDirectory() as tmp:
        service = MemoryService(MemoryStore(f'{tmp}/m.db'), registry)
        service.set_settings('alice', enabled=True, auto_capture=False)

        await service.remember('alice', 'deploys go out on Thursdays')
        stored = recorder.calls[-1]
        assert stored['input_type'] == 'document'
        assert not stored['texts'][0].startswith('Instruct:'), 'a passage must go in bare'
        assert stored['dimensions'] == 1024, 'the route said 1024'

        await service.recall('alice', 'when do deploys go out')
        asked = recorder.calls[-1]
        assert asked['input_type'] == 'query'
        assert asked['texts'][0].startswith('Instruct:'), 'a query wants the instruction'


# -- where Ollama reports what a model can do -------------------------------


def test_capabilities_are_read_from_wherever_the_build_puts_them():
    """Builds disagree, and reading only one place is indistinguishable from
    the model declaring nothing — which `can_serve` treats as "assume it can",
    which is how an embedding model gets picked for chat."""
    from openmirror.providers.ollama import _capabilities

    assert _capabilities({'capabilities': ['embedding']}) == ['embedding']
    assert _capabilities({'details': {'capabilities': ['tools']}}) == ['tools']
    assert _capabilities({'details': {}}) == []

    # An empty list at the top is not an answer, so it falls through rather
    # than shadowing a nested one that has something to say.
    assert _capabilities({'capabilities': [], 'details': {'capabilities': ['tools']}}) == ['tools']


@pytest.mark.asyncio
async def test_a_model_the_listing_did_not_classify_is_asked_about(monkeypatch):
    """`/api/tags` officially carries no capability field at all. Where a build
    honours that, every model comes back unclassified and the fallback is the
    only thing standing between an embedding model and the chat route."""
    from openmirror.providers.ollama import OllamaProvider

    provider = OllamaProvider('http://example.invalid')
    asked: list[str] = []

    async def fake_tags():
        return {'models': [{'name': 'chatty:7b'}, {'name': 'embedder:1b'}]}

    async def fake_show(model):
        asked.append(model)
        return ['embedding'] if 'embed' in model else ['completion', 'tools']

    monkeypatch.setattr(provider, '_show_capabilities', fake_show)
    monkeypatch.setattr(
        provider, 'models',
        lambda: _models_with(provider, fake_tags, OllamaProvider.models),
    )

    rows = await provider.models()
    by_id = {r['id']: r['capabilities'] for r in rows}
    assert by_id['embedder:1b'] == ['embedding']
    assert by_id['chatty:7b'] == ['completion', 'tools']
    assert sorted(asked) == ['chatty:7b', 'embedder:1b'], 'both were unclassified'

    assert pick_model(rows, Modality.CHAT, need_tools=True) == 'chatty:7b'


async def _models_with(provider, fake_tags, real_models):
    """Run the real `models()` against a stubbed HTTP layer.

    Written this way rather than with a mock session because the thing under
    test is the fallback logic, not aiohttp.
    """
    import types

    body = await fake_tags()

    class _Resp:
        status = 200

        async def json(self):
            return body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Session:
        def get(self, *a, **kw):
            return _Resp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    stub = types.SimpleNamespace(session=lambda *a, **kw: _Session())
    original = provider.transport
    provider.transport = stub
    try:
        return await real_models(provider)
    finally:
        provider.transport = original


@pytest.mark.asyncio
async def test_asking_twice_only_costs_one_request(monkeypatch):
    """A picker opening is not a reason to re-ask about every model, and a
    server with no `/api/show` must not be asked forever."""
    from openmirror.providers.ollama import OllamaProvider

    provider = OllamaProvider('http://example.invalid')
    calls: list[str] = []

    class _Failing:
        def session(self, *a, **kw):
            calls.append('tried')
            raise OSError('no such endpoint')

    provider.transport = _Failing()

    assert await provider._show_capabilities('x') == []
    assert await provider._show_capabilities('x') == []
    assert len(calls) == 1, 'a failure is remembered, not retried on every call'
