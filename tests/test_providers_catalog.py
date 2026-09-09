"""The catalogue, the connections, and how a model gets chosen.

Three things that are cheap to get wrong and expensive to notice: an embedding
model whose dimensions are recorded incorrectly (the store silently stops
matching), a key written back out to a client that should never see it, and a
chat route that lands on a model which cannot hold a conversation.
"""

from __future__ import annotations

import json

import pytest

from roost.providers import catalog, connections
from roost.providers.base import Modality
from roost.providers.connections import Connection, ConnectionStore
from roost.providers.registry import can_serve, pick_model

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
    """Ticking video on an OpenAI-shaped host must not register a video provider
    that fails on its first call."""
    conn = Connection(id='x', label='X', adapter='openai', base_url='http://x',
                      modalities=['chat', 'video'])
    info, impls = connections.build(conn)
    assert Modality.CHAT in impls
    assert Modality.VIDEO not in impls
    assert info.id == 'x'


def test_voyage_and_google_are_embedding_only():
    for host_id in ('voyage', 'google'):
        _, impls = connections.build(connections.from_host(host_id))
        assert set(impls) == {Modality.EMBEDDING}, host_id


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

    from roost.memory.service import MemoryService
    from roost.memory.store import MemoryStore
    from roost.providers.base import ProviderInfo
    from roost.providers.registry import ProviderRegistry, Route, RouteSet

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
