"""Connecting Perch from the settings dialog, against a Perch this test runs.

Perch is one host, one token and up to five services on their own ports, and a
stored connection used to be exactly one provider. Every place a connection is
touched — saving it, starting up with it, testing it, removing it — had to learn
that one Perch is several, and each is asserted here.

The fake answers the way the real one does, measured against a live install:
`/healthz` open on every service and naming itself `perch`, and `/api/tags`
returning 401 for no token and for a token it never issued. That last part is
what the settings dialog now checks before it saves, because the probe cannot:
a wrong token used to be saved, reported as connected, and then fail every
request somewhere far away from the dialog.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openmirror.providers import catalog, connections
from openmirror.providers import perch as perch_mod
from openmirror.providers.connections import Connection, ConnectionStore
from openmirror.providers.registry import ProviderRegistry

GOOD = 'perch_' + 'g' * 43


def _handler(endpoint: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 - the stdlib's name
            if self.path == '/healthz':
                # Open, as on the real thing: it says something is listening
                # and nothing about the token.
                return self._send(200, {'ok': True, 'service': 'perch', 'endpoint': endpoint})
            if self.headers.get('Authorization') != f'Bearer {GOOD}':
                return self._send(401, {'error': 'unauthorised'})
            if self.path == '/api/tags':
                return self._send(200, {'models': [{'name': 'qwen3.5:9b', 'model': 'qwen3.5:9b'}]})
            return self._send(404, {'error': 'not found'})

        def log_message(self, *args):
            return

    return Handler


@pytest.fixture
def fake_perch(monkeypatch):
    """Perch with chat, voice and audio switched on, and image and video off."""
    servers = {}
    for service in ('chat', 'voice', 'audio'):
        server = ThreadingHTTPServer(('127.0.0.1', 0), _handler(service))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers[service] = server
        monkeypatch.setitem(perch_mod.DEFAULT_PORTS, service, server.server_port)
    # Ports nothing is listening on, so those two services probe as off.
    for service in ('image', 'video'):
        with ThreadingHTTPServer(('127.0.0.1', 0), _handler(service)) as closed:
            port = closed.server_port
        monkeypatch.setitem(perch_mod.DEFAULT_PORTS, service, port)
    try:
        yield servers
    finally:
        for server in servers.values():
            server.shutdown()
            server.server_close()


@pytest.fixture
def api(tmp_path, monkeypatch, fake_perch):
    """The providers router with its own registry and its own connections file."""
    from openmirror.config import config
    from openmirror.routers import providers as providers_router

    live = ProviderRegistry()
    monkeypatch.setattr(providers_router, 'registry', live)
    monkeypatch.setattr(providers_router, 'store', ConnectionStore(tmp_path / 'connections.json'))
    monkeypatch.setattr(config, 'perch_host', '')
    monkeypatch.delenv('PERCH_TOKEN', raising=False)

    app = FastAPI()
    app.include_router(providers_router.router)
    with TestClient(app) as client:
        yield client, live, providers_router.store


def save(client, **body):
    return client.put('/api/providers/connections', json={'host_id': 'perch', 'base_url': 'http://127.0.0.1', **body})


# -- the preset --------------------------------------------------------------


def test_perch_is_the_first_thing_the_add_form_offers():
    """An install with only Perch is a complete install, so it is the default
    rather than something to scroll to."""
    assert catalog.HOSTS[0].id == 'perch'
    assert catalog.HOSTS[0].local is True
    assert catalog.HOSTS[0].env_key == 'PERCH_TOKEN'


def test_the_address_is_a_host_and_a_typed_port_is_ignored():
    """`http://127.0.0.1:11434` taken literally would send dictation to Ollama."""
    for typed in ('http://127.0.0.1:11434', '127.0.0.1', 'http://127.0.0.1/', ''):
        cfg = perch_mod.from_connection(Connection(id='perch', label='', adapter='perch', base_url=typed))
        assert cfg.host == '127.0.0.1', typed
        assert cfg.url('voice').endswith(f':{perch_mod.DEFAULT_PORTS["voice"]}')


# -- saving ------------------------------------------------------------------


def test_a_good_token_connects_each_live_service_and_is_saved(api):
    client, live, store = api
    reply = save(client, api_key=GOOD)
    assert reply.status_code == 200, reply.text
    body = reply.json()

    assert sorted(body['registered']) == ['perch:audio', 'perch:chat', 'perch:voice']
    # Said, not hidden: somebody about to route video at Perch needs to know it
    # is switched off.
    assert body['services'] == {'chat': True, 'voice': True, 'audio': True, 'image': False, 'video': False}
    assert store.get('perch') is not None

    rows = {p['id']: p for p in body['providers']}
    assert set(rows) == {'perch:chat', 'perch:voice', 'perch:audio'}
    # Every row is editable and knows which connection it belongs to, which is
    # what lets Test and Remove on any of them act on the one connection.
    assert all(r['editable'] and r['connection_id'] == 'perch' for r in rows.values())


def test_a_wrong_token_is_refused_and_nothing_is_saved(api):
    client, live, store = api
    reply = save(client, api_key='perch_' + 'x' * 43)
    assert reply.status_code == 400
    assert 'refused that token' in reply.json()['detail']
    assert 'Connect page' in reply.json()['detail']
    assert store.get('perch') is None
    assert live.list() == []


def test_a_blank_key_uses_perch_token_from_the_environment(api, monkeypatch):
    """The form says "leave the key blank to use PERCH_TOKEN", so it must."""
    client, _, store = api
    monkeypatch.setenv('PERCH_TOKEN', GOOD)
    assert save(client).status_code == 200
    assert store.get('perch').api_key == GOOD


def test_nothing_answering_is_refused_with_where_to_look(api, monkeypatch):
    client, _, store = api
    for service in perch_mod.DEFAULT_PORTS:
        monkeypatch.setitem(perch_mod.DEFAULT_PORTS, service, 9)   # discard: nothing listens
    reply = save(client, api_key=GOOD)
    assert reply.status_code == 502
    assert 'nothing answered' in reply.json()['detail']
    assert store.get('perch') is None


def test_tor_is_refused_rather_than_silently_ignored(api):
    """Perch's providers carry no transport, so a Tor toggle on one would be a
    promise nothing keeps — the exact failure the toggle exists to prevent."""
    client, _, store = api
    reply = save(client, api_key=GOOD, tor=True)
    assert reply.status_code == 400
    assert 'Tor' in reply.json()['detail']
    assert store.get('perch') is None


def test_the_old_endpoint_now_saves_too(api):
    """It used to register into the running process only, so a Perch connected
    through it was gone at the next restart."""
    client, _, store = api
    reply = client.post('/api/providers/perch/connect', json={'host': '127.0.0.1', 'token': GOOD})
    assert reply.status_code == 200, reply.text
    assert store.get('perch') is not None
    assert 'capabilities' in reply.json()


# -- testing and removing ----------------------------------------------------


def test_the_test_button_checks_the_token(api):
    client, _, _ = api
    save(client, api_key=GOOD)
    result = client.post('/api/providers/connections/perch/test').json()
    assert result['ok'] is True
    assert result['models'] == 1


def test_the_test_button_works_on_a_provider_from_the_environment(api, fake_perch):
    """Env-configured rows have no stored connection, and their Test button used
    to answer "no answer" for exactly that reason."""
    client, live, _ = api
    cfg = perch_mod.PerchConfig(host='127.0.0.1', token=GOOD)
    live.register_perch(cfg, {'chat': True, 'voice': False, 'audio': False, 'image': False, 'video': False})
    result = client.post('/api/providers/connections/perch:chat/test').json()
    assert result['ok'] is True


def test_removing_perch_removes_every_one_of_its_services(api):
    client, live, store = api
    save(client, api_key=GOOD)
    assert client.delete('/api/providers/connections/perch').status_code == 200
    assert store.get('perch') is None
    assert not [p for p in live.list() if p.id.startswith('perch:')]


def test_removing_a_saved_perch_puts_the_environments_back(api, monkeypatch):
    """A saved Perch replaces the environment's, since both register the same
    ids. Removing it must not leave an install with PERCH_HOST set and no Perch."""
    from openmirror.config import config

    client, live, _ = api
    monkeypatch.setattr(config, 'perch_host', '127.0.0.1')
    monkeypatch.setattr(config, 'perch_token', GOOD)
    save(client, api_key=GOOD)
    client.delete('/api/providers/connections/perch')

    rows = {p.id: live.connection(p.id) for p in live.list()}
    assert set(rows) == {'perch:chat', 'perch:voice', 'perch:audio'}
    # Back to the environment's, which is not editable from the dialog.
    assert all(conn is None for conn in rows.values())


def test_a_service_that_went_down_is_dropped_on_reconnect(api, fake_perch, monkeypatch):
    """Reconnecting after switching a service off must not keep offering it."""
    client, live, _ = api
    save(client, api_key=GOOD)
    assert 'perch:audio' in {p.id for p in live.list()}

    fake_perch['audio'].shutdown()
    fake_perch['audio'].server_close()
    save(client)   # blank key: keeps the stored one
    assert 'perch:audio' not in {p.id for p in live.list()}


# -- starting up -------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_saved_perch_is_there_after_a_restart(tmp_path, fake_perch):
    """What the old endpoint could not do. A stored Perch is probed and each of
    its live services registered, under its own id, at start-up."""
    store = ConnectionStore(tmp_path / 'connections.json')
    store.put(connections.from_host('perch', api_key=GOOD, base_url='http://127.0.0.1'))

    live = ProviderRegistry()
    ids = await connections.register(store.get('perch'), live)
    assert sorted(ids) == ['perch:audio', 'perch:chat', 'perch:voice']
    assert live.connection('perch:chat').id == 'perch'


def test_building_perch_gives_every_service_it_could_offer():
    """Build is the shape of a connection and never touches the network; the
    probe is register's job. Every modality the preset claims has an adapter."""
    _, impls = connections.build(connections.from_host('perch', api_key=GOOD))
    assert set(impls) >= set(catalog.HOSTS_BY_ID['perch'].modalities)


# -- a refused key is a failure, not "0 models" ------------------------------
#
# How this was found: connections saved as plain Ollama and ComfyUI, pointed at
# Perch's ports with no key, both reported "ok — 0 models" from the Test
# button. Perch had answered 401 to both. The Ollama adapter turned the 401 into
# an empty list, and the ComfyUI one never asked the server at all.


def test_a_refused_key_is_a_no_provider_error():
    """So every existing handler — the agent route, both sockets — shows it as a
    message instead of learning a new type."""
    from openmirror.providers.base import NoProviderError, ProviderRefused
    from openmirror.providers.registry import NoProviderError as from_registry

    assert issubclass(ProviderRefused, NoProviderError)
    assert from_registry is NoProviderError


def test_an_ollama_connection_at_perch_without_a_key_fails_its_test(api, fake_perch):
    """The exact connection that was sitting in a real install, reporting success."""
    client, _, _ = api
    port = perch_mod.DEFAULT_PORTS['chat']
    saved = client.put('/api/providers/connections', json={
        'host_id': 'ollama', 'base_url': f'http://127.0.0.1:{port}',
    })
    assert saved.status_code == 200, saved.text

    result = client.post('/api/providers/connections/ollama/test').json()
    assert result['ok'] is False
    assert 'refused the key (HTTP 401)' in result['error']
    # And it says what is probably going on, since stock Ollama has no keys.
    assert 'connect it as Perch' in result['error']


def test_a_comfyui_connection_is_tested_against_the_server(api, tmp_path):
    """ComfyUI's models are local templates, so its Test used to succeed without
    leaving this machine. It now asks the server's queue, with the key."""
    client, _, _ = api
    server = ThreadingHTTPServer(('127.0.0.1', 0), _handler('video'))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f'http://127.0.0.1:{server.server_port}'
        client.put('/api/providers/connections', json={'host_id': 'comfyui', 'base_url': url})
        refused = client.post('/api/providers/connections/comfyui/test').json()
        assert refused['ok'] is False
        assert 'HTTP 401' in refused['error']

        client.put('/api/providers/connections', json={'host_id': 'comfyui', 'base_url': url, 'api_key': GOOD})
        # The queue path is not one the fake serves, so a good key reaches a
        # 404 — which is still the server answering, and still not "ok".
        answered = client.post('/api/providers/connections/comfyui/test').json()
        assert 'refused' not in answered.get('error', '')
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_the_agent_says_the_key_was_refused_not_that_there_are_no_models(fake_perch):
    """What a person starting a session saw before: "reports no models"."""
    from openmirror.providers.base import ProviderRefused
    from openmirror.providers.ollama import OllamaProvider

    port = perch_mod.DEFAULT_PORTS['chat']
    with pytest.raises(ProviderRefused) as caught:
        await OllamaProvider(f'http://127.0.0.1:{port}', '', provider_id='ollama').models()
    assert 'HTTP 401' in str(caught.value)

    # With the key, the same call lists the model.
    listed = await OllamaProvider(f'http://127.0.0.1:{port}', GOOD, provider_id='ollama').models()
    assert [m['id'] for m in listed] == ['qwen3.5:9b']


@pytest.mark.asyncio
async def test_a1111_says_the_key_was_refused(fake_perch):
    """The fifth adapter with the same bug, reached through a shared helper that
    turned every failure — a refused key included — into None."""
    from openmirror.providers.a1111 import A1111Provider
    from openmirror.providers.base import ProviderRefused

    server = ThreadingHTTPServer(('127.0.0.1', 0), _handler('image'))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(ProviderRefused):
            await A1111Provider(f'http://127.0.0.1:{server.server_port}', '', provider_id='a1111').models()
    finally:
        server.shutdown()
        server.server_close()
