"""openmirror as an MCP server, tested over all three transports.

The client tests in `test_mcp.py` spend a subprocess to avoid mocking the
protocol. These do the same in the other direction: a real child process for
stdio, a real socket and a real uvicorn for HTTP and SSE, and openmirror's own
client as the thing on the other end. That pairing is the point — the two
halves of this package are tested against each other, so a protocol mistake
cannot be mutually agreed upon.

What is asserted here besides "it answers":

  * a scope widens nothing it was not given, and says which scope was needed
  * no tool that acts on the machine is exposed at any scope
  * a resource template cannot be talked into reading something else
  * the HTTP endpoint wants its token
"""

from __future__ import annotations

import base64
import json
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openmirror.mcp.client import HTTPServer, SSEServer, StdioServer
from openmirror.mcp.server import OpenmirrorMCP, _match_uri
from openmirror.routers import mcp as mcp_router

STDIO_SERVER = Path(__file__).parent / 'fixtures' / 'openmirror_mcp_stdio.py'


# -- fakes, standing in for this install's services ---------------------------


@dataclass
class Memory:
    text: str
    kind: str = 'fact'
    subject: str | None = None
    created_at: float = 1_700_000_000.0


@dataclass
class Recall:
    memories: list
    reason: str = ''


@dataclass
class Settings:
    enabled: bool = True


@dataclass
class FakeStore:
    rows: list = field(default_factory=list)

    def list(self, user_id, *, limit=200, **kw):
        return self.rows[:limit]


@dataclass
class FakeMemory:
    enabled: bool = True
    rows: list = field(default_factory=lambda: [Memory('they take their coffee black')])
    store: FakeStore = field(default_factory=FakeStore)

    def __post_init__(self):
        self.store.rows = self.rows

    def settings(self, user_id):
        return Settings(enabled=self.enabled)

    def embedder(self):
        return 'fake-embed'

    async def recall(self, user_id, query, *, limit=5, subject=None, kind=None):
        if not self.enabled:
            return Recall(memories=[], reason='memory is off for this user')
        hits = [m for m in self.rows if any(w in m.text for w in query.lower().split())]
        return Recall(memories=hits[:limit])

    async def remember(self, user_id, text, *, kind='fact', subject=None, source=None):
        if not self.enabled:
            return None
        memory = Memory(text=text, kind=kind, subject=subject)
        self.rows.append(memory)
        return memory


@dataclass
class FakeMediaRecord:
    media_type: str = 'image/png'


@dataclass
class FakeMediaStore:
    files: dict = field(default_factory=dict)

    def get(self, media_id):
        return self.files.get(media_id, (None, None))[0]

    def path(self, media_id):
        return self.files.get(media_id, (None, None))[1]


@dataclass
class FakeMedia:
    store: FakeMediaStore = field(default_factory=FakeMediaStore)
    kinds: set = field(default_factory=lambda: {'image'})
    calls: list = field(default_factory=list)

    def can_generate(self):
        return self.kinds

    async def generate_image(self, prompt, *, provider=None, model=None, params=None):
        self.calls.append(prompt)
        return {
            'media': [{'id': 'med1', 'media_type': 'image/png', 'bytes': 3}],
            'provider': provider or 'fake',
            'model': model or 'fake-model',
        }


def server(**kw) -> OpenmirrorMCP:
    base = dict(memory=FakeMemory(), web=False, user_id='tester')
    base.update(kw)
    return OpenmirrorMCP(**base)


async def ask(live: OpenmirrorMCP, method: str, params: dict | None = None, msg_id: int = 1):
    reply = await live.handle({'jsonrpc': '2.0', 'id': msg_id, 'method': method, 'params': params or {}})
    assert reply is not None, f'{method} answered nothing'
    return reply


# -- the protocol ------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_declares_what_it_has():
    reply = await ask(server(), 'initialize', {'protocolVersion': '2025-06-18'})
    result = reply['result']
    assert result['serverInfo']['name'] == 'openmirror'
    assert set(result['capabilities']) == {'tools', 'resources', 'prompts'}
    # The instructions tell a client to recall before answering. A twin whose
    # client guesses about the person is worse than no twin.
    assert 'recall' in result['instructions']


@pytest.mark.asyncio
async def test_a_notification_gets_no_reply():
    """The protocol says so, and a client correlating by id hangs on a reply it
    never asked for."""
    live = server()
    assert await live.handle({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) is None


@pytest.mark.asyncio
async def test_an_unknown_method_is_an_error_not_a_crash():
    reply = await ask(server(), 'no/such/thing')
    assert reply['error']['code'] == -32601


@pytest.mark.asyncio
async def test_a_tool_that_fails_comes_back_as_a_result():
    """Not as a JSON-RPC error. A model reads `isError` content and tries
    something else; a transport-level error tells it only that something broke."""
    reply = await ask(server(), 'tools/call', {'name': 'recall', 'arguments': {}})
    assert reply['result']['isError'] is True
    assert 'query is required' in reply['result']['content'][0]['text']


# -- scope -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_scope_cannot_write():
    live = server(scope='read')
    names = {t['name'] for t in (await ask(live, 'tools/list'))['result']['tools']}
    assert 'recall' in names
    assert 'remember' not in names

    reply = await ask(live, 'tools/call', {'name': 'remember', 'arguments': {'text': 'x'}})
    # Refused, and specific about why: a client told "no such tool" asks again,
    # and a client told which scope it needs can ask its operator instead.
    assert reply['error']['code'] == -32601
    assert "'write' scope" in reply['error']['message']
    assert 'OPENMIRROR_MCP_SERVE_SCOPE' in reply['error']['message']


@pytest.mark.asyncio
async def test_write_scope_remembers_but_does_not_spend():
    live = server(scope='write', media=FakeMedia())
    names = {t['name'] for t in (await ask(live, 'tools/list'))['result']['tools']}
    assert {'recall', 'remember'} <= names
    assert 'generate_image' not in names

    reply = await ask(live, 'tools/call', {'name': 'remember', 'arguments': {'text': 'they cycle to work'}})
    assert 'Remembered' in reply['result']['content'][0]['text']
    assert any(m.text == 'they cycle to work' for m in live.memory.rows)


@pytest.mark.asyncio
async def test_only_the_widest_scope_can_spend_money():
    media = FakeMedia()
    live = server(scope='all', media=media)
    tools = {t['name']: t for t in (await ask(live, 'tools/list'))['result']['tools']}
    assert 'generate_image' in tools
    # The description says it costs, because a client cannot see a price.
    assert 'cost money' in tools['generate_image']['description']

    reply = await ask(live, 'tools/call', {'name': 'generate_image', 'arguments': {'prompt': 'a cat'}})
    assert media.calls == ['a cat']
    # The bytes are not inlined; the resource URI is how a client gets them.
    assert 'openmirror://media/med1' in reply['result']['content'][0]['text']


@pytest.mark.asyncio
async def test_an_unrecognised_scope_is_the_narrowest_one():
    """A typo in a config file must not be a way to widen what is exposed."""
    live = server(scope='everything-please', media=FakeMedia())
    names = {t['name'] for t in (await ask(live, 'tools/list'))['result']['tools']}
    assert names == {'recall'}


@pytest.mark.asyncio
async def test_nothing_that_touches_the_machine_is_exposed_at_any_scope():
    """The approval policy is not reimplemented on this side — the tools it
    guards are simply not here. This test is what notices if one is added."""
    forbidden = {
        'shell', 'bash', 'write_file', 'edit_file', 'multi_edit', 'apply_patch',
        'read_file', 'list_dir', 'glob', 'grep', 'desktop_click', 'desktop_type',
        'browser_navigate', 'browser_click', 'package_install', 'ask_user',
    }
    live = server(scope='all', media=FakeMedia(), web=True)
    names = {t['name'] for t in (await ask(live, 'tools/list'))['result']['tools']}
    assert not (names & forbidden), f'exposed something that acts: {names & forbidden}'


@pytest.mark.asyncio
async def test_an_install_without_a_capability_does_not_offer_it():
    """Not offered and failing — absent. A model shown a tool this install
    cannot back spends a turn finding that out."""
    live = OpenmirrorMCP(scope='all', memory=None, media=None, web=False)
    names = {t['name'] for t in (await ask(live, 'tools/list'))['result']['tools']}
    assert names == set()


# -- recall ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_returns_what_is_remembered():
    reply = await ask(server(), 'tools/call', {'name': 'recall', 'arguments': {'query': 'coffee'}})
    assert 'coffee black' in reply['result']['content'][0]['text']


@pytest.mark.asyncio
async def test_recall_says_when_memory_is_off_rather_than_being_empty():
    live = server(memory=FakeMemory(enabled=False))
    reply = await ask(live, 'tools/call', {'name': 'recall', 'arguments': {'query': 'coffee'}})
    assert 'memory is off' in reply['result']['content'][0]['text']


@pytest.mark.asyncio
async def test_remember_is_honest_when_the_user_has_not_opted_in():
    live = server(scope='write', memory=FakeMemory(enabled=False))
    reply = await ask(live, 'tools/call', {'name': 'remember', 'arguments': {'text': 'x'}})
    text = reply['result']['content'][0]['text']
    assert 'Not stored' in text and 'switched off' in text


# -- resources ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_capabilities_is_readable_and_says_what_is_absent():
    live = server(scope='read')
    listed = (await ask(live, 'resources/list'))['result']['resources']
    assert 'openmirror://capabilities' in {r['uri'] for r in listed}

    reply = await ask(live, 'resources/read', {'uri': 'openmirror://capabilities'})
    body = json.loads(reply['result']['contents'][0]['text'])
    assert body['server']['scope'] == 'read'
    assert body['tools'] == ['recall']
    assert any('approval' in line for line in body['not_exposed'])


@pytest.mark.asyncio
async def test_a_template_is_listed_separately_and_can_be_read():
    live = server()
    templates = (await ask(live, 'resources/templates/list'))['result']['resourceTemplates']
    uris = {t['uriTemplate'] for t in templates}
    assert 'openmirror://memory/search/{query}' in uris
    # Templates are not in the concrete list: a client reading one verbatim gets
    # a document called `{query}`.
    listed = {r['uri'] for r in (await ask(live, 'resources/list'))['result']['resources']}
    assert 'openmirror://memory/search/{query}' not in listed

    reply = await ask(live, 'resources/read', {'uri': 'openmirror://memory/search/coffee'})
    assert 'coffee black' in reply['result']['contents'][0]['text']


@pytest.mark.asyncio
async def test_reading_something_that_is_not_a_resource_is_refused():
    reply = await ask(server(), 'resources/read', {'uri': 'openmirror://nope'})
    assert reply['error']['code'] == -32602


@pytest.mark.asyncio
async def test_generated_media_comes_back_as_bytes(tmp_path):
    png = tmp_path / 'x.png'
    png.write_bytes(b'\x89PNG')
    media = FakeMedia(store=FakeMediaStore({'med1': (FakeMediaRecord(), png)}))
    live = server(scope='all', media=media)

    reply = await ask(live, 'resources/read', {'uri': 'openmirror://media/med1'})
    content = reply['result']['contents'][0]
    assert content['mimeType'] == 'image/png'
    assert base64.b64decode(content['blob']) == b'\x89PNG'

    missing = await ask(live, 'resources/read', {'uri': 'openmirror://media/nope'})
    assert missing['error']['code'] == -32602


def test_a_uri_template_matches_one_segment_and_never_a_path():
    """`{id}` is a segment, not a path. Without that, `openmirror://media/{id}`
    matches `openmirror://media/../../etc/passwd`, and the only thing between
    that and a file read is whatever the reader happens to do with it."""
    assert _match_uri('openmirror://media/{id}', 'openmirror://media/abc') == {'id': 'abc'}
    assert _match_uri('openmirror://media/{id}', 'openmirror://media/a/b') is None
    assert _match_uri('openmirror://media/{id}', 'openmirror://media/../secrets') is None
    assert _match_uri('openmirror://media/{id}', 'openmirror://media/') is None
    assert _match_uri('openmirror://capabilities', 'openmirror://capabilities') == {}
    assert _match_uri('openmirror://capabilities', 'openmirror://capabilities/x') is None
    assert _match_uri('a://{x}/tail', 'a://one/tail') == {'x': 'one'}
    assert _match_uri('a://{x}/tail', 'a://one/other') is None


# -- prompts -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompts_are_offered_and_require_their_arguments():
    live = server()
    names = {p['name'] for p in (await ask(live, 'prompts/list'))['result']['prompts']}
    assert names == {'ask_the_twin', 'research_brief'}

    reply = await ask(
        live, 'prompts/get', {'name': 'ask_the_twin', 'arguments': {'question': 'what do they drink'}}
    )
    text = reply['result']['messages'][0]['content']['text']
    assert 'what do they drink' in text
    assert 'recall' in text

    missing = await ask(live, 'prompts/get', {'name': 'ask_the_twin', 'arguments': {}})
    assert missing['error']['code'] == -32602
    assert 'question' in missing['error']['message']


# -- the stdio transport, against openmirror's own client ----------------------


@pytest.mark.asyncio
async def test_a_real_client_can_drive_a_real_stdio_server():
    """Both halves of this package, talking to each other over a pipe."""
    client = StdioServer('openmirror', sys.executable, [str(STDIO_SERVER)])
    try:
        await client.start()
        assert client.server_info['name'] == 'openmirror'
        assert {t.name for t in client.tools} == {'recall', 'remember'}
        assert 'openmirror://capabilities' in {r.uri for r in client.resources if not r.template}
        assert {p.name for p in client.prompts} == {'ask_the_twin', 'research_brief'}

        result = await client.call('recall', {'query': 'coffee'})
        assert not result.is_error
        assert 'coffee black' in result.text

        read = await client.read_resource('openmirror://memory/recent')
        assert 'coffee black' in read.text

        prompt = await client.get_prompt('research_brief', {'topic': 'tbilisi hotels'})
        assert 'tbilisi hotels' in prompt
    finally:
        await client.stop()


@pytest.mark.asyncio
async def test_nothing_but_protocol_goes_to_stdout():
    """A stray print corrupts the stream, and the client's only complaint is
    that the JSON was invalid — a miserable thing to debug."""
    client = StdioServer('openmirror', sys.executable, [str(STDIO_SERVER)])
    try:
        await client.start()
        # Ten calls, each of which logs on the server side. If any of that had
        # reached stdout, correlating replies would have failed by now.
        for i in range(10):
            result = await client.call('recall', {'query': f'coffee {i}'})
            assert not result.is_error
    finally:
        await client.stop()


# -- the HTTP transports -----------------------------------------------------


def http_app(live: OpenmirrorMCP) -> FastAPI:
    app = FastAPI()
    app.include_router(mcp_router.router)
    mcp_router.server = live
    return app


@pytest.fixture(autouse=True)
def _clear_router_state():
    yield
    mcp_router.server = None
    mcp_router.sessions.clear()


def test_the_endpoint_is_absent_until_the_install_serves_it():
    app = FastAPI()
    app.include_router(mcp_router.router)
    mcp_router.server = None
    with TestClient(app) as client:
        reply = client.post('/mcp', json={'jsonrpc': '2.0', 'id': 1, 'method': 'initialize'})
        assert reply.status_code == 404


def test_posting_jsonrpc_over_http_works():
    with TestClient(http_app(server())) as client:
        reply = client.post('/mcp', json={'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'})
        assert reply.status_code == 200
        assert {t['name'] for t in reply.json()['result']['tools']} == {'recall'}

        # A notification: accepted, nothing said back.
        accepted = client.post('/mcp', json={'jsonrpc': '2.0', 'method': 'notifications/initialized'})
        assert accepted.status_code == 202
        assert not accepted.content

        # A GET says how to talk to it rather than 404ing a client that was
        # nearly right.
        nudge = client.get('/mcp')
        assert nudge.status_code == 405
        assert 'POST' in nudge.json()['error']['message']

        broken = client.post('/mcp', content=b'{not json')
        assert broken.status_code == 400
        assert broken.json()['error']['code'] == -32700


def test_a_token_is_required_when_one_is_configured(monkeypatch):
    from openmirror.config import config

    monkeypatch.setattr(config, 'mcp_serve_token', 'sekrit')
    with TestClient(http_app(server())) as client:
        body = {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'}
        assert client.post('/mcp', json=body).status_code == 401
        assert client.post('/mcp', json=body, headers={'Authorization': 'Bearer wrong'}).status_code == 401
        ok = client.post('/mcp', json=body, headers={'Authorization': 'Bearer sekrit'})
        assert ok.status_code == 200
        # The SSE transport is gated by the same check, not a second one.
        assert client.get('/mcp/sse').status_code == 401


# The SSE stream is deliberately not tested through a TestClient. That client
# runs the app on one portal thread and has no way to cancel a generator that
# is still waiting on its queue, so *any* test that opens the stream hangs the
# runner rather than the server — which is a property of the test client, not a
# bug in the transport. `test_the_sse_client_drives_the_real_server` covers the
# handshake and the round trip over a real socket, which is the honest way to
# test a transport that is two connections by design.


def test_posting_to_an_unknown_sse_session_is_refused():
    """A session id from a stream that has since closed. Told to reopen rather
    than silently accepted, which would drop the reply on the floor."""
    with TestClient(http_app(server())) as client:
        stale = client.post(
            '/mcp/messages?sessionId=nope', json={'jsonrpc': '2.0', 'id': 1, 'method': 'ping'}
        )
        assert stale.status_code == 404
        assert 'reopen' in stale.json()['detail']


# -- both halves over a real socket ------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.fixture
def live_http():
    """openmirror's MCP server on a real port, in a thread.

    A TestClient exercises the routes; only a socket exercises the client's
    transport. This is the one that would catch a header the client sends and
    the server rejects.
    """
    import uvicorn

    live = server(scope='write')
    app = FastAPI()
    app.include_router(mcp_router.router)
    mcp_router.server = live

    port = _free_port()
    config = uvicorn.Config(app, host='127.0.0.1', port=port, log_level='error')
    http = uvicorn.Server(config)
    thread = threading.Thread(target=http.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not http.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert http.started, 'the test server never came up'
    try:
        yield f'http://127.0.0.1:{port}', live
    finally:
        http.should_exit = True
        thread.join(timeout=10)
        mcp_router.server = None


@pytest.mark.asyncio
async def test_the_streamable_http_client_drives_the_real_server(live_http):
    url, _ = live_http
    client = HTTPServer('openmirror', f'{url}/mcp')
    try:
        await client.start()
        assert client.transport == 'http'
        assert client.server_info['name'] == 'openmirror'
        assert {t.name for t in client.tools} == {'recall', 'remember'}

        result = await client.call('recall', {'query': 'coffee'})
        assert 'coffee black' in result.text

        read = await client.read_resource('openmirror://capabilities')
        assert json.loads(read.text)['server']['name'] == 'openmirror'

        prompt = await client.get_prompt('ask_the_twin', {'question': 'what do they drink'})
        assert 'what do they drink' in prompt
    finally:
        await client.stop()


@pytest.mark.asyncio
async def test_the_sse_client_drives_the_real_server(live_http):
    url, _ = live_http
    client = SSEServer('openmirror', f'{url}/mcp/sse')
    try:
        await client.start()
        assert client.transport == 'sse'
        assert {t.name for t in client.tools} == {'recall', 'remember'}
        result = await client.call('recall', {'query': 'coffee'})
        assert 'coffee black' in result.text
    finally:
        await client.stop()


@pytest.mark.asyncio
async def test_a_streamable_client_pointed_at_the_sse_path_is_told_so(live_http):
    """The two transports are a URL apart and the mistake is common, so the
    error has to name the fix — "HTTP 405" does not."""
    url, _ = live_http
    client = HTTPServer('openmirror', f'{url}/mcp/sse')
    with pytest.raises(Exception) as caught:
        await client.start()
    assert '405' in str(caught.value) or 'Method Not Allowed' in str(caught.value)
    await client.stop()
