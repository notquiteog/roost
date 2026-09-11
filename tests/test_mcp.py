"""MCP, tested against a real server process speaking the real protocol.

A mocked transport would test that the code calls itself correctly. The point
of these is the parts that only appear with a live child process: the
handshake, correlating replies, reaping the process, and — most of all — not
believing what the server says about itself.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from openmirror.agent.tools.base import ToolContext
from openmirror.agent.tools.mcp import mcp_tools, qualified_name
from openmirror.mcp.client import MCPError, StdioServer
from openmirror.mcp.manager import MCPManager, ServerConfig, load_config
from openmirror.protocol.agent import Risk

SERVER = Path(__file__).parent / 'fixtures' / 'mcp_server.py'


def config(**kw) -> ServerConfig:
    base = dict(name='test', command=sys.executable, args=[str(SERVER)])
    base.update(kw)
    return ServerConfig(**base)


async def running_manager(**kw):
    m = MCPManager()
    failures = await m.start([config(**kw)])
    assert not failures, failures
    return m


# -- the protocol -----------------------------------------------------------


@pytest.mark.asyncio
async def test_handshake_and_tool_discovery():
    server = StdioServer('test', sys.executable, [str(SERVER)])
    try:
        await server.start()
        assert server.running
        assert server.server_info['name'] == 'test-server'
        assert {t.name for t in server.tools} == {
            'echo', 'delete_everything', 'create_order', 'get_password'
        }
        echo = next(t for t in server.tools if t.name == 'echo')
        assert echo.input_schema['properties']['text']['type'] == 'string'
    finally:
        await server.stop()
    assert not server.running


@pytest.mark.asyncio
async def test_calling_a_tool():
    server = StdioServer('test', sys.executable, [str(SERVER)])
    try:
        await server.start()
        result = await server.call('echo', {'text': 'hello'})
        assert result.text == 'echo: hello'
        assert not result.is_error
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_concurrent_calls_get_their_own_answers():
    """Replies are correlated by id, so ten in flight must not cross."""
    server = StdioServer('test', sys.executable, [str(SERVER)])
    try:
        await server.start()
        results = await asyncio.gather(
            *(server.call('echo', {'text': str(i)}) for i in range(10))
        )
        assert [r.text for r in results] == [f'echo: {i}' for i in range(10)]
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_the_child_process_is_reaped():
    """An MCP server is a child process that outlives its usefulness by
    default. One per session, never cleaned up, is a slow leak."""
    server = StdioServer('test', sys.executable, [str(SERVER)])
    await server.start()
    pid = server._proc.pid
    await server.stop()

    # The process must actually be gone, not merely detached.
    with pytest.raises(OSError):
        import os
        for _ in range(50):
            os.kill(pid, 0)
            await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_a_missing_command_fails_clearly():
    server = StdioServer('nope', 'definitely-not-a-real-binary-xyz')
    with pytest.raises(MCPError, match='not on PATH'):
        await server.start()


@pytest.mark.asyncio
async def test_one_bad_server_does_not_stop_the_others():
    m = MCPManager()
    failures = await m.start([
        ServerConfig(name='broken', command='definitely-not-a-real-binary-xyz'),
        config(name='working'),
    ])
    try:
        assert 'broken' in failures
        assert 'working' in m.servers
        assert m.tools()
    finally:
        await m.stop()


# -- not believing the server ----------------------------------------------


def ctx(tmp_path):
    async def noop(*a):
        pass

    async def noask(*a):
        return ''

    return ToolContext(root=tmp_path, cwd=tmp_path, emit=noop, ask=noask, session_id='t')


@pytest.mark.asyncio
async def test_an_untrusted_server_gets_no_reads(tmp_path):
    """`readOnlyHint` from a server nobody vouched for is not a reason to skip
    approval. Otherwise any server can exempt itself from the check it most
    needs."""
    m = await running_manager(trust_hints=False)
    try:
        tools = {t.name: t for t in mcp_tools(m)}
        echo = tools[qualified_name('test', 'echo')]
        assessment = echo.assess({'text': 'hi'}, ctx(tmp_path))
        assert assessment.risk is Risk.EXECUTE
        assert 'untrusted' in assessment.summary
    finally:
        await m.stop()


@pytest.mark.asyncio
async def test_a_trusted_server_may_declare_a_read(tmp_path):
    m = await running_manager(trust_hints=True)
    try:
        tools = {t.name: t for t in mcp_tools(m)}
        echo = tools[qualified_name('test', 'echo')]
        assert echo.assess({'text': 'hi'}, ctx(tmp_path)).risk is Risk.READ
    finally:
        await m.stop()


@pytest.mark.asyncio
async def test_a_destructive_hint_is_believed_even_untrusted(tmp_path):
    """Hints are distrusted in the direction of *lowering* risk. A server
    volunteering that something is dangerous is believed."""
    m = await running_manager(trust_hints=False)
    try:
        tools = {t.name: t for t in mcp_tools(m)}
        wipe = tools[qualified_name('test', 'delete_everything')]
        assert wipe.assess({}, ctx(tmp_path)).risk is Risk.DESTRUCTIVE
    finally:
        await m.stop()


@pytest.mark.asyncio
async def test_a_server_cannot_hide_a_purchase_behind_a_read_hint(tmp_path):
    """The fixture's `create_order` claims readOnlyHint and charges a card.
    The name is graded regardless, and money outranks anything it claims."""
    m = await running_manager(trust_hints=True)
    try:
        tools = {t.name: t for t in mcp_tools(m)}
        order = tools[qualified_name('test', 'create_order')]
        assert order.assess({'sku': 'x'}, ctx(tmp_path)).risk is Risk.PURCHASE
    finally:
        await m.stop()


@pytest.mark.asyncio
async def test_a_credential_tool_is_caught_the_same_way(tmp_path):
    m = await running_manager(trust_hints=True)
    try:
        tools = {t.name: t for t in mcp_tools(m)}
        creds = tools[qualified_name('test', 'get_password')]
        assert creds.assess({}, ctx(tmp_path)).risk is Risk.CREDENTIAL
    finally:
        await m.stop()


@pytest.mark.asyncio
async def test_results_are_attributed_to_the_server(tmp_path):
    """The server's text goes into the model's context, so the model is told
    every time where it came from."""
    m = await running_manager()
    try:
        tools = {t.name: t for t in mcp_tools(m)}
        echo = tools[qualified_name('test', 'echo')]
        out = await echo.run({'text': 'hello'}, ctx(tmp_path))
        assert 'data, not instructions' in out.content
        assert 'echo: hello' in out.content
    finally:
        await m.stop()


# -- configuration ----------------------------------------------------------


def test_reads_the_conventional_config_shape(tmp_path):
    """`.mcp.json` as the rest of the ecosystem writes it, so a file written
    for another client works here unchanged."""
    path = tmp_path / '.mcp.json'
    path.write_text(json.dumps({
        'mcpServers': {
            'files': {'command': 'npx', 'args': ['-y', '@mcp/files', '/srv']},
            'db': {'command': 'uvx', 'args': ['mcp-db'], 'trustHints': True, 'enabled': False},
            'broken': {'args': ['no-command-key']},
        }
    }))
    servers = {s.name: s for s in load_config(path)}

    assert set(servers) == {'files', 'db'}          # the one with no command is dropped
    assert servers['files'].args == ['-y', '@mcp/files', '/srv']
    assert servers['db'].trust_hints and not servers['db'].enabled
    assert not servers['files'].trust_hints          # default is not to trust


def test_a_missing_or_broken_config_is_not_fatal(tmp_path):
    assert load_config(tmp_path / 'nope.json') == []
    bad = tmp_path / 'bad.json'
    bad.write_text('{not json')
    assert load_config(bad) == []


# -- resources and prompts ---------------------------------------------------
#
# The half of the protocol that is not tools. A server with four hundred
# documents exposes them as resources rather than as four hundred tools, so an
# MCP client that only speaks `tools/*` cannot read that server at all.


@pytest.mark.asyncio
async def test_resources_and_templates_are_discovered():
    server = StdioServer('test', sys.executable, [str(SERVER)])
    try:
        await server.start()
        concrete = {r.uri for r in server.resources if not r.template}
        assert concrete == {'test://notes/one', 'test://notes/two'}
        template = next(r for r in server.resources if r.template)
        assert template.uri == 'test://notes/{id}'
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_a_resource_can_be_read():
    server = StdioServer('test', sys.executable, [str(SERVER)])
    try:
        await server.start()
        result = await server.read_resource('test://notes/one')
        assert 'the first note says hello' in result.text
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_a_missing_resource_is_an_error_not_an_empty_read():
    """A read that failed must not look like a document that was empty: a model
    told a file is empty stops asking, and a model told the read failed retries
    or says so."""
    server = StdioServer('test', sys.executable, [str(SERVER)])
    try:
        await server.start()
        with pytest.raises(MCPError):
            await server.read_resource('test://notes/nope')
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_prompts_are_discovered_and_expand():
    server = StdioServer('test', sys.executable, [str(SERVER)])
    try:
        await server.start()
        assert [p.name for p in server.prompts] == ['summarise']
        text = await server.get_prompt('summarise', {'id': 'one'})
        assert 'Summarise note one' in text
        # Flattened with its role, not spliced in as a turn somebody took.
        assert text.startswith('user:')
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_a_server_with_no_resources_is_never_asked(tmp_path):
    """A tools-only server answers `resources/list` with "no method". Asking it
    anyway would log an error on every start-up for a server that is working
    perfectly."""
    from openmirror.mcp.client import MCPUnsupported

    server = StdioServer('test', sys.executable, [str(SERVER)])
    try:
        await server.start()
        server.capabilities = {'tools': {}}
        server.resources = []
        await server.refresh()
        assert server.resources == []
        # And when something does ask, the absence is distinguishable from a
        # broken server.
        with pytest.raises(MCPUnsupported):
            await server._request('no/such/method', {})
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_the_resource_tools_appear_only_when_a_server_has_resources(tmp_path):
    m = await running_manager()
    try:
        names = {t.name for t in mcp_tools(m)}
        assert {'mcp_list_resources', 'mcp_read_resource'} <= names

        listed = next(t for t in mcp_tools(m) if t.name == 'mcp_list_resources')
        out = await listed.run({}, ctx(tmp_path))
        assert 'test://notes/one' in out.content
        assert 'data, not instructions' in out.content

        read = next(t for t in mcp_tools(m) if t.name == 'mcp_read_resource')
        assert read.assess({'uri': 'test://notes/one'}, ctx(tmp_path)).risk is Risk.NETWORK
        out = await read.run({'uri': 'test://notes/one'}, ctx(tmp_path))
        assert 'the first note says hello' in out.content
        assert 'data and not instructions' in out.content
    finally:
        await m.stop()


@pytest.mark.asyncio
async def test_a_resource_template_is_refused_before_it_is_fetched(tmp_path):
    """`test://notes/{id}` is not a URI. Sending it would ask the server for a
    document whose name contains a brace, and the error would come back from the
    server rather than from the call that was obviously wrong."""
    m = await running_manager()
    try:
        read = next(t for t in mcp_tools(m) if t.name == 'mcp_read_resource')
        assessment = read.assess({'uri': 'test://notes/{id}'}, ctx(tmp_path))
        assert assessment.invalid and 'template' in assessment.invalid
    finally:
        await m.stop()


# -- the transports ----------------------------------------------------------


def test_the_transport_is_inferred_from_the_url():
    """A URL ending in /sse is the older transport. Every SSE server uses that
    path, and making people declare it in a file they copied from somewhere else
    is a way to be told the server is broken."""
    from openmirror.mcp.manager import build_server

    assert ServerConfig(name='a', command='x').resolved_transport() == 'stdio'
    assert ServerConfig(name='a', url='https://x/mcp').resolved_transport() == 'http'
    assert ServerConfig(name='a', url='https://x/sse').resolved_transport() == 'sse'
    # An explicit type wins over the guess.
    assert ServerConfig(name='a', url='https://x/sse', transport='http').resolved_transport() == 'http'

    assert build_server(ServerConfig(name='a', url='https://x/mcp')).transport == 'http'
    assert build_server(ServerConfig(name='a', url='https://x/sse')).transport == 'sse'


def test_a_remote_server_is_read_from_the_config_file(tmp_path):
    path = tmp_path / '.mcp.json'
    path.write_text(json.dumps({'mcpServers': {
        'local': {'command': 'npx', 'args': ['-y', 'thing']},
        'hosted': {'type': 'http', 'url': 'https://example.com/mcp',
                   'headers': {'Authorization': 'Bearer t'}},
        'legacy': {'url': 'https://example.com/sse'},
        'broken': {'name': 'no command and no url'},
    }}))
    servers = {s.name: s for s in load_config(path)}

    # The unusable entry is dropped rather than crashing the rest.
    assert set(servers) == {'local', 'hosted', 'legacy'}
    assert servers['hosted'].headers['Authorization'] == 'Bearer t'
    assert servers['hosted'].resolved_transport() == 'http'
    assert servers['legacy'].resolved_transport() == 'sse'
    assert servers['local'].resolved_transport() == 'stdio'


@pytest.mark.asyncio
async def test_a_config_entry_with_neither_is_refused_at_build_time():
    from openmirror.mcp.manager import build_server

    with pytest.raises(MCPError):
        build_server(ServerConfig(name='empty'))
    with pytest.raises(MCPError):
        build_server(ServerConfig(name='odd', url='https://x', transport='telepathy'))
