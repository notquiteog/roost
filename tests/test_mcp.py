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

from roost.agent.tools.base import ToolContext
from roost.agent.tools.mcp import mcp_tools, qualified_name
from roost.mcp.client import MCPError, StdioServer
from roost.mcp.manager import MCPManager, ServerConfig, load_config
from roost.protocol.agent import Risk

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
