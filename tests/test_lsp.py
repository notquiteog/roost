"""Language servers: the client, the tool, and errors after an edit.

Two layers of evidence. A small fake server (tests/fixtures/lsp_server.py)
speaks the real protocol — framing, its own requests to the client, indexing
announced with $/progress — so every path through the client runs against
something that behaves like a server rather than like the client's idea of
one. And where rust-analyzer is installed, the same questions are asked of
it, because a fake written by the author of the client shares the author's
misunderstandings.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from openmirror.agent.approval import Mode
from openmirror.agent.lsp import LspPool, ServerSpec, find_servers
from openmirror.agent.runtime import build_session
from openmirror.agent.tools.base import ToolContext, ToolError
from openmirror.agent.tools.lsp import LspTool
from openmirror.protocol.agent import Risk, ToolCompleted
from openmirror.providers.base import StreamDone, StreamText, StreamToolUse
from tests.test_agent import ScriptedProvider, turn

FAKE = Path(__file__).parent / 'fixtures' / 'lsp_server.py'
SOURCE = 'def area(r):\n    return 3 * r * r\n\n\nclass Shape:\n    pass\n\n\nx = area(2)\ny = area(3)\n'


def fake() -> ServerSpec:
    return ServerSpec(
        name='fakels', command=[sys.executable, str(FAKE)], extensions=('.py',),
        language='Python', binary=sys.executable,
    )


def ctx(root: Path) -> ToolContext:
    async def emit(text, stream):
        pass

    async def ask(question, options, multi):
        return ''

    return ToolContext(root=root, cwd=root, emit=emit, ask=ask, session_id='lsp')


@pytest.fixture
async def tool(tmp_path: Path):
    (tmp_path / 'm.py').write_text(SOURCE)
    lsp = LspTool(LspPool(tmp_path, [fake()]))
    yield lsp
    await lsp.close()


async def test_the_first_question_starts_the_server_and_is_graded_as_a_command(tool, tmp_path: Path):
    args = {'operation': 'definition', 'path': 'm.py', 'line': 9, 'symbol': 'area'}
    first = tool.assess(args, ctx(tmp_path))
    # Starting a server can run the project's build scripts, so it is asked
    # about like a command; once it is up, asking it things is a read.
    assert first.risk is Risk.EXECUTE and 'start fakels' in first.summary

    out = await tool.run(args, ctx(tmp_path))
    assert 'm.py:1:5' in out.content and 'def area(r):' in out.content
    assert tool.assess(args, ctx(tmp_path)).risk is Risk.READ


async def test_references_hover_symbols_and_search(tool, tmp_path: Path):
    c = ctx(tmp_path)
    refs = await tool.run({'operation': 'references', 'path': 'm.py', 'line': 1, 'symbol': 'area'}, c)
    assert refs.content.startswith('3 results')
    assert 'm.py:9:5' in refs.content and 'm.py:10:5' in refs.content

    hover = await tool.run({'operation': 'hover', 'path': 'm.py', 'line': 10, 'symbol': 'area'}, c)
    assert 'A thing called area.' in hover.content

    symbols = await tool.run({'operation': 'symbols', 'path': 'm.py'}, c)
    assert 'function' in symbols.content and 'class' in symbols.content and 'Shape' in symbols.content

    found = await tool.run({'operation': 'search', 'path': 'm.py', 'query': 'Sha'}, c)
    assert 'Shape  (class)' in found.content


async def test_a_name_that_is_not_on_the_line_is_refused_with_the_line(tool, tmp_path: Path):
    with pytest.raises(ToolError, match='That line is: y = area'):
        await tool.run({'operation': 'definition', 'path': 'm.py', 'line': 10, 'symbol': 'volume'}, ctx(tmp_path))


async def test_diagnostics_are_asked_for_and_reported(tool, tmp_path: Path):
    (tmp_path / 'bad.py').write_text('ok = 1\nthis is BROKEN\n')
    out = await tool.run({'operation': 'diagnostics', 'path': 'bad.py'}, ctx(tmp_path))
    assert 'bad.py:2:9  error  this line is broken  [fakels]' in out.content
    clean = await tool.run({'operation': 'diagnostics', 'path': 'm.py'}, ctx(tmp_path))
    assert clean.content == 'No problems reported in m.py.'


async def test_a_question_the_server_does_not_answer_says_what_to_do_instead(tool, tmp_path: Path):
    """The fake declares no implementationProvider, as pylsp declares no
    project-wide search. The server's own answer would be "Method Not Found"."""
    with pytest.raises(ToolError, match='does not answer implementation questions. Ask for its definition'):
        await tool.run({'operation': 'implementation', 'path': 'm.py', 'line': 1, 'symbol': 'area'}, ctx(tmp_path))


def test_a_file_no_server_knows_is_not_a_question_for_approval(tool, tmp_path: Path):
    assessment = tool.assess({'operation': 'symbols', 'path': 'notes.md'}, ctx(tmp_path))
    assert assessment.invalid and 'Python (fakels)' in assessment.invalid


async def test_an_edit_hears_what_it_broke_from_a_server_already_running(tmp_path: Path):
    (tmp_path / 'm.py').write_text(SOURCE)
    provider = ScriptedProvider([
        [StreamToolUse(id='c1', name='lsp', input={'operation': 'symbols', 'path': 'm.py'}), StreamDone(stop_reason='tool_use')],
        [StreamToolUse(id='c2', name='read_file', input={'path': 'm.py'}), StreamDone(stop_reason='tool_use')],
        [StreamToolUse(id='c3', name='edit_file', input={
            'path': 'm.py', 'old_string': 'return 3 * r * r', 'new_string': 'return BROKEN'}), StreamDone(stop_reason='tool_use')],
        [StreamText(text='Oops.'), StreamDone()],
    ])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.TRUSTED, lsp=[fake()])
    await session.start()
    try:
        events = await turn(session, 'break it')
    finally:
        await session.close()
    edit = [e for e in events if isinstance(e, ToolCompleted) and e.result.name == 'edit_file'][0]
    assert edit.result.ok
    assert 'fakels now reports 1 error in this file' in edit.result.content
    assert 'line 2: this line is broken' in edit.result.content


async def test_an_edit_never_starts_a_server_by_itself(tmp_path: Path):
    """Starting one is running the project's code, and an edit was not graded as that."""
    (tmp_path / 'm.py').write_text(SOURCE)
    provider = ScriptedProvider([
        [StreamToolUse(id='c1', name='read_file', input={'path': 'm.py'}), StreamDone(stop_reason='tool_use')],
        [StreamToolUse(id='c2', name='edit_file', input={
            'path': 'm.py', 'old_string': 'return 3 * r * r', 'new_string': 'return BROKEN'}), StreamDone(stop_reason='tool_use')],
        [StreamText(text='ok'), StreamDone()],
    ])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.TRUSTED, lsp=[fake()])
    await session.start()
    events = await turn(session, 'edit it')
    pool = session.tools['lsp'].pool
    await session.close()
    edit = [e for e in events if isinstance(e, ToolCompleted) and e.result.name == 'edit_file'][0]
    assert 'reports' not in edit.result.content
    assert not pool._servers


def test_no_server_means_no_tool(tmp_path: Path):
    session = build_session(root=tmp_path, provider=ScriptedProvider([[StreamDone()]]), model='x')
    assert 'lsp' not in session.tools


def test_a_configured_server_takes_its_extensions_and_a_disabled_one_frees_them(tmp_path: Path):
    config = tmp_path / '.lsp.json'
    config.write_text(json.dumps({'servers': {
        'mine': {'command': sys.executable, 'args': [str(FAKE)], 'extensions': ['.foo']},
        'off': {'disabled': True, 'extensions': ['.py', '.pyi']},
        'missing': {'command': 'definitely-not-installed-ls', 'extensions': ['.bar']},
    }}))
    found = {s.name: s for s in find_servers(config)}
    assert found['mine'].extensions == ('.foo',) and found['mine'].binary
    assert 'missing' not in found
    assert not any('.py' in s.extensions for s in found.values()), 'disabled means no default either'


@pytest.mark.skipif(sys.platform == 'win32', reason='a shell-script stand-in for rustup')
def test_a_rustup_proxy_with_nothing_behind_it_is_not_a_server(tmp_path: Path, monkeypatch):
    """rustup links a proxy for every tool it knows about, installed or not.
    Found on the machine this was written on: `rust-analyzer` was on PATH
    and had never been installed, so the tool was offered and failed."""
    from openmirror.agent import lsp as lsp_mod

    rustup = tmp_path / 'rustup'
    rustup.write_text('#!/bin/sh\necho "error: unknown binary" >&2\nexit 1\n')
    rustup.chmod(0o755)
    (tmp_path / 'rust-analyzer').symlink_to(rustup)
    monkeypatch.setenv('PATH', f'{tmp_path}{os.pathsep}{os.environ["PATH"]}')
    monkeypatch.setattr(lsp_mod, '_PROXIED', {})
    assert 'rust-analyzer' not in {s.name for s in find_servers(None)}


# -- the real thing ---------------------------------------------------------

# Found *and working*: a proxy with no component behind it is on PATH on most
# machines with Rust, and find_servers is what knows the difference.
RUST = shutil.which('cargo') and any(s.name == 'rust-analyzer' for s in find_servers(None))


@pytest.mark.skipif(not RUST, reason='needs rust-analyzer and cargo')
async def test_rust_analyzer_answers_the_same_questions(tmp_path: Path):
    await asyncio.to_thread(
        subprocess.run, ['cargo', 'init', '--quiet', '--name', 'shapes', str(tmp_path)], check=True, timeout=60,
    )
    (tmp_path / 'src' / 'main.rs').write_text(
        'fn area(r: f64) -> f64 {\n    3.14 * r * r\n}\n\nfn main() {\n    let a = area(2.0);\n'
        '    let b = area(3.0);\n    println!("{} {}", a, b);\n}\n'
    )
    specs = [s for s in find_servers(None) if s.name == 'rust-analyzer']
    lsp = LspTool(LspPool(tmp_path, specs))
    c = ctx(tmp_path)
    try:
        definition = await lsp.run({'operation': 'definition', 'path': 'src/main.rs', 'line': 6, 'symbol': 'area'}, c)
        assert 'src/main.rs:1:4' in definition.content, definition.content
        refs = await lsp.run({'operation': 'references', 'path': 'src/main.rs', 'line': 1, 'symbol': 'area'}, c)
        assert refs.content.startswith('3 results'), refs.content
        hover = await lsp.run({'operation': 'hover', 'path': 'src/main.rs', 'line': 7, 'symbol': 'area'}, c)
        assert 'fn area(r: f64) -> f64' in hover.content, hover.content
    finally:
        await lsp.close()
