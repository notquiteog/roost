"""Jupyter notebooks: read as cells, pictures and all, and edited a cell at a time."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmirror.agent.approval import Mode
from openmirror.agent.runtime import build_session
from openmirror.agent.tools.base import ToolContext, ToolError
from openmirror.agent.tools.files import ReadTool
from openmirror.agent.tools.notebook import NotebookEditTool
from openmirror.providers.base import ImageBlock, StreamDone, StreamText, StreamToolUse
from tests.test_agent import ScriptedProvider, turn

PIXEL = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='

NOTEBOOK = {
    'cells': [
        {'cell_type': 'markdown', 'id': 'intro', 'metadata': {}, 'source': ['# Area\n', 'Of a circle.']},
        {
            'cell_type': 'code', 'execution_count': 3, 'id': 'calc', 'metadata': {'tags': ['keep']},
            'outputs': [
                {'name': 'stdout', 'output_type': 'stream', 'text': ['hello\n']},
                # Base64 wrapped across lines, as notebooks written by some tools have it.
                {'data': {'image/png': PIXEL[:40] + '\n' + PIXEL[40:], 'text/plain': ['<Figure>']},
                 'metadata': {}, 'output_type': 'display_data'},
                {'ename': 'ValueError', 'evalue': 'bad', 'output_type': 'error',
                 'traceback': ['\x1b[31mValueError\x1b[0m: bad']},
            ],
            'source': ['r = 2\n', 'print("hello")'],
        },
    ],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
    'nbformat': 4,
    'nbformat_minor': 5,
}


def write(root: Path, name: str = 'area.ipynb', nb: dict | None = None) -> Path:
    path = root / name
    path.write_text(json.dumps(nb or NOTEBOOK, indent=1) + '\n')
    return path


def ctx(root: Path, session: str = 'nb') -> ToolContext:
    async def emit(text, stream):
        pass

    async def ask(question, options, multi):
        return ''

    return ToolContext(root=root, cwd=root, emit=emit, ask=ask, session_id=session)


async def test_a_notebook_reads_as_its_cells_with_its_pictures(tmp_path: Path):
    write(tmp_path)
    out = await ReadTool().run({'path': 'area.ipynb'}, ctx(tmp_path))

    assert 'cell 0 · id intro · markdown' in out.content
    assert 'cell 1 · id calc · code · ran [3]' in out.content
    assert 'print("hello")' in out.content
    assert 'hello' in out.content and 'image 1, attached' in out.content
    # The traceback, without the terminal colour codes it was stored with.
    assert 'ValueError: bad' in out.content and '\x1b' not in out.content
    # Not the JSON: no escaped newlines, no quoted source lines.
    assert '\\n' not in out.content
    # And the picture goes to the model as a picture.
    assert out.images == [(PIXEL, 'image/png')]


async def test_replacing_a_code_cell_clears_what_the_old_code_printed(tmp_path: Path):
    path = write(tmp_path)
    c = ctx(tmp_path)
    await ReadTool().run({'path': 'area.ipynb'}, c)
    out = await NotebookEditTool().run(
        {'path': 'area.ipynb', 'cell': 'calc', 'source': 'r = 3\nprint(3.14 * r * r)'}, c
    )
    nb = json.loads(path.read_text())
    cell = nb['cells'][1]
    assert cell['source'] == ['r = 3\n', 'print(3.14 * r * r)']
    assert cell['outputs'] == [] and cell['execution_count'] is None
    assert cell['metadata'] == {'tags': ['keep']}, 'nothing but the cell source and its outputs changes'
    assert nb['metadata'] == NOTEBOOK['metadata']
    # Written back the way Jupyter writes: one-space indent, trailing newline.
    assert path.read_text().startswith('{\n "cells"') and path.read_text().endswith('}\n')
    assert '+r = 3' in out.display['diff']


async def test_cells_are_inserted_after_the_one_named_or_at_the_end(tmp_path: Path):
    path = write(tmp_path)
    c = ctx(tmp_path)
    tool = NotebookEditTool()
    await ReadTool().run({'path': 'area.ipynb'}, c)
    await tool.run({'path': 'area.ipynb', 'action': 'insert', 'cell': 'intro', 'source': 'import math'}, c)
    await tool.run({'path': 'area.ipynb', 'action': 'insert', 'source': '## Notes', 'cell_type': 'markdown'}, c)

    cells = json.loads(path.read_text())['cells']
    assert [c_['cell_type'] for c_ in cells] == ['markdown', 'code', 'code', 'markdown']
    assert cells[1]['source'] == ['import math'] and cells[1]['outputs'] == []
    # nbformat 4.5 requires ids, so a new cell gets one.
    assert len(cells[1]['id']) == 8 and cells[1]['id'] not in ('intro', 'calc')


async def test_a_cell_can_be_deleted_by_its_number(tmp_path: Path):
    path = write(tmp_path)
    c = ctx(tmp_path)
    await ReadTool().run({'path': 'area.ipynb'}, c)
    await NotebookEditTool().run({'path': 'area.ipynb', 'action': 'delete', 'cell': '0'}, c)
    assert [cell['id'] for cell in json.loads(path.read_text())['cells']] == ['calc']


async def test_a_notebook_must_be_read_before_it_is_changed(tmp_path: Path):
    write(tmp_path)
    with pytest.raises(ToolError, match='read it before editing'):
        await NotebookEditTool().run({'path': 'area.ipynb', 'cell': '0', 'source': 'x'}, ctx(tmp_path, 'fresh'))


async def test_an_unknown_cell_is_refused_with_the_ones_that_exist(tmp_path: Path):
    write(tmp_path)
    c = ctx(tmp_path)
    await ReadTool().run({'path': 'area.ipynb'}, c)
    with pytest.raises(ToolError, match='intro, calc'):
        await NotebookEditTool().run({'path': 'area.ipynb', 'cell': 'nope', 'source': 'x'}, c)
    with pytest.raises(ToolError, match='cells 0 to 1'):
        await NotebookEditTool().run({'path': 'area.ipynb', 'cell': '7', 'source': 'x'}, c)


def test_only_notebooks_are_edited_this_way(tmp_path: Path):
    tool = NotebookEditTool()
    assert tool.assess({'path': 'x.py', 'cell': '0', 'source': 'y'}, ctx(tmp_path)).invalid
    assert tool.assess({'path': 'x.ipynb', 'source': 'y'}, ctx(tmp_path)).invalid, 'replace needs a cell'


async def test_the_plot_reaches_the_model(tmp_path: Path):
    write(tmp_path)
    provider = ScriptedProvider([
        [StreamToolUse(id='c1', name='read_file', input={'path': 'area.ipynb'}), StreamDone(stop_reason='tool_use')],
        [StreamText(text='It is a dot.'), StreamDone()],
    ])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.ASK)
    await session.start()
    await turn(session, 'what does the plot show?')
    images = [b for m in provider.seen[-1].messages for b in m.content if isinstance(b, ImageBlock)]
    assert images and images[0].data == PIXEL
