"""Jupyter notebooks, as cells rather than as JSON.

A notebook is a JSON document that happens to contain code, and both halves of
that are a trap for a model working through the ordinary file tools. Read as
text, the code is a list of quoted strings with an escaped newline at the end
of each, and the outputs bury it — one plot is a hundred kilobytes of base64.
Edited as text, one misplaced comma leaves a file Jupyter refuses to open, and
the edit tool's exact-match rule has to match JSON escaping the model never
sees in the source it is thinking about.

So `read_file` renders a notebook as its cells — numbered, typed, with their
outputs as text and their pictures as pictures the model can actually look
at — and `notebook_edit` changes one cell at a time, through the JSON rather
than around it.

No nbformat. The format is documented, stable and plain JSON; what nbformat
would add is validation, and the property that matters — that what is written
back is still a notebook — is kept here by never writing anything but cells.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from openmirror.agent.tools.base import (
    Assessment,
    Output,
    Tool,
    ToolContext,
    ToolError,
    resolve_in_root,
    truncate,
)
from openmirror.agent.tools.files import _diff, journal
from openmirror.protocol.agent import Risk

# Notebooks carry their pictures inside them, so the ordinary read limit is too
# small for an ordinary one. The pictures are pulled out rather than read as
# text, so what reaches the model is a fraction of this.
MAX_NOTEBOOK_BYTES = 50_000_000
OUTPUT_CHARS = 3_000
MAX_IMAGES = 6
IMAGE_TYPES = ('image/png', 'image/jpeg', 'image/gif', 'image/webp')
CELL_TYPES = ('code', 'markdown', 'raw')
ACTIONS = ('replace', 'insert', 'delete')
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


def load(path: Path) -> tuple[dict[str, Any], str]:
    """The notebook, and the text it was read from (for writing it back alike)."""
    if path.stat().st_size > MAX_NOTEBOOK_BYTES:
        raise ToolError(f'{path.name}: {path.stat().st_size} bytes is too large a notebook to open here')
    raw = path.read_text(encoding='utf-8', errors='replace')
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolError(f'{path.name}: not valid JSON, so not a notebook ({exc})') from exc
    if not isinstance(data, dict) or not isinstance(data.get('cells'), list):
        if isinstance(data, dict) and 'worksheets' in data:
            raise ToolError(
                f'{path.name}: an nbformat 3 notebook. Open and save it in a current Jupyter to '
                'upgrade it first.'
            )
        raise ToolError(f'{path.name}: has no cells, so it is not a Jupyter notebook')
    return data, raw


def source_of(cell: dict[str, Any]) -> str:
    src = cell.get('source', '')
    return ''.join(src) if isinstance(src, list) else str(src)


def _joined(value: Any) -> str:
    return ''.join(value) if isinstance(value, list) else str(value or '')


def language_of(nb: dict[str, Any]) -> str:
    meta = nb.get('metadata') or {}
    return (
        (meta.get('kernelspec') or {}).get('language')
        or (meta.get('language_info') or {}).get('name')
        or 'an unknown language'
    )


def _output(out: dict[str, Any]) -> tuple[str, list[tuple[str, str]]]:
    """One cell output as (text, pictures)."""
    kind = out.get('output_type')
    if kind == 'stream':
        return _joined(out.get('text')), []
    if kind == 'error':
        trace = ANSI.sub('', '\n'.join(out.get('traceback') or []))
        head = f'{out.get("ename", "Error")}: {out.get("evalue", "")}'
        return (trace if head.split(':')[0] in trace else f'{head}\n{trace}').strip(), []
    if kind in ('execute_result', 'display_data'):
        data = out.get('data') or {}
        pictures = [
            (re.sub(r'\s+', '', _joined(data[kind_])), kind_) for kind_ in IMAGE_TYPES if kind_ in data
        ]
        text = _joined(data.get('text/plain')) or _joined(data.get('text/markdown'))
        if not text and 'text/html' in data:
            text = '[HTML output, not shown]'
        if not text and 'image/svg+xml' in data:
            text = '[an SVG image, not shown]'
        return text, pictures
    return '', []


def render(path: Path, nb: dict[str, Any]) -> tuple[str, list[tuple[str, str]], int]:
    """The notebook as a model should read it, its pictures, and how many were left out."""
    cells = nb['cells']
    images: list[tuple[str, str]] = []
    left_out = 0
    parts = [
        f'{path.name}: {len(cells)} cell{"s" if len(cells) != 1 else ""}, {language_of(nb)}. '
        'Cells are numbered from 0; notebook_edit takes either the number or the id.'
    ]

    for index, cell in enumerate(cells):
        kind = cell.get('cell_type', 'code')
        head = f'\n── cell {index}'
        if cell.get('id'):
            head += f' · id {cell["id"]}'
        head += f' · {kind}'
        if kind == 'code' and cell.get('execution_count') is not None:
            head += f' · ran [{cell["execution_count"]}]'
        parts.append(head + ' ──')
        parts.append(source_of(cell).rstrip('\n') or '(empty)')

        for out in cell.get('outputs') or []:
            text, pictures = _output(out)
            if text.strip():
                body, _ = truncate(text.rstrip('\n'), OUTPUT_CHARS, keep='both')
                parts.append(f'── output ──\n{body}')
            for data, media_type in pictures:
                if len(images) < MAX_IMAGES:
                    images.append((data, media_type))
                    parts.append(f'── output: image {len(images)}, attached ──')
                else:
                    left_out += 1
                    parts.append('── output: an image, not attached ──')

    return '\n'.join(parts), images, left_out


def read(path: Path, ctx: ToolContext) -> Output:
    """What `read_file` returns for a notebook."""
    nb, _ = load(path)
    journal.note_read(ctx.session_id, path)
    text, images, left_out = render(path, nb)
    body, cut = truncate(text, 80_000, keep='head')
    if left_out:
        body += f'\n\n[{left_out} more image{"s" if left_out != 1 else ""} in the outputs were not attached]'
    return Output(
        content=body,
        truncated=cut,
        images=images,
        display={'path': str(path), 'cells': len(nb['cells']), 'notebook': True},
    )


def _find(cells: list[dict[str, Any]], ref: Any) -> int:
    """A cell by id, or by number. Ids first: an id can be all digits."""
    wanted = str(ref).strip()
    for i, cell in enumerate(cells):
        if cell.get('id') == wanted:
            return i
    if re.fullmatch(r'\d+', wanted):
        number = int(wanted)
        if number < len(cells):
            return number
        raise ToolError(
            f'there is no cell {number}: this notebook has cells 0 to {len(cells) - 1}'
            if cells else 'this notebook has no cells yet — insert one'
        )
    ids = [c.get('id') for c in cells if c.get('id')]
    listed = ', '.join(ids[:12]) + (' …' if len(ids) > 12 else '')
    raise ToolError(
        f'no cell with id {wanted!r}. ' + (f'The ids here are: {listed}.' if ids else 'Cells here have no ids; use the number.')
    )


def _needs_ids(nb: dict[str, Any]) -> bool:
    """nbformat 4.5 made cell ids required; a notebook older than that has none."""
    major = nb.get('nbformat', 4)
    return major > 4 or (major == 4 and nb.get('nbformat_minor', 0) >= 5)


def _dump(nb: dict[str, Any], original: str) -> str:
    """Written back the way it was written: same indent, same key order.

    Jupyter writes one-space indents and most tools copy it; matching whatever
    this file used means the diff in a pull request is the cell that changed,
    not every line in the file.
    """
    match = re.match(r'\{\s*\n( +)"', original)
    indent = len(match.group(1)) if match else 1
    text = json.dumps(nb, indent=indent, ensure_ascii=False)
    return text + '\n' if original.endswith('\n') or not original else text


class NotebookEditTool(Tool):
    name = 'notebook_edit'
    description = (
        'Change one cell of a Jupyter notebook: replace its source, insert a new cell, or delete one. '
        'Read the notebook with read_file first — it shows every cell with its number and id, and '
        'either will do here. `source` is the complete new source of the cell, not a fragment. '
        'Replacing a code cell clears its outputs, because they no longer match the code. `insert` '
        'puts the new cell after the one named, or at the end if none is.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'path': {'type': 'string', 'description': 'The .ipynb file.'},
            'action': {'type': 'string', 'enum': list(ACTIONS), 'description': 'Default replace.'},
            'cell': {'type': 'string', 'description': 'The cell: its id, or its number from 0.'},
            'source': {'type': 'string', 'description': 'The whole new source, for replace and insert.'},
            'cell_type': {
                'type': 'string', 'enum': list(CELL_TYPES),
                'description': 'For insert (default code), or to change a cell\'s type on replace.',
            },
        },
        'required': ['path'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        path = args.get('path') or ''
        action = args.get('action') or 'replace'
        if not path:
            return Assessment(risk=Risk.WRITE, summary='', invalid='path is required')
        if not path.lower().endswith('.ipynb'):
            return Assessment(
                risk=Risk.WRITE, summary='', invalid=f'{path} is not a notebook — use edit_file for other files'
            )
        if action not in ACTIONS:
            return Assessment(risk=Risk.WRITE, summary='', invalid=f'action must be one of {", ".join(ACTIONS)}')
        if action in ('replace', 'insert') and args.get('source') is None:
            return Assessment(risk=Risk.WRITE, summary='', invalid=f'{action} needs the source')
        if action in ('replace', 'delete') and args.get('cell') in (None, ''):
            return Assessment(risk=Risk.WRITE, summary='', invalid=f'{action} needs the cell')
        if args.get('cell_type') and args['cell_type'] not in CELL_TYPES:
            return Assessment(risk=Risk.WRITE, summary='', invalid='cell_type must be code, markdown or raw')

        cell = args.get('cell')
        if action == 'replace':
            summary = f'replace cell {cell} in {path}'
        elif action == 'delete':
            summary = f'delete cell {cell} from {path}'
        else:
            kind = args.get('cell_type') or 'code'
            summary = f'insert a {kind} cell {"after cell " + str(cell) if cell not in (None, "") else "at the end"} of {path}'
        # A write, deletion included: it is an edit to a file that a checkpoint
        # can put back, the same as edit_file removing lines.
        return Assessment(risk=Risk.WRITE, summary=summary)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        path = resolve_in_root(args['path'], ctx)
        if not path.is_file():
            raise ToolError(f'{args["path"]}: no such file')
        if not journal.has_read(ctx.session_id, path):
            raise ToolError(f'{args["path"]}: read it before editing it.')
        if journal.changed_since_read(ctx.session_id, path):
            raise ToolError(f'{args["path"]}: changed on disk since you read it. Read it again first.')

        nb, original = load(path)
        cells: list[dict[str, Any]] = nb['cells']
        action = args.get('action') or 'replace'
        label = args['path']

        if action == 'insert':
            after = _find(cells, args['cell']) if args.get('cell') not in (None, '') else len(cells) - 1
            kind = args.get('cell_type') or 'code'
            cell: dict[str, Any] = {'cell_type': kind, 'metadata': {}, 'source': args['source'].splitlines(keepends=True)}
            if kind == 'code':
                cell['execution_count'] = None
                cell['outputs'] = []
            if _needs_ids(nb):
                cell['id'] = uuid.uuid4().hex[:8]
            # Keys in the order Jupyter writes them, which is alphabetical.
            cell = dict(sorted(cell.items()))
            index = after + 1
            cells.insert(index, cell)
            before_src, after_src = '', args['source']
            said = f'Inserted a {kind} cell at {index}' + (f' (id {cell["id"]})' if cell.get('id') else '')

        elif action == 'delete':
            index = _find(cells, args['cell'])
            removed = cells.pop(index)
            before_src, after_src = source_of(removed), ''
            said = f'Deleted cell {index}; the cells after it are renumbered'

        else:
            index = _find(cells, args['cell'])
            cell = cells[index]
            before_src, after_src = source_of(cell), args['source']
            kind = args.get('cell_type') or cell.get('cell_type', 'code')
            if before_src == after_src and kind == cell.get('cell_type'):
                raise ToolError(f'{label}: cell {index} already has exactly that source')
            cell['source'] = after_src.splitlines(keepends=True)
            cell['cell_type'] = kind
            if kind == 'code':
                # Outputs are what the *old* code produced. Leaving them under
                # new code is a notebook that lies about what it computed.
                cell['outputs'] = []
                cell['execution_count'] = None
            else:
                cell.pop('outputs', None)
                cell.pop('execution_count', None)
            said = f'Replaced cell {index}' + (', and cleared its outputs' if kind == 'code' else '')

        if ctx.checkpoint is not None:
            ctx.checkpoint.record(path)
        path.write_text(_dump(nb, original), encoding='utf-8')
        journal.note_read(ctx.session_id, path)

        return Output(
            content=f'{said} in {label}. It now has {len(cells)} cell{"s" if len(cells) != 1 else ""}.',
            display={
                'path': str(path),
                'cell': index,
                'diff': _diff(before_src, after_src, f'{label} (cell {index})'),
            },
        )
