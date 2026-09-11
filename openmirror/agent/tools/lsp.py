"""Asking the language server.

Pointing at a symbol is the hard part for a model, and the protocol makes it
harder: a position is a zero-based line and a character offset counted in
UTF-16 code units. Nobody should have to count columns, and a small model
cannot. So the tool takes the line as `read_file` numbers it and the *name*
on that line, and works out the rest; `column` is there for the rare line
with the same name on it twice.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from openmirror.agent.lsp import LspError, LspPool, uri_to_path
from openmirror.agent.tools.base import Assessment, Output, Tool, ToolContext, ToolError, resolve_in_root, truncate
from openmirror.protocol.agent import Risk

OPERATIONS = (
    'definition', 'references', 'hover', 'implementation', 'type_definition',
    'symbols', 'search', 'diagnostics',
)
POSITIONAL = ('definition', 'references', 'hover', 'implementation', 'type_definition')
METHODS = {
    'definition': 'textDocument/definition',
    'references': 'textDocument/references',
    'hover': 'textDocument/hover',
    'implementation': 'textDocument/implementation',
    'type_definition': 'textDocument/typeDefinition',
    'symbols': 'textDocument/documentSymbol',
    'search': 'workspace/symbol',
}
# What a server has to have said it does, for each question. Not every server
# does everything — pylsp has no project-wide search — and the question
# unasked, with something to do instead, beats the server's own error.
CAPABILITIES = {
    'definition': 'definitionProvider',
    'references': 'referencesProvider',
    'hover': 'hoverProvider',
    'implementation': 'implementationProvider',
    'type_definition': 'typeDefinitionProvider',
    'symbols': 'documentSymbolProvider',
    'search': 'workspaceSymbolProvider',
}
INSTEAD = {
    'search': 'Use grep for the name, or ask for its definition from somewhere it is used.',
    'implementation': 'Ask for its definition and references instead.',
    'type_definition': 'Ask for hover instead, which usually says the type.',
    'symbols': 'Use outline instead.',
}
KINDS = {
    1: 'file', 2: 'module', 3: 'namespace', 4: 'package', 5: 'class', 6: 'method', 7: 'property',
    8: 'field', 9: 'constructor', 10: 'enum', 11: 'interface', 12: 'function', 13: 'variable',
    14: 'constant', 15: 'string', 16: 'number', 17: 'boolean', 18: 'array', 19: 'object', 20: 'key',
    21: 'null', 22: 'enum member', 23: 'struct', 24: 'event', 25: 'operator', 26: 'type parameter',
}
SEVERITY = {1: 'error', 2: 'warning', 3: 'info', 4: 'hint'}
MAX_RESULTS = 200

DESCRIPTION = (
    'Ask the language server about code: where something is defined, everywhere it is used, its '
    'type and documentation, what a file declares, a symbol anywhere in the project, or the errors '
    'the compiler sees in a file. Better than grep for anything with a name — it knows which '
    '`open` is which. Point at a symbol with `path`, `line` (as read_file numbers it) and `symbol`, '
    'the name on that line.'
)


def _utf16(text: str) -> int:
    return len(text.encode('utf-16-le')) // 2


def _position(path: Path, args: dict[str, Any]) -> dict[str, int]:
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    number = int(args['line'])
    if not 1 <= number <= len(lines):
        raise ToolError(f'line {number} is outside {path.name}, which has {len(lines)} lines')
    text = lines[number - 1]

    symbol = str(args.get('symbol') or '').strip()
    if symbol:
        match = re.search(rf'(?<![\w$]){re.escape(symbol)}(?![\w$])', text)
        at = match.start() if match else text.find(symbol)
        if at < 0:
            raise ToolError(f'{symbol!r} is not on line {number}. That line is: {text.strip()}')
    elif args.get('column'):
        at = max(0, int(args['column']) - 1)
    else:
        at = len(text) - len(text.lstrip())
    return {'line': number - 1, 'character': _utf16(text[:at])}


def _locations(result: Any) -> list[tuple[str, dict[str, Any]]]:
    if not result:
        return []
    items = result if isinstance(result, list) else [result]
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if 'targetUri' in item:
            out.append((item['targetUri'], item.get('targetSelectionRange') or item.get('targetRange') or {}))
        elif 'uri' in item:
            out.append((item['uri'], item.get('range') or {}))
    return out


class _Lines:
    """Source lines by file, read once each, for quoting a result's line."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[Path, list[str]] = {}

    def name(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def at(self, path: Path, line: int) -> str:
        if path not in self._cache:
            try:
                self._cache[path] = path.read_text(encoding='utf-8', errors='replace').splitlines()
            except OSError:
                self._cache[path] = []
        lines = self._cache[path]
        return lines[line].strip() if 0 <= line < len(lines) else ''

    def where(self, uri: str, rng: dict[str, Any]) -> str:
        path = uri_to_path(uri)
        start = rng.get('start') or {}
        line = int(start.get('line', 0))
        quoted = self.at(path, line)
        quoted = quoted if len(quoted) <= 160 else quoted[:157] + '...'
        return f'{self.name(path)}:{line + 1}:{int(start.get("character", 0)) + 1}  {quoted}'


def _hover_text(result: Any) -> str:
    contents = result.get('contents') if isinstance(result, dict) else None
    if isinstance(contents, dict):
        return str(contents.get('value', ''))
    if isinstance(contents, list):
        return '\n\n'.join(c.get('value', '') if isinstance(c, dict) else str(c) for c in contents)
    return str(contents or '')


def _symbol_lines(items: list[dict[str, Any]], depth: int = 0) -> list[str]:
    lines: list[str] = []
    for item in items:
        kind = KINDS.get(item.get('kind', 0), 'symbol')
        rng = item.get('selectionRange') or item.get('range') or (item.get('location') or {}).get('range') or {}
        line = int((rng.get('start') or {}).get('line', 0)) + 1
        detail = f'  {item["detail"]}' if item.get('detail') else ''
        lines.append(f'{line:>5}  {"  " * depth}{kind:<11} {item.get("name", "?")}{detail}')
        if item.get('children'):
            lines.extend(_symbol_lines(item['children'], depth + 1))
    return lines


class LspTool(Tool):
    name = 'lsp'

    def __init__(self, pool: LspPool) -> None:
        self.pool = pool
        self.description = f'{DESCRIPTION} Available here for: {", ".join(pool.describe())}.'
        self.input_schema = {
            'type': 'object',
            'properties': {
                'operation': {'type': 'string', 'enum': list(OPERATIONS)},
                'path': {'type': 'string', 'description': 'The file. For search, any file in the language to search.'},
                'line': {'type': 'integer', 'description': 'The line, numbered from 1 as read_file shows it.'},
                'symbol': {'type': 'string', 'description': 'The name on that line to ask about.'},
                'column': {'type': 'integer', 'description': 'Instead of symbol: the column, from 1.'},
                'query': {'type': 'string', 'description': 'For search: the name, or part of it.'},
                'wait': {'type': 'integer', 'description': 'For diagnostics: seconds to wait for the server. Default 10.'},
            },
            'required': ['operation'],
        }

    async def close(self) -> None:
        await self.pool.close()

    def _spec(self, args: dict[str, Any]) -> tuple[Any, str]:
        """The server for this call, or the reason there is none."""
        path = str(args.get('path') or '')
        if not path:
            if args.get('operation') == 'search' and len(self.pool.specs) == 1:
                return self.pool.specs[0], ''
            return None, 'path is required' + (' — any file in the language to search' if args.get('operation') == 'search' else '')
        spec = self.pool.spec_for(path)
        if spec is None:
            return None, (
                f'there is no language server here for {Path(path).suffix or "that kind of"} files. '
                f'There is one for: {", ".join(self.pool.describe())}'
            )
        return spec, ''

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        op = args.get('operation')
        if op not in OPERATIONS:
            return Assessment(risk=Risk.READ, summary='', invalid=f'operation must be one of {", ".join(OPERATIONS)}')
        spec, why = self._spec(args)
        if spec is None:
            return Assessment(risk=Risk.READ, summary='', invalid=why)
        if op in POSITIONAL and not args.get('line'):
            return Assessment(
                risk=Risk.READ, summary='', invalid=f'{op} needs a line — the line number as read_file shows it'
            )
        if op == 'search' and not str(args.get('query') or '').strip():
            return Assessment(risk=Risk.READ, summary='', invalid='search needs a query')

        path = args.get('path') or ''
        if op == 'search':
            what = f'search for {args["query"]!r}'
        elif op in POSITIONAL:
            target = args.get('symbol') or f'line {args["line"]}'
            what = f'{op.replace("_", " ")} of {target} in {path}:{args["line"]}'
        else:
            what = f'{op} in {path}'

        if not self.pool.running(spec):
            return Assessment(
                risk=Risk.EXECUTE,
                summary=f'start {spec.name} for this project, then {what}   '
                        '(a language server can run the project\'s own build scripts)',
            )
        return Assessment(risk=Risk.READ, summary=what)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        op = args['operation']
        spec, why = self._spec(args)
        if spec is None:
            raise ToolError(why)
        path = resolve_in_root(args['path'], ctx) if args.get('path') else None
        if path is not None and not path.is_file():
            raise ToolError(f'{args["path"]}: no such file')

        try:
            server = await self.pool.server(spec)
            await server.settle(20)
            lines = _Lines(ctx.root)

            if op in CAPABILITIES and not server.supports(METHODS[op], CAPABILITIES[op]):
                raise ToolError(
                    f'{spec.name} does not answer {op.replace("_", " ")} questions. {INSTEAD.get(op, "Use grep instead.")}'
                )

            if op == 'diagnostics':
                return await self._diagnostics(server, path, args, lines)

            if op == 'search':
                result = await server.request(METHODS['search'], {'query': str(args['query']).strip()})
                return self._search(server, result, args, lines)

            await server.sync(path)
            uri = path.as_uri()
            if op == 'symbols':
                result = await server.request(METHODS['symbols'], {'textDocument': {'uri': uri}})
                return self._symbols(server, result or [], path, lines)

            params: dict[str, Any] = {'textDocument': {'uri': uri}, 'position': _position(path, args)}
            if op == 'references':
                params['context'] = {'includeDeclaration': True}
            result = await server.request(METHODS[op], params)
        except LspError as exc:
            raise ToolError(str(exc)) from exc

        if op == 'hover':
            text = _hover_text(result).strip()
            if not text:
                return Output(content=self._nothing(server, 'nothing to say about that position'))
            body, cut = truncate(text, 6_000, keep='head')
            return Output(content=body, truncated=cut, display={'server': spec.name})

        found = _locations(result)
        if not found:
            return Output(content=self._nothing(server, f'no {op.replace("_", " ")} found'))
        rows = [lines.where(uri_, rng) for uri_, rng in found[:MAX_RESULTS]]
        head = f'{len(found)} result{"s" if len(found) != 1 else ""}'
        if len(found) > MAX_RESULTS:
            head += f', the first {MAX_RESULTS} shown'
        return Output(content=f'{head}:\n' + '\n'.join(rows), display={'server': spec.name, 'results': len(found)})

    def _nothing(self, server: Any, said: str) -> str:
        # An empty answer from a busy server is not an answer, and saying so
        # is the difference between the model trying again and the model
        # concluding the symbol does not exist.
        if server.busy:
            return f'{said.capitalize()} — but {server.spec.name} is still busy ({server.busy}), so ask again shortly.'
        return f'{said.capitalize()}.'

    def _symbols(self, server: Any, result: list[dict[str, Any]], path: Path, lines: _Lines) -> Output:
        rows = _symbol_lines(result)
        if not rows:
            return Output(content=self._nothing(server, f'no symbols reported in {lines.name(path)}'))
        body, cut = truncate('\n'.join(rows[:600]), 40_000, keep='head')
        return Output(
            content=f'{lines.name(path)}: {len(rows)} symbols\n{body}',
            truncated=cut or len(rows) > 600,
            display={'server': server.spec.name, 'symbols': len(rows)},
        )

    def _search(self, server: Any, result: Any, args: dict[str, Any], lines: _Lines) -> Output:
        items = result or []
        if not items:
            return Output(content=self._nothing(server, f'nothing called {args["query"]!r} found'))
        rows = []
        for item in items[:MAX_RESULTS]:
            loc = item.get('location') or {}
            kind = KINDS.get(item.get('kind', 0), 'symbol')
            where = lines.where(loc['uri'], loc.get('range') or {}) if loc.get('uri') else ''
            rows.append(f'{item.get("name", "?")}  ({kind})  {where}'.rstrip())
        return Output(
            content=f'{len(items)} match{"es" if len(items) != 1 else ""}:\n' + '\n'.join(rows),
            display={'server': server.spec.name, 'results': len(items)},
        )

    async def _diagnostics(self, server: Any, path: Path | None, args: dict[str, Any], lines: _Lines) -> Output:
        if path is None:
            raise ToolError('diagnostics needs a path')
        wait = max(1, min(int(args.get('wait') or 10), 120))
        items = await server.diagnostics_for(path, wait)
        name = lines.name(path)
        if items is None:
            busy = f' — it is still {server.busy}' if server.busy else ''
            return Output(
                content=f'{server.spec.name} has not reported on {name} yet{busy}. Ask again in a moment, '
                        'or give wait more seconds.'
            )
        if not items:
            return Output(content=f'No problems reported in {name}.', display={'server': server.spec.name, 'problems': 0})

        items = sorted(items, key=lambda d: (d.get('severity', 1), d['range']['start']['line']))
        rows = []
        for d in items[:MAX_RESULTS]:
            start = d['range']['start']
            where = f'{name}:{start["line"] + 1}:{start.get("character", 0) + 1}'
            code = d.get('code')
            tag = ' '.join(str(x) for x in (d.get('source'), code) if x)
            message = ' '.join(str(d.get('message', '')).split())
            rows.append(f'{where}  {SEVERITY.get(d.get("severity", 1), "error")}  {message}' + (f'  [{tag}]' if tag else ''))
        errors = sum(1 for d in items if d.get('severity', 1) == 1)
        return Output(
            content=f'{len(items)} problem{"s" if len(items) != 1 else ""} ({errors} error{"s" if errors != 1 else ""}):\n'
                    + '\n'.join(rows),
            display={'server': server.spec.name, 'problems': len(items), 'errors': errors},
        )
