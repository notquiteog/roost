"""A language server small enough to read, for testing the client against.

Knows files by regular expression: `def name` and `class name` declare, a word
is a reference wherever it appears, and any line containing BROKEN is an
error. It speaks the real protocol over stdio — Content-Length framing,
requests of its own to the client, and indexing announced with $/progress —
because those are the parts of a real server a mock of the client would skip.
"""

from __future__ import annotations

import json
import re
import sys
import threading

DOCS: dict[str, str] = {}
LOCK = threading.Lock()


def read() -> dict | None:
    length = 0
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        text = line.decode().strip()
        if not text:
            break
        key, _, value = text.partition(':')
        if key.lower() == 'content-length':
            length = int(value)
    return json.loads(sys.stdin.buffer.read(length))


def send(message: dict) -> None:
    body = json.dumps(message).encode()
    with LOCK:
        sys.stdout.buffer.write(f'Content-Length: {len(body)}\r\n\r\n'.encode() + body)
        sys.stdout.buffer.flush()


def span(uri: str, line: int, start: int, end: int) -> dict:
    return {'uri': uri, 'range': {'start': {'line': line, 'character': start}, 'end': {'line': line, 'character': end}}}


def word_at(uri: str, position: dict) -> str:
    lines = DOCS[uri].splitlines()
    text = lines[position['line']] if position['line'] < len(lines) else ''
    for match in re.finditer(r'\w+', text):
        if match.start() <= position['character'] <= match.end():
            return match.group(0)
    return ''


def declarations() -> list[tuple[str, int, int, str, int]]:
    """(uri, line, column, name, kind) for every def and class."""
    out = []
    for uri, text in DOCS.items():
        for number, line in enumerate(text.splitlines()):
            match = re.match(r'\s*(def|class)\s+(\w+)', line)
            if match:
                out.append((uri, number, match.start(2), match.group(2), 5 if match.group(1) == 'class' else 12))
    return out


def publish(uri: str) -> None:
    problems = []
    for number, line in enumerate(DOCS[uri].splitlines()):
        if 'BROKEN' in line:
            at = line.index('BROKEN')
            problems.append({**span(uri, number, at, at + 6), 'severity': 1, 'message': 'this line is broken', 'source': 'fakels'})
            del problems[-1]['uri']
    send({'jsonrpc': '2.0', 'method': 'textDocument/publishDiagnostics', 'params': {'uri': uri, 'diagnostics': problems}})


def handle(message: dict) -> None:
    method = message.get('method')
    params = message.get('params') or {}
    if method is None:
        return  # the client answering one of our requests

    result = None
    if method == 'initialize':
        result = {
            'capabilities': {
                'textDocumentSync': 1, 'definitionProvider': True, 'referencesProvider': True,
                'hoverProvider': True, 'documentSymbolProvider': True, 'workspaceSymbolProvider': True,
            },
            'serverInfo': {'name': 'fakels'},
        }
    elif method == 'initialized':
        # What real servers do next: ask for configuration, and index.
        send({'jsonrpc': '2.0', 'id': 900, 'method': 'workspace/configuration', 'params': {'items': [{'section': 'fakels'}]}})
        send({'jsonrpc': '2.0', 'id': 901, 'method': 'window/workDoneProgress/create', 'params': {'token': 'index'}})
        send({'jsonrpc': '2.0', 'method': '$/progress', 'params': {'token': 'index', 'value': {'kind': 'begin', 'title': 'Indexing'}}})
        threading.Timer(0.3, lambda: send({
            'jsonrpc': '2.0', 'method': '$/progress', 'params': {'token': 'index', 'value': {'kind': 'end'}},
        })).start()
        return
    elif method == 'textDocument/didOpen':
        DOCS[params['textDocument']['uri']] = params['textDocument']['text']
        publish(params['textDocument']['uri'])
        return
    elif method == 'textDocument/didChange':
        DOCS[params['textDocument']['uri']] = params['contentChanges'][-1]['text']
        publish(params['textDocument']['uri'])
        return
    elif method == 'textDocument/didSave':
        return
    elif method == 'exit':
        sys.exit(0)
    elif method == 'textDocument/definition':
        word = word_at(params['textDocument']['uri'], params['position'])
        result = [span(u, n, c, c + len(name)) for u, n, c, name, _ in declarations() if name == word]
    elif method == 'textDocument/references':
        word = word_at(params['textDocument']['uri'], params['position'])
        result = [
            span(uri, number, m.start(), m.end())
            for uri, text in DOCS.items()
            for number, line in enumerate(text.splitlines())
            for m in re.finditer(rf'\b{re.escape(word)}\b', line)
        ] if word else []
    elif method == 'textDocument/hover':
        word = word_at(params['textDocument']['uri'], params['position'])
        result = {'contents': {'kind': 'markdown', 'value': f'```python\ndef {word}()\n```\nA thing called {word}.'}} if word else None
    elif method == 'textDocument/documentSymbol':
        uri = params['textDocument']['uri']
        result = [
            {'name': name, 'kind': kind, 'range': span(uri, n, c, c)['range'], 'selectionRange': span(uri, n, c, c + len(name))['range']}
            for u, n, c, name, kind in declarations() if u == uri
        ]
    elif method == 'workspace/symbol':
        query = params.get('query', '')
        result = [
            {'name': name, 'kind': kind, 'location': span(u, n, c, c + len(name))}
            for u, n, c, name, kind in declarations() if query in name
        ]

    if 'id' in message:
        send({'jsonrpc': '2.0', 'id': message['id'], 'result': result})


def main() -> None:
    while True:
        message = read()
        if message is None:
            return
        handle(message)


if __name__ == '__main__':
    main()
