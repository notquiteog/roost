"""Tools for the way models actually work on code.

The file tools cover reading and changing one thing at a time, which is
correct and slow. Everything here exists because of a specific failure watched
happening rather than because it seemed useful:

**`multi_edit`, because a half-applied refactor is worse than none.** Six
edits to one file as six calls means six chances for the fourth to fail on a
non-unique match, and by then three have landed. This applies all of them to
one buffer and writes once, or writes nothing and says which one failed and
why. One diff, one checkpoint, one approval.

**`read_files`, because reading four files is four round trips.** Each of them
costs a model turn, and a model that has to spend four turns orienting itself
spends them instead of thinking. The cost is a bigger result, which is a much
better trade than it sounds — the files were going to be read anyway.

**`outline`, because reading a 2000-line file to find one function is a waste
of a context window.** A structural sketch — what is defined, and where — is
usually the whole of what "have a look at X" needs, and it is a tenth of the
tokens. Regex per language rather than a parser: a real parser means a
dependency per language and a build step, and gets the same answer for the
question actually being asked. Where a language server is installed, `lsp`
answers the harder version of the question.
"""

from __future__ import annotations

import re
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


class MultiEditTool(Tool):
    name = 'multi_edit'
    description = (
        'Make several exact-string replacements in one file, in order, as one change. '
        'Every edit must match, or none is applied — so a refactor cannot land halfway. '
        'Each edit sees the result of the ones before it. The file must have been read '
        'in this session, and old_string must never include the line numbers read_file adds.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'path': {'type': 'string'},
            'edits': {
                'type': 'array',
                'description': 'Applied in order. Later edits see earlier ones.',
                'items': {
                    'type': 'object',
                    'properties': {
                        'old_string': {'type': 'string'},
                        'new_string': {'type': 'string'},
                        'replace_all': {'type': 'boolean'},
                    },
                    'required': ['old_string', 'new_string'],
                },
            },
        },
        'required': ['path', 'edits'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        path = args.get('path')
        edits = args.get('edits')
        if not path:
            return Assessment(risk=Risk.WRITE, summary='', invalid='path is required')
        if not isinstance(edits, list) or not edits:
            return Assessment(risk=Risk.WRITE, summary='', invalid='edits must be a non-empty list')
        for i, edit in enumerate(edits):
            if not isinstance(edit, dict) or edit.get('old_string') is None or edit.get('new_string') is None:
                return Assessment(
                    risk=Risk.WRITE, summary='',
                    invalid=f'edit {i + 1} needs both old_string and new_string',
                )
            if edit['old_string'] == edit['new_string']:
                return Assessment(
                    risk=Risk.WRITE, summary='', invalid=f'edit {i + 1}: the two strings are identical'
                )
        return Assessment(risk=Risk.WRITE, summary=f'{len(edits)} edits to {path}')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        path = resolve_in_root(args['path'], ctx)
        if not path.is_file():
            raise ToolError(f'{args["path"]}: no such file')
        if not journal.has_read(ctx.session_id, path):
            raise ToolError(f'{args["path"]}: read it before editing it.')
        if journal.changed_since_read(ctx.session_id, path):
            raise ToolError(f'{args["path"]}: changed on disk since you read it. Read it again first.')

        before = path.read_text(encoding='utf-8', errors='replace')
        working = before
        applied = 0

        for i, edit in enumerate(args['edits'], start=1):
            old, new = edit['old_string'], edit['new_string']
            count = working.count(old)
            if count == 0:
                # Nothing has been written yet, so this is a clean failure —
                # which is the whole point of doing it in a buffer. The message
                # says how far it got, because "edit 4 of 6 did not match" tells
                # the model which of its assumptions was wrong.
                raise ToolError(
                    f'{args["path"]}: edit {i} of {len(args["edits"])} did not match, so nothing was '
                    'written. It must match the file exactly, including whitespace, and must not '
                    'include line numbers. Note that edits are applied in order, so this one had to '
                    'match the file as the earlier edits left it.'
                )
            if count > 1 and not edit.get('replace_all'):
                raise ToolError(
                    f'{args["path"]}: edit {i} matches {count} places, so nothing was written. '
                    'Add surrounding context to make it unique, or set replace_all on that edit.'
                )
            working = working.replace(old, new) if edit.get('replace_all') else working.replace(old, new, 1)
            applied += count if edit.get('replace_all') else 1

        if working == before:
            raise ToolError(f'{args["path"]}: those edits leave the file unchanged')

        if ctx.checkpoint is not None:
            ctx.checkpoint.record(path)
        path.write_text(working, encoding='utf-8')
        journal.note_read(ctx.session_id, path)

        return Output(
            content=f'Applied {len(args["edits"])} edits to {args["path"]} ({applied} replacements).',
            display={'path': str(path), 'diff': _diff(before, working, args['path']), 'replacements': applied},
        )


class ReadFilesTool(Tool):
    name = 'read_files'
    description = (
        'Read several files at once. Use this whenever you need more than one — reading '
        'them one at a time costs a round trip each, and you were going to read them all '
        'anyway. Files that do not exist are reported rather than failing the call.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'paths': {'type': 'array', 'items': {'type': 'string'}},
            'limit': {
                'type': 'integer',
                'description': 'Characters per file before it is truncated. Default 20000.',
            },
        },
        'required': ['paths'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        paths = args.get('paths')
        if not isinstance(paths, list) or not paths:
            return Assessment(risk=Risk.READ, summary='', invalid='paths must be a non-empty list')
        if len(paths) > 40:
            return Assessment(
                risk=Risk.READ, summary='',
                invalid=f'{len(paths)} files at once is too many — take the ones you need most, '
                        'or use grep to find where to look.',
            )
        return Assessment(risk=Risk.READ, summary=f'read {len(paths)} files')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        limit = int(args.get('limit') or 20_000)
        chunks: list[str] = []
        read: list[str] = []
        missing: list[str] = []

        for raw in args['paths']:
            try:
                path = resolve_in_root(raw, ctx)
            except ToolError as exc:
                missing.append(f'{raw}: {exc}')
                continue
            if not path.is_file():
                missing.append(f'{raw}: no such file')
                continue
            try:
                text = path.read_text(encoding='utf-8', errors='replace')
            except OSError as exc:
                missing.append(f'{raw}: {exc}')
                continue

            journal.note_read(ctx.session_id, path)
            body, cut = truncate(text, limit, keep='head')
            # Numbered the same way read_file numbers, so a line referred to
            # from here means the same thing there.
            numbered = '\n'.join(
                f'{i:>5}│{line}' for i, line in enumerate(body.splitlines(), start=1)
            )
            chunks.append(f'=== {raw} ===\n{numbered}' + ('\n[truncated]' if cut else ''))
            read.append(raw)

        if not chunks and missing:
            raise ToolError('none of those could be read:\n' + '\n'.join(missing))

        content = '\n\n'.join(chunks)
        if missing:
            content += '\n\nNot read:\n' + '\n'.join(f'  {m}' for m in missing)

        body, cut = truncate(content, 120_000, keep='head')
        return Output(content=body, truncated=cut, display={'files': read, 'missing': len(missing)})


# ---------------------------------------------------------------------------
# Outline
# ---------------------------------------------------------------------------

# Deliberately shallow. These find declarations at the start of a line, which
# is where declarations are in every language here — and being fooled by the
# word "class" inside a string costs one wrong line in a sketch, whereas a real
# parser costs a dependency per language.
PATTERNS: dict[str, list[tuple[str, str]]] = {
    'python': [
        (r'^(class\s+\w+.*?):', 'class'),
        (r'^(\s*(?:async\s+)?def\s+\w+\s*\(.*?\)).*?:', 'def'),
    ],
    'javascript': [
        (r'^\s*(export\s+)?(default\s+)?(class\s+\w+.*?)\s*\{', 'class'),
        (r'^\s*(?:export\s+)?(?:async\s+)?(function\s*\*?\s*\w+\s*\([^)]*\))', 'function'),
        (r'^\s*(?:export\s+)?(const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(', 'function'),
        (r'^\s{2,}(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{', 'method'),
    ],
    'rust': [
        (r'^\s*(pub\s+)?(fn\s+\w+.*?)[{;]', 'fn'),
        (r'^\s*(pub\s+)?((?:struct|enum|trait|impl)\s+[\w<>, ]+)', 'type'),
    ],
    'go': [
        (r'^(func\s+(?:\([^)]*\)\s*)?\w+\s*\([^)]*\))', 'func'),
        (r'^(type\s+\w+\s+\w+)', 'type'),
    ],
    'shell': [(r'^(\w+)\s*\(\)\s*\{', 'function')],
}

SUFFIXES = {
    '.py': 'python', '.js': 'javascript', '.mjs': 'javascript', '.jsx': 'javascript',
    '.ts': 'javascript', '.tsx': 'javascript', '.rs': 'rust', '.go': 'go',
    '.sh': 'shell', '.bash': 'shell',
}


def outline_text(text: str, language: str) -> list[tuple[int, str, str]]:
    """(line number, kind, declaration) for everything declared in a file."""
    rules = [(re.compile(pattern), kind) for pattern, kind in PATTERNS.get(language, [])]
    if not rules:
        return []

    found: list[tuple[int, str, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if len(line) > 400:
            continue
        for pattern, kind in rules:
            match = pattern.match(line)
            if not match:
                continue
            # The last non-empty group is the declaration itself; the earlier
            # ones are modifiers like `pub` and `export` that the regex has to
            # match but nobody wants in a sketch.
            groups = [g for g in match.groups() if g and g.strip()]
            text_of = (groups[-1] if groups else match.group(0)).strip()
            found.append((number, kind, text_of))
            break
    return found


class OutlineTool(Tool):
    name = 'outline'
    description = (
        'Sketch what a file defines — classes, functions, methods — with line numbers, '
        'without reading the whole thing. Use it to find where something lives before '
        'reading around it. Python, JavaScript, TypeScript, Rust, Go and shell.'
    )
    input_schema = {
        'type': 'object',
        'properties': {'path': {'type': 'string'}},
        'required': ['path'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        if not args.get('path'):
            return Assessment(risk=Risk.READ, summary='', invalid='path is required')
        return Assessment(risk=Risk.READ, summary=f'outline {args["path"]}')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        path = resolve_in_root(args['path'], ctx)
        if not path.is_file():
            raise ToolError(f'{args["path"]}: no such file')

        language = SUFFIXES.get(path.suffix.lower(), '')
        if not language:
            raise ToolError(
                f'{args["path"]}: no outline for {path.suffix or "this kind of file"} — read it instead.'
            )

        text = path.read_text(encoding='utf-8', errors='replace')
        found = outline_text(text, language)
        total = len(text.splitlines())

        if not found:
            # An outline is a *sketch*, and an empty sketch of a file with
            # content in it means this file is not organised the way the
            # patterns expect — worth saying, so the model reads it rather
            # than concluding the file is empty.
            return Output(
                content=f'{args["path"]}: {total} lines, nothing that looks like a declaration. '
                        'Read it directly.',
                display={'path': str(path), 'symbols': 0},
            )

        lines = [f'{args["path"]}  ({total} lines, {len(found)} declarations)']
        lines += [f'{number:>5}  {kind:<8} {what}' for number, kind, what in found]
        body, cut = truncate('\n'.join(lines), 20_000, keep='head')
        return Output(content=body, truncated=cut, display={'path': str(path), 'symbols': len(found)})


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------


class ApplyPatchTool(Tool):
    name = 'apply_patch'
    description = (
        'Apply a unified diff to files under the working root. Use it when you already have '
        'a patch — from git, from a review, from a tool. For edits you are composing '
        'yourself, edit_file and multi_edit are better: they fail loudly on a mismatch, '
        'where a patch can apply at an offset and land in the wrong place.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'patch': {'type': 'string', 'description': 'A unified diff, with a/ and b/ prefixes.'},
        },
        'required': ['patch'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        patch = args.get('patch') or ''
        if '@@' not in patch:
            return Assessment(
                risk=Risk.WRITE, summary='', invalid='that does not look like a unified diff (no @@ hunks)'
            )
        files = re.findall(r'(?m)^\+\+\+ b/(.+)$', patch)
        if not files:
            return Assessment(
                risk=Risk.WRITE, summary='',
                invalid='no +++ b/<path> headers — the patch does not say which files it changes',
            )
        return Assessment(risk=Risk.WRITE, summary=f'patch {len(files)} file(s): {", ".join(files[:3])}')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        hunks = _split_patch(args['patch'])
        if not hunks:
            raise ToolError('no file sections found in that patch')

        # Every file is applied to a buffer first. A patch that fails on the
        # third of four files must not leave the first two changed — the same
        # rule as multi_edit, for the same reason.
        staged: list[tuple[Any, str, str, str]] = []
        for name, body in hunks.items():
            path = resolve_in_root(name, ctx)
            before = path.read_text(encoding='utf-8', errors='replace') if path.is_file() else ''
            try:
                after = _apply_hunks(before, body)
            except ValueError as exc:
                raise ToolError(f'{name}: {exc} — nothing was written.') from exc
            staged.append((path, name, before, after))

        changed: list[str] = []
        diffs: list[str] = []
        for path, name, before, after in staged:
            if before == after:
                continue
            if ctx.checkpoint is not None:
                ctx.checkpoint.record(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(after, encoding='utf-8')
            journal.note_read(ctx.session_id, path)
            changed.append(name)
            diffs.append(_diff(before, after, name))

        if not changed:
            raise ToolError('that patch changes nothing — it may already be applied')

        return Output(
            content=f'Applied to {len(changed)} file(s): {", ".join(changed)}',
            display={'diff': ''.join(diffs), 'files': changed},
        )


def _split_patch(patch: str) -> dict[str, str]:
    """A unified diff, split into one body per file."""
    out: dict[str, list[str]] = {}
    current: str | None = None
    for line in patch.splitlines(keepends=True):
        header = re.match(r'^\+\+\+ (?:b/)?(.+?)\s*$', line)
        if header:
            current = header.group(1)
            if current != '/dev/null':
                out.setdefault(current, [])
            continue
        if line.startswith('--- ') or line.startswith('diff --git'):
            continue
        if current and current in out:
            out[current].append(line)
    return {name: ''.join(body) for name, body in out.items() if body}


def _apply_hunks(before: str, body: str) -> str:
    """Apply one file's hunks, matching by content rather than by line number.

    Line numbers in a patch are a hint, not an address: a patch generated
    against a file that has since changed above the hunk has the right context
    and the wrong numbers. So each hunk's context is *searched for*, near where
    it claims to be first, and the patch is refused if it is not found — which
    is the right failure, because applying it anyway is how a hunk lands in
    the wrong function.
    """
    lines = before.splitlines(keepends=True)
    result = list(lines)
    offset = 0

    for hunk in re.split(r'(?m)^(?=@@)', body):
        if not hunk.startswith('@@'):
            continue
        header = re.match(r'^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@', hunk)
        if not header:
            raise ValueError('a hunk header is malformed')
        start = int(header.group(1))

        old: list[str] = []
        new: list[str] = []
        for line in hunk.splitlines(keepends=True)[1:]:
            if line.startswith('-'):
                old.append(line[1:])
            elif line.startswith('+'):
                new.append(line[1:])
            elif line.startswith(' ') or line in ('\n', ''):
                text = line[1:] if line.startswith(' ') else line
                old.append(text)
                new.append(text)
            elif line.startswith('\\'):
                continue        # "\ No newline at end of file"

        at = _find(result, old, max(0, start - 1 + offset))
        if at is None:
            near = ''.join(old[:2]).strip()[:60]
            raise ValueError(f'a hunk did not match the file (looking for {near!r})')
        result[at : at + len(old)] = new
        offset += len(new) - len(old)

    return ''.join(result)


def _find(lines: list[str], block: list[str], near: int) -> int | None:
    """Where a block of lines occurs, preferring the position closest to `near`."""
    if not block:
        return None
    span = len(block)
    candidates = [i for i in range(0, max(1, len(lines) - span + 1)) if lines[i : i + span] == block]
    if not candidates:
        # Trailing whitespace is the commonest reason a patch that is
        # otherwise right does not match, and being strict about it helps
        # nobody, so it is tried once with the ends stripped.
        stripped = [line.rstrip() for line in block]
        candidates = [
            i for i in range(0, max(1, len(lines) - span + 1))
            if [x.rstrip() for x in lines[i : i + span]] == stripped
        ]
    if not candidates:
        return None
    return min(candidates, key=lambda i: abs(i - near))


def code_tools() -> list[Tool]:
    return [MultiEditTool(), ReadFilesTool(), OutlineTool(), ApplyPatchTool()]


__all__ = ['code_tools', 'outline_text']
