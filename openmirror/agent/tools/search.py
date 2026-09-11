"""Finding things: by name, and by content.

Both shell out to `rg` when it is present and fall back to Python when it is
not, because ripgrep is roughly two orders of magnitude faster on a real
repository and already knows to skip `.git`, `node_modules` and anything in a
`.gitignore`. The fallback exists so the agent is not simply broken on a box
without it, not because it is meant to be pleasant.

Results are capped and sorted by modification time, newest first. On a large
tree the useful match is nearly always in something touched recently, and an
uncapped grep can return more text than the model can read.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shutil
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
from openmirror.protocol.agent import Risk

SKIP_DIRS = {
    '.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build',
    '.next', '.nuxt', 'target', '.mypy_cache', '.pytest_cache', '.ruff_cache',
    'vendor', '.tox', '.gradle', 'Pods', '.terraform',
}
MAX_RESULTS = 200


class GlobTool(Tool):
    name = 'glob'
    description = (
        'Find files by name pattern, e.g. "**/*.py" or "src/**/test_*.ts". '
        'Returns paths sorted by modification time, newest first.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'pattern': {'type': 'string', 'description': 'Glob pattern.'},
            'path': {'type': 'string', 'description': 'Directory to search in. Defaults to the working directory.'},
        },
        'required': ['pattern'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        if not args.get('pattern'):
            return Assessment(risk=Risk.READ, summary='', invalid='pattern is required')
        return Assessment(risk=Risk.READ, summary=f'glob {args["pattern"]}')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        root = resolve_in_root(args['path'], ctx) if args.get('path') else ctx.cwd
        pattern = args['pattern']

        matches: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith('.')]
            for name in filenames:
                full = Path(dirpath) / name
                rel = full.relative_to(root)
                # Matched against both the relative path and the bare name, so
                # "*.py" behaves the way people expect rather than only matching
                # files directly in the search root.
                if fnmatch.fnmatch(str(rel), pattern) or fnmatch.fnmatch(name, pattern):
                    matches.append(full)
            if len(matches) > MAX_RESULTS * 5:
                break

        def mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        matches.sort(key=mtime, reverse=True)
        shown = matches[:MAX_RESULTS]

        if not shown:
            return Output(content=f'No files match {pattern!r} under {root}.')

        body = '\n'.join(str(p.relative_to(root)) for p in shown)
        if len(matches) > len(shown):
            body += f'\n\n[{len(matches) - len(shown)} more matches not shown; narrow the pattern]'
        return Output(content=body, display={'count': len(matches), 'root': str(root)})


class GrepTool(Tool):
    name = 'grep'
    description = (
        'Search file contents with a regular expression. Use this to find where something is '
        'defined or used. Prefer it over running grep through the shell: it skips build '
        'directories and caps its own output.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'pattern': {'type': 'string', 'description': 'Regular expression.'},
            'path': {'type': 'string', 'description': 'Directory or file to search.'},
            'glob': {'type': 'string', 'description': 'Only search files matching this glob, e.g. "*.ts".'},
            'case_insensitive': {'type': 'boolean'},
            'context': {'type': 'integer', 'description': 'Lines of context around each match. Default 0.'},
        },
        'required': ['pattern'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        pattern = args.get('pattern')
        if not pattern:
            return Assessment(risk=Risk.READ, summary='', invalid='pattern is required')
        try:
            re.compile(pattern)
        except re.error as exc:
            return Assessment(risk=Risk.READ, summary='', invalid=f'invalid regular expression: {exc}')
        return Assessment(risk=Risk.READ, summary=f'grep {pattern!r}')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        root = resolve_in_root(args['path'], ctx) if args.get('path') else ctx.cwd
        if shutil.which('rg'):
            return await self._ripgrep(args, root)
        return await self._python(args, root)

    async def _ripgrep(self, args: dict[str, Any], root: Path) -> Output:
        argv = ['rg', '--line-number', '--with-filename', '--color', 'never', '--max-columns', '400']
        if args.get('case_insensitive'):
            argv.append('-i')
        if args.get('context'):
            argv += ['-C', str(int(args['context']))]
        if args.get('glob'):
            argv += ['--glob', args['glob']]
        argv += ['--regexp', args['pattern'], str(root)]

        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        except TimeoutError:
            proc.kill()
            raise ToolError('search timed out after 60s — narrow the pattern or the path') from None

        # rg exits 1 for "no matches", which is not an error.
        if proc.returncode not in (0, 1):
            raise ToolError(f'rg failed: {stderr.decode("utf-8", "replace")[:300]}')

        text = stdout.decode('utf-8', 'replace')
        if not text.strip():
            return Output(content=f'No matches for {args["pattern"]!r} under {root}.')

        lines = text.splitlines()
        capped = lines[: MAX_RESULTS * 3]
        body = '\n'.join(line.replace(f'{root}/', '', 1) for line in capped)
        if len(lines) > len(capped):
            body += f'\n\n[{len(lines) - len(capped)} more matching lines not shown]'
        body, cut = truncate(body, 40_000, keep='head')
        return Output(content=body, display={'matches': len(lines), 'engine': 'rg'}, truncated=cut)

    async def _python(self, args: dict[str, Any], root: Path) -> Output:
        # Off the event loop. Walking a large tree and reading every file is
        # seconds of blocking work, and this process may be holding a voice
        # call at the same time — a stalled loop there is audible.
        results = await asyncio.to_thread(
            _scan,
            root,
            args['pattern'],
            bool(args.get('case_insensitive')),
            args.get('glob'),
        )

        if not results:
            return Output(content=f'No matches for {args["pattern"]!r} under {root}.')
        body, cut = truncate('\n'.join(results), 40_000, keep='head')
        return Output(content=body, display={'matches': len(results), 'engine': 'python'}, truncated=cut)


def _scan(root: Path, pattern: str, case_insensitive: bool, globpat: str | None) -> list[str]:
    """The ripgrep fallback, synchronous by nature. Run in a worker thread."""
    rx = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)
    results: list[str] = []

    targets = [root] if root.is_file() else []
    if not targets:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith('.')]
            for name in filenames:
                if globpat and not fnmatch.fnmatch(name, globpat):
                    continue
                targets.append(Path(dirpath) / name)

    for file in targets:
        if len(results) >= MAX_RESULTS * 3:
            break
        try:
            if file.stat().st_size > 5_000_000:
                continue
            with file.open('r', encoding='utf-8', errors='replace') as fh:
                for n, line in enumerate(fh, 1):
                    if rx.search(line):
                        rel = file.relative_to(root) if file != root else file.name
                        results.append(f'{rel}:{n}:{line.rstrip()[:400]}')
                        if len(results) >= MAX_RESULTS * 3:
                            break
        except (OSError, UnicodeDecodeError):
            continue

    return results
