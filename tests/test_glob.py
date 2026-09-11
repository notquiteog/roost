"""`glob`, and what `**/` means.

Found in a live run: gemma4:12b's first move in a folder holding two Python
files was `glob **/*.py`, and it was told there were none — fnmatch reads
`**/` as "some directory, then a slash", so a file in the root never matched.
"""

from __future__ import annotations

from pathlib import Path

from openmirror.agent.tools.base import ToolContext
from openmirror.agent.tools.search import GlobTool


def ctx(root: Path) -> ToolContext:
    async def emit(text, stream):
        pass

    async def ask(question, options, multi):
        return ''

    return ToolContext(root=root, cwd=root, emit=emit, ask=ask, session_id='glob')


def tree(root: Path) -> None:
    for rel in ('geometry.py', 'pkg/shapes.py', 'pkg/deep/more.py', 'src/a.ts', 'src/x/b.ts', 'README.md'):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text('x\n')


async def found(root: Path, pattern: str) -> set[str]:
    out = await GlobTool().run({'pattern': pattern}, ctx(root))
    return set() if out.content.startswith('No files match') else set(out.content.splitlines())


async def test_double_star_includes_the_root_itself(tmp_path: Path):
    tree(tmp_path)
    assert await found(tmp_path, '**/*.py') == {'geometry.py', 'pkg/shapes.py', 'pkg/deep/more.py'}


async def test_double_star_in_the_middle_includes_no_directory_at_all(tmp_path: Path):
    tree(tmp_path)
    assert await found(tmp_path, 'src/**/*.ts') == {'src/a.ts', 'src/x/b.ts'}


async def test_a_bare_pattern_still_matches_by_name_anywhere(tmp_path: Path):
    tree(tmp_path)
    assert await found(tmp_path, '*.md') == {'README.md'}
    assert await found(tmp_path, 'shapes.py') == {'pkg/shapes.py'}
    assert await found(tmp_path, '**/*.rs') == set()
