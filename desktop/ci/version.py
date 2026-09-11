"""Check that a release tag and the versions it ships agree.

    python desktop/ci/version.py v0.2.0     # the tag being released
    python desktop/ci/version.py            # only that the files agree

The daemon's version is in pyproject.toml and the app's in Cargo.toml — the
Tauri config deliberately has none, so it takes Cargo's. A tag that disagrees
with either would publish installers whose About box and file names contradict
the release they came from, so the release stops here instead. Prints the
version on success, for the workflow to name things with.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def versions() -> dict[str, str]:
    pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    cargo = tomllib.loads((ROOT / 'desktop/src-tauri/Cargo.toml').read_text())
    found = {
        'pyproject.toml': pyproject['project']['version'],
        'desktop/src-tauri/Cargo.toml': cargo['package']['version'],
    }
    conf = json.loads((ROOT / 'desktop/src-tauri/tauri.conf.json').read_text())
    if 'version' in conf:
        found['desktop/src-tauri/tauri.conf.json'] = conf['version']
    return found


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else None
    found = versions()
    want = tag.removeprefix('v') if tag else found['pyproject.toml']
    wrong = {path: v for path, v in found.items() if v != want}
    if wrong:
        against = f'the tag {tag}' if tag else 'pyproject.toml'
        for path, v in wrong.items():
            print(f'{path} says {v}, which does not match {against} ({want})', file=sys.stderr)
        raise SystemExit(1)
    print(want)


if __name__ == '__main__':
    main()
