"""Freeze the daemon, and put it where Tauri looks for it.

    python desktop/sidecar/build.py

Run it with the Python whose packages should be frozen — an environment with
`openmirror[desktop,tor,vec]` and `pyinstaller` installed — on the platform the
release is for: PyInstaller does not cross-compile any more than Tauri does.

Tauri wants a sidecar named for the Rust target it ships with, so the result is
copied to `src-tauri/binaries/openmirror-daemon-<target>[.exe]`. The target
defaults to rustc's host, which is right whenever the Python doing the freezing
is native to the machine, as it is on every runner that builds a release.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINARIES = HERE.parent / 'src-tauri' / 'binaries'


def host_target() -> str:
    out = subprocess.run(['rustc', '-vV'], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith('host:'):
            return line.partition(':')[2].strip()
    raise SystemExit('rustc -vV did not report a host target')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--target', help="the Rust target triple to name it for (default: rustc's host)")
    parser.add_argument('--work', type=Path, default=HERE / 'build', help='where PyInstaller works')
    args = parser.parse_args()

    target = args.target or host_target()
    subprocess.run(
        [
            sys.executable,
            '-m',
            'PyInstaller',
            '--noconfirm',
            '--clean',
            '--distpath',
            str(args.work / 'dist'),
            '--workpath',
            str(args.work / 'work'),
            str(HERE / 'openmirror-daemon.spec'),
        ],
        check=True,
    )

    suffix = '.exe' if sys.platform == 'win32' else ''
    built = args.work / 'dist' / f'openmirror-daemon{suffix}'
    dest = BINARIES / f'openmirror-daemon-{target}{suffix}'
    BINARIES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, dest)
    print(f'{dest} ({dest.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
