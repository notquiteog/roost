"""Smoke tests for a desktop release, run by CI on a runner that has only the
installer — the stand-in for the machines nobody on the project owns.

    python desktop/ci/smoke.py daemon path/to/openmirror-daemon
    python desktop/ci/smoke.py app --stop kill:openmirror-desktop --again -- /usr/bin/openmirror-desktop
    python desktop/ci/smoke.py app --stop quit:openmirror -- open -n /Applications/openmirror.app

`daemon` runs the frozen daemon by itself. It has to answer, serve the page
and everything the page loads — which is how a file that freezing left out
shows up — and stop when its stdin closes.

`app` runs the installed app as a person would and checks the whole chain:
the app starts a daemon, the window loads the interface from it (the window's
own requests are in the daemon's log), and stopping the app stops the daemon.
`--stop kill:NAME` kills the app's process outright, which is what a crash
looks like too, so it tests the stdin pipe; `--stop quit:NAME` asks macOS to
quit the app, which tests the app's own shutdown; plain `--stop kill` kills
whatever was launched. `--again` launches a second copy, which should hand
over to the first and leave.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

# The app's port, which the app test has to use because the app does. Moved
# only for running this on a machine that already has a daemon on 8477.
PORT = int(os.environ.get('OPENMIRROR_PORT') or 8477)
LOGS = Path.home() / '.openmirror' / 'logs'

# The page, and a sample of what it loads. A missing module is a blank window
# in a release that worked from a checkout.
PAGE = ['/', '/static/app.js', '/static/native.js', '/static/style.css', '/static/companions/index.js']


class Failed(Exception):
    pass


def get(port: int, path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=3) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def is_openmirror(port: int) -> bool:
    try:
        status, body = get(port, '/healthz')
        return status == 200 and json.loads(body).get('service') == 'openmirror'
    except (OSError, ValueError):
        return False


def listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', port)) == 0


def wait(condition: Callable[[], bool], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.5)
    return condition()


def tail(path: Path, lines: int = 150) -> str:
    try:
        return '\n'.join(path.read_text(errors='replace').splitlines()[-lines:])
    except OSError as exc:
        return f'({exc})'


def kill_named(name: str) -> int:
    """Kill every process running the executable `name`, the way a crash
    ends one. By name, because what was launched is not always the app: it
    is `open` on macOS, and the AppImage's own runtime on Linux."""
    import psutil  # only this path needs it, and CI installs it

    killed = 0
    for proc in psutil.process_iter(['name', 'exe']):
        names = {proc.info['name'] or '', Path(proc.info['exe'] or '').name}
        if name in names or f'{name}.exe' in names:
            try:
                proc.kill()
                killed += 1
            except psutil.Error:
                pass
    return killed


def check_page(port: int) -> None:
    for path in PAGE:
        # Marked, so that these are never mistaken for the window's own.
        status, body = get(port, f'{path}?smoke')
        if status != 200:
            raise Failed(f'GET {path} answered {status}')
        if path == '/' and b'openmirror' not in body:
            raise Failed('GET / answered, but not with the interface')
    print(f'serves the page and {len(PAGE) - 1} of the files it loads')


def daemon(exe: Path) -> None:
    # Absolute, because it is started in a temporary directory, and a relative
    # program path is looked up from the new working directory — as the
    # workflow's relative path was, once, and found nothing.
    exe = exe.resolve()
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    tmp = Path(tempfile.mkdtemp(prefix='openmirror-smoke-'))
    log = tmp / 'daemon.log'
    env = {
        **os.environ,
        'OPENMIRROR_PORT': str(port),
        'OPENMIRROR_DATA_DIR': str(tmp / 'data'),
        'OPENMIRROR_ENV': str(tmp / 'absent.env'),
        'OPENMIRROR_EXIT_WITH_STDIN': '1',
    }
    started = time.monotonic()
    with log.open('wb') as out:
        proc = subprocess.Popen(
            [str(exe)], cwd=tmp, env=env, stdin=subprocess.PIPE, stdout=out, stderr=subprocess.STDOUT
        )
    try:
        if not wait(lambda: is_openmirror(port) or proc.poll() is not None, 180) or proc.poll() is not None:
            raise Failed(f'the daemon did not answer (exit code {proc.poll()})')
        print(f'answered after {time.monotonic() - started:.1f}s')
        check_page(port)

        proc.stdin.close()
        try:
            code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            raise Failed('still running 30s after its stdin closed') from None
        if code != 0:
            raise Failed(f'exited with {code} when its stdin closed')
        if listening(port):
            raise Failed('exited, but something still holds its port')
        print('stopped cleanly when its stdin closed')
    finally:
        if proc.poll() is None:
            proc.kill()
        print(f'--- {log}\n{tail(log)}')


def app(command: list[str], stop: str, again: bool) -> None:
    if listening(PORT):
        raise Failed(f'something was already listening on {PORT} before the app started')

    started = time.monotonic()
    proc = subprocess.Popen(command)
    try:
        if not wait(lambda: is_openmirror(PORT), 180):
            raise Failed(f'the app did not bring a daemon up on {PORT}')
        print(f'the app brought the daemon up after {time.monotonic() - started:.1f}s')
        check_page(PORT)

        # The window's own requests, as the daemon logged them: the difference
        # between a daemon that runs and an app that shows it. Only the window
        # asks for these paths bare — check_page marks its own — after a run
        # in which the test's requests alone satisfied this check, and the app
        # was killed before its window had asked for anything. An /api/ call
        # is the page's script running, not merely its files arriving.
        log = LOGS / 'daemon.log'
        wanted = ['"GET / HTTP/', '"GET /static/app.js HTTP/', '"GET /static/native.js HTTP/', '"GET /api/']
        if not wait(lambda: all(w in tail(log, 2000) for w in wanted), 90):
            raise Failed('the daemon is up, but the window never loaded the interface from it')
        print('the window loaded the interface')

        if again:
            second = subprocess.Popen(command)
            try:
                code = second.wait(timeout=30)
            except subprocess.TimeoutExpired:
                second.kill()
                raise Failed('a second launch did not hand over to the first and leave') from None
            if not is_openmirror(PORT):
                raise Failed('a second launch took the daemon down with it')
            print(f'a second launch handed over to the first and left ({code})')

        if stop == 'kill':
            proc.kill()
        elif stop.startswith('kill:'):
            if not kill_named(stop.removeprefix('kill:')):
                raise Failed(f'there was no {stop.removeprefix("kill:")} process to kill')
        elif stop.startswith('quit:'):
            subprocess.run(['osascript', '-e', f'quit app "{stop.removeprefix("quit:")}"'], check=True)
        else:
            raise Failed(f'no such way to stop: {stop}')

        if not wait(lambda: not listening(PORT), 45):
            raise Failed(f'the daemon outlived the app ({stop})')
        print(f'the daemon stopped with the app ({stop})')
    finally:
        if proc.poll() is None:
            proc.kill()
        for path in sorted(LOGS.glob('*.log')):
            print(f'--- {path}\n{tail(path)}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='what', required=True)
    d = sub.add_parser('daemon')
    d.add_argument('exe', type=Path)
    a = sub.add_parser('app')
    a.add_argument('--stop', default='kill', help='kill, or quit:NAME to ask macOS to quit it')
    a.add_argument('--again', action='store_true', help='also check that a second launch hands over')
    a.add_argument('command', nargs='+')
    args = parser.parse_args()

    try:
        if args.what == 'daemon':
            daemon(args.exe)
        else:
            app(args.command, args.stop, args.again)
    except Failed as exc:
        print(f'FAILED: {exc}', file=sys.stderr)
        raise SystemExit(1) from None
    print('ok')


if __name__ == '__main__':
    main()
