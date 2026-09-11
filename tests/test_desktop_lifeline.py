"""The daemon stops when the desktop app that started it goes away.

The app holds a pipe to the daemon's stdin and never writes to it. The pipe
closing is the signal, and it closes however the app ends — quit, crash or
killed — because the operating system closes it. These run the real daemon in
a subprocess, because what is under test is the process boundary itself.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _answers(port: int) -> bool:
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2) as r:
            return json.loads(r.read()).get('service') == 'openmirror'
    except (OSError, ValueError):
        return False


def _start(tmp_path: Path, port: int, *, exit_with_stdin: bool) -> subprocess.Popen:
    env = {
        **os.environ,
        'OPENMIRROR_PORT': str(port),
        'OPENMIRROR_DATA_DIR': str(tmp_path / 'data'),
        # Not the checkout's own .env, which may configure anything at all.
        'OPENMIRROR_ENV': str(tmp_path / 'absent.env'),
        'OPENMIRROR_EXIT_WITH_STDIN': '1' if exit_with_stdin else '0',
    }
    log = (tmp_path / 'daemon.log').open('wb')
    return subprocess.Popen(
        [sys.executable, '-m', 'openmirror.main'],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.PIPE,
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def _wait_up(proc: subprocess.Popen, port: int, tmp_path: Path) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if _answers(port):
            return
        if proc.poll() is not None:
            break
        time.sleep(0.2)
    proc.kill()
    pytest.fail('the daemon never answered:\n' + (tmp_path / 'daemon.log').read_text(errors='replace'))


def test_closing_stdin_stops_the_daemon(tmp_path):
    port = _free_port()
    proc = _start(tmp_path, port, exit_with_stdin=True)
    try:
        _wait_up(proc, port, tmp_path)
        proc.stdin.close()
        # A clean exit, through the same shutdown a Ctrl-C gets — not a kill.
        assert proc.wait(timeout=30) == 0
        assert not _answers(port)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_without_the_flag_a_closed_stdin_changes_nothing(tmp_path):
    """Run from a terminal, stdin is the keyboard, and Ctrl-D is not "stop"."""
    port = _free_port()
    proc = _start(tmp_path, port, exit_with_stdin=False)
    try:
        _wait_up(proc, port, tmp_path)
        proc.stdin.close()
        time.sleep(1.5)
        assert proc.poll() is None
        assert _answers(port)
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_a_taken_port_is_a_failed_start(tmp_path):
    """What `uvicorn.run` did before `main` needed the server object: whoever
    started the daemon hears that it did not start, rather than a clean exit."""
    with socket.socket() as taken:
        taken.bind(('127.0.0.1', 0))
        taken.listen()
        port = taken.getsockname()[1]
        proc = _start(tmp_path, port, exit_with_stdin=True)
        try:
            assert proc.wait(timeout=60) == 3
        finally:
            if proc.poll() is None:
                proc.kill()
