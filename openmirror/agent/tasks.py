"""Work that carries on while the conversation does.

Two kinds, one board: a shell command started in the background — a dev
server, a watcher, a test suite nobody wants to sit through — and a subagent
sent off with `background: true`. Both are things the agent starts, does not
wait for, and needs a way back to.

The board belongs to the session, for the reason the browser does: a dev
server nobody stopped is a port still held tomorrow. Closing the session stops
everything on it. Interrupting a turn does not. A turn being cut short is not
a request to kill the server it started, and conflating the two is how an
Escape meant for the model takes down the thing it was working against.

When something finishes the model is told at its next request, in the
conversation, without asking. Polling for a result costs a round trip a time;
being told costs nothing and cannot be forgotten.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# What one task keeps of its own output. The tail: for a build or a server it
# is the end that matters, and the head of a log that has run for an hour is
# its banner.
KEEP_CHARS = 200_000


@dataclass
class BackgroundTask:
    id: str
    kind: str                       # shell | agent
    label: str                      # the command, or what the agent was sent to do
    agent: str = ''                 # an agent's call id, which its events carry
    status: str = 'running'         # running | done | failed | stopped
    started: float = field(default_factory=time.time)
    ended: float | None = None
    exit_code: int | None = None
    report: str = ''                # an agent's final message
    # Who stopped it, when something did: `agent` through the tool, `person`
    # from the client, `session` on close. Only the first needs no telling.
    stopped_by: str = ''

    _buf: str = field(default='', repr=False)
    _dropped: int = field(default=0, repr=False)    # characters cut off the front of _buf
    _read: int = field(default=0, repr=False)       # how far `output` has read, absolutely
    _task: asyncio.Task[Any] | None = field(default=None, repr=False)
    _proc: Any = field(default=None, repr=False)
    _done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    @property
    def running(self) -> bool:
        return self.status == 'running'

    def write(self, text: str) -> None:
        self._buf += text
        if len(self._buf) > KEEP_CHARS:
            cut = len(self._buf) - KEEP_CHARS
            self._buf = self._buf[cut:]
            self._dropped += cut

    def unread(self) -> tuple[str, int]:
        """What has been printed since the last call, and how much of it was lost.

        Lost meaning scrolled out of the buffer before anyone asked — said,
        rather than silently skipped, so a model reading a log with a hole in
        it knows the hole is there.
        """
        start = max(self._read, self._dropped)
        lost = start - self._read
        text = self._buf[start - self._dropped:]
        self._read = self._dropped + len(self._buf)
        return text, lost

    def tail(self, lines: int = 20) -> str:
        return '\n'.join(self._buf.rstrip('\n').splitlines()[-lines:])

    def elapsed(self) -> float:
        return (self.ended or time.time()) - self.started

    def describe(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'kind': self.kind,
            'label': self.label,
            'agent': self.agent,
            'status': self.status,
            'started': self.started,
            'ended': self.ended,
            'exit_code': self.exit_code,
            'stopped_by': self.stopped_by,
        }


OnChange = Callable[[BackgroundTask], Awaitable[None]]


class TaskBoard:
    """Background work for one session."""

    def __init__(self, on_change: OnChange | None = None) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        self._count = 0
        self.on_change = on_change

    def get(self, task_id: str) -> BackgroundTask | None:
        return self._tasks.get(task_id.strip())

    def list(self) -> list[BackgroundTask]:
        return list(self._tasks.values())

    @property
    def running(self) -> list[BackgroundTask]:
        return [t for t in self._tasks.values() if t.running]

    def _new(self, kind: str, label: str, agent: str = '') -> BackgroundTask:
        self._count += 1
        task = BackgroundTask(id=f't{self._count}', kind=kind, label=label, agent=agent)
        self._tasks[task.id] = task
        return task

    # -- starting -----------------------------------------------------------

    async def start_shell(self, command: str, cwd: Path, env: dict[str, str]) -> BackgroundTask:
        from openmirror.agent.tools.shell import _spawn_kwargs

        task = self._new('shell', command)
        task._proc = await asyncio.create_subprocess_shell(
            command,
            # Nothing to type into. A background command that stops to read a
            # prompt would otherwise wait for ever for input that cannot come.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            # One stream, in the order it was written: a server's errors mean
            # most next to the line they interrupted.
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd),
            env=env,
            **_spawn_kwargs(),
        )
        task._task = asyncio.create_task(self._watch_shell(task))
        await self._changed(task)
        return task

    async def start_agent(self, work: Callable[[], Awaitable[Any]], label: str, agent: str) -> BackgroundTask:
        """Run `work()` in the background. A factory rather than a coroutine,
        so a task stopped before it ever ran leaves nothing half-made behind."""
        task = self._new('agent', label, agent)
        task._task = asyncio.create_task(self._watch_agent(task, work))
        await self._changed(task)
        return task

    # -- watching -----------------------------------------------------------

    async def _watch_shell(self, task: BackgroundTask) -> None:
        proc = task._proc
        # Incremental, because a read can end halfway through a character and
        # decoding each chunk alone would print a replacement glyph there.
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            task.write(decoder.decode(chunk))
        task.write(decoder.decode(b'', final=True))
        task.exit_code = await proc.wait()
        if task.running:
            task.status = 'done' if task.exit_code == 0 else 'failed'
        await self._finish(task)

    async def _watch_agent(self, task: BackgroundTask, work: Callable[[], Awaitable[Any]]) -> None:
        try:
            report = await work()
        except asyncio.CancelledError:
            # Whoever cancelled it is recording why; see stop().
            raise
        except Exception as exc:  # noqa: BLE001 - reported on the task, not raised into nothing
            log.exception('background agent %s failed', task.id)
            task.status = 'failed'
            task.report = f'{type(exc).__name__}: {exc}'
        else:
            task.report = getattr(report, 'text', str(report))
            if task.running:
                task.status = 'done' if getattr(report, 'ok', True) else 'failed'
        await self._finish(task)

    async def _finish(self, task: BackgroundTask) -> None:
        if task.ended is not None:
            return
        task.ended = time.time()
        task._done.set()
        await self._changed(task)

    async def _changed(self, task: BackgroundTask) -> None:
        if self.on_change is None:
            return
        try:
            await self.on_change(task)
        except Exception:  # noqa: BLE001 - a listener failing must not take the task with it
            log.exception('task listener failed for %s', task.id)

    # -- stopping -----------------------------------------------------------

    async def stop(self, task_id: str, by: str = 'agent') -> BackgroundTask:
        task = self.get(task_id)
        if task is None:
            raise KeyError(task_id)
        if not task.running:
            return task
        task.status = 'stopped'
        task.stopped_by = by

        if task._proc is not None and task._proc.returncode is None:
            from openmirror.agent.tools.shell import _kill_tree

            # Politely, then not: the same order the foreground shell uses on a
            # timeout, and for the same reason — a server given the chance to
            # shut down releases its port and its lock files.
            await _kill_tree(task._proc, hard=False)
            try:
                await asyncio.wait_for(task._proc.wait(), timeout=3)
            except TimeoutError:
                await _kill_tree(task._proc, hard=True)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(task._proc.wait(), timeout=3)

        if task._task is not None and not task._task.done():
            task._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task._task

        await self._finish(task)
        return task

    async def wait(self, task_id: str, seconds: float) -> bool:
        """Wait for a task to finish, up to `seconds`. True if it has."""
        task = self.get(task_id)
        if task is None:
            raise KeyError(task_id)
        if task.running and seconds > 0:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(task._done.wait(), timeout=seconds)
        return not task.running

    async def close(self) -> None:
        for task in self.running:
            try:
                await self.stop(task.id, by='session')
            except Exception:  # noqa: BLE001
                log.debug('background task %s did not stop cleanly', task.id, exc_info=True)
