"""Work left running: started, read back, stopped, and reported when it ends."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from openmirror.agent.approval import Mode
from openmirror.agent.runtime import build_session
from openmirror.agent.tasks import TaskBoard
from openmirror.agent.tools.base import ToolContext
from openmirror.agent.tools.tasks import TasksTool
from openmirror.protocol.agent import Risk, TaskUpdated, ToolCompleted
from openmirror.providers.base import StreamDone, StreamText, StreamToolUse, TextBlock
from tests.test_agent import ScriptedProvider, turn

posix = pytest.mark.skipif(sys.platform == 'win32', reason='process groups are a POSIX idea')


def ctx(root: Path, board: TaskBoard) -> ToolContext:
    async def emit(text, stream):
        pass

    async def ask(question, options, multi):
        return ''

    return ToolContext(root=root, cwd=root, emit=emit, ask=ask, session_id='bg', tasks=board)


async def test_a_command_left_running_can_be_read_back(tmp_path: Path):
    board = TaskBoard()
    task = await board.start_shell('echo one; sleep 0.2; echo two', tmp_path, dict(os.environ))
    assert task.running
    assert await board.wait(task.id, 10)
    assert task.status == 'done' and task.exit_code == 0

    text, lost = task.unread()
    assert 'one' in text and 'two' in text and lost == 0
    assert task.unread() == ('', 0), 'output is read once'


async def test_a_failing_command_is_reported_as_failing(tmp_path: Path):
    board = TaskBoard()
    task = await board.start_shell('echo oops >&2; exit 3', tmp_path, dict(os.environ))
    await board.wait(task.id, 10)
    assert task.status == 'failed' and task.exit_code == 3
    assert 'oops' in task.tail(), 'stderr is in the same stream, in order'


def running_in_group(group: int) -> list[int]:
    """The members of a process group that are still running.

    Not merely present: a killed child whose shell died with it is re-parented
    and stays a zombie until its new parent gets round to reaping it — which
    is that parent's business, not the task board's, and not evidence that
    anything is still running. `killpg(group, 0)` cannot tell the difference;
    /proc can.
    """
    proc = Path('/proc')
    if not proc.is_dir():
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return []
        return [group]
    live = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text()
        except OSError:
            continue
        # The command name can hold spaces and brackets; what follows the
        # last ')' is fixed: state, parent, group.
        state, _, pgrp = stat.rsplit(')', 1)[1].split()[:3]
        if int(pgrp) == group and state != 'Z':
            live.append(int(entry.name))
    return live


@posix
async def test_stopping_reaches_the_commands_children(tmp_path: Path):
    board = TaskBoard()
    task = await board.start_shell('sleep 30 & sleep 30; wait', tmp_path, dict(os.environ))
    group = os.getpgid(task._proc.pid)
    await asyncio.sleep(0.2)
    assert len(running_in_group(group)) >= 2, 'the shell and its sleeps should be running'
    await board.stop(task.id, by='person')
    assert task.status == 'stopped' and task.stopped_by == 'person'
    for _ in range(60):
        if not running_in_group(group):
            break
        await asyncio.sleep(0.05)
    assert not running_in_group(group), 'the backgrounded sleep outlived the stop'


async def test_the_tasks_tool_lists_reads_and_stops(tmp_path: Path):
    board = TaskBoard()
    tool = TasksTool()
    c = ctx(tmp_path, board)
    task = await board.start_shell('echo ready; sleep 30', tmp_path, dict(os.environ))

    assert tool.assess({'action': 'output', 'id': 't9'}, c).invalid
    assert tool.assess({'action': 'stop', 'id': task.id}, c).risk is Risk.EXECUTE
    assert tool.assess({'action': 'output', 'id': task.id}, c).risk is Risk.READ

    await asyncio.sleep(0.3)
    listed = await tool.run({'action': 'list'}, c)
    assert 't1' in listed.content and 'still running' in listed.content
    out = await tool.run({'action': 'output', 'id': 't1'}, c)
    assert 'ready' in out.content
    stopped = await tool.run({'action': 'stop', 'id': 't1'}, c)
    assert 'stopped' in stopped.content and not task.running


async def test_output_can_wait_for_the_end(tmp_path: Path):
    board = TaskBoard()
    await board.start_shell('sleep 0.3; echo finished', tmp_path, dict(os.environ))
    out = await TasksTool().run({'action': 'output', 'id': 't1', 'wait': 10}, ctx(tmp_path, board))
    assert 'exited with code 0' in out.content and 'finished' in out.content


async def test_the_model_is_told_when_background_work_ends(tmp_path: Path):
    provider = ScriptedProvider([
        [StreamToolUse(id='c1', name='shell', input={'command': 'sleep 0.3; echo built', 'background': True}),
         StreamDone(stop_reason='tool_use')],
        [StreamText(text='It is building.'), StreamDone()],
        [StreamText(text='It built.'), StreamDone()],
    ])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.TRUSTED)
    await session.start()
    events = await turn(session, 'build it in the background')

    started = [e for e in events if isinstance(e, ToolCompleted)][0]
    assert started.result.display['background'] and started.result.display['task'] == 't1'

    done = asyncio.Event()
    since = session.seq

    async def watch():
        async for event in session.events(since=since):
            if isinstance(event, TaskUpdated) and event.task['status'] != 'running':
                done.set()
                return

    watcher = asyncio.create_task(watch())
    await asyncio.wait_for(done.wait(), timeout=10)
    await watcher

    await turn(session, 'how did it go?')
    told = ' '.join(b.text for m in provider.seen[-1].messages for b in m.content if isinstance(b, TextBlock))
    assert 'Background task t1' in told and 'exited with code 0' in told and 'built' in told


async def test_interrupting_a_turn_leaves_background_work_alone(tmp_path: Path):
    """Stopping the model is not a request to stop the server it started."""
    provider = ScriptedProvider([
        [StreamToolUse(id='c1', name='shell', input={'command': 'sleep 30', 'background': True}),
         StreamToolUse(id='c2', name='shell', input={'command': 'sleep 30'}),
         StreamDone(stop_reason='tool_use')],
    ])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.TRUSTED)
    await session.start()
    session.submit('start the server, then wait')
    for _ in range(300):
        if session.tasks.get('t1') is not None:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.2)
    session.interrupt()
    for _ in range(300):
        if not session.busy:
            break
        await asyncio.sleep(0.01)
    assert session.tasks.get('t1').running
    await session.close()
    assert session.tasks.get('t1').status == 'stopped'


async def test_a_subagent_cannot_leave_anything_running(tmp_path: Path):
    from openmirror.agent.tools.base import ToolError
    from openmirror.agent.tools.shell import ShellTool

    no_board = ctx(tmp_path, None)
    with pytest.raises(ToolError, match='foreground'):
        await ShellTool().run({'command': 'sleep 1', 'background': True}, no_board)
