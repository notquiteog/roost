"""Coming back to work left running in the background.

One tool with three actions rather than three tools. A tool list is not free —
a small model given thirty tools failed a task it finished with six — and
listing, reading and stopping background work are one thing to know about, not
three.
"""

from __future__ import annotations

from typing import Any

from openmirror.agent.tools.base import Assessment, Output, Tool, ToolContext, ToolError, truncate
from openmirror.protocol.agent import Risk

ACTIONS = ('list', 'output', 'stop')
MAX_WAIT = 600


def _took(seconds: float) -> str:
    if seconds < 60:
        return f'{seconds:.0f}s'
    if seconds < 3600:
        return f'{seconds // 60:.0f}m{seconds % 60:02.0f}s'
    return f'{seconds // 3600:.0f}h{(seconds % 3600) // 60:02.0f}m'


def status_line(task: Any) -> str:
    what = 'agent' if task.kind == 'agent' else 'command'
    if task.running:
        return f'{task.id} ({what}) is still running, {_took(task.elapsed())} so far'
    if task.status == 'stopped':
        return f'{task.id} ({what}) was stopped after {_took(task.elapsed())}'
    if task.kind == 'shell':
        return f'{task.id} (command) exited with code {task.exit_code} after {_took(task.elapsed())}'
    return f'{task.id} (agent) {"finished" if task.status == "done" else "failed"} after {_took(task.elapsed())}'


class TasksTool(Tool):
    name = 'tasks'
    description = (
        'Check on work running in the background: shell commands started with background: true, '
        'and agents started the same way. `list` shows every task and its state. `output` returns '
        'what a task has printed since you last asked — or an agent\'s report once it has finished '
        '— and with `wait` it first waits up to that many seconds for the task to finish. `stop` '
        'ends one. You are told when a task finishes without asking, so there is no need to poll.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'action': {'type': 'string', 'enum': list(ACTIONS)},
            'id': {'type': 'string', 'description': 'Which task: t1, t2 …'},
            'wait': {
                'type': 'integer',
                'description': f'For output: seconds to wait for it to finish first. At most {MAX_WAIT}.',
            },
        },
        'required': ['action'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        action = args.get('action') or 'list'
        if action not in ACTIONS:
            return Assessment(risk=Risk.READ, summary='', invalid=f'action must be one of {", ".join(ACTIONS)}')
        board = ctx.tasks
        if board is None:
            return Assessment(risk=Risk.READ, summary='', invalid='there is no background work in this session')
        if action == 'list':
            return Assessment(risk=Risk.READ, summary='list what is running')

        task_id = str(args.get('id') or '').strip()
        if not task_id:
            return Assessment(risk=Risk.READ, summary='', invalid=f'{action} needs the id of a task')
        task = board.get(task_id)
        if task is None:
            known = ', '.join(t.id for t in board.list())
            return Assessment(
                risk=Risk.READ, summary='',
                invalid=f'there is no task {task_id}. ' + (f'There are: {known}.' if known else 'Nothing has been started.'),
            )
        label = task.label if len(task.label) <= 80 else task.label[:77] + '...'
        if action == 'stop':
            # It kills a process, if only one this session started. Cheap to
            # confirm under a cautious mode, and it runs unasked under trusted.
            return Assessment(risk=Risk.EXECUTE, summary=f'stop {task.id}: {label}')
        return Assessment(risk=Risk.READ, summary=f'output of {task.id}: {label}')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        board = ctx.tasks
        action = args.get('action') or 'list'

        if action == 'list':
            tasks = board.list()
            if not tasks:
                return Output(content='Nothing has been started in the background.', display={'tasks': []})
            lines = [f'{status_line(t)}: {t.label}' for t in tasks]
            return Output(content='\n'.join(lines), display={'tasks': [t.describe() for t in tasks]})

        task = board.get(str(args['id']))
        if task is None:
            raise ToolError(f'there is no task {args["id"]}')

        if action == 'stop':
            await board.stop(task.id, by='agent')
            tail = task.tail(15) if task.kind == 'shell' else ''
            content = status_line(task) + (f'.\n\nIts last lines:\n{tail}' if tail else '.')
            return Output(content=content, display={'task': task.describe()})

        wait = max(0, min(int(args.get('wait') or 0), MAX_WAIT))
        if wait:
            await board.wait(task.id, wait)

        if task.kind == 'agent':
            if task.running:
                content = f'{status_line(task)}. Its report is not ready yet.'
            else:
                body, _ = truncate(task.report or '(it finished without saying anything)', 30_000, keep='head')
                content = f'{status_line(task)}. Its report:\n\n{body}'
            return Output(content=content, display={'task': task.describe()})

        text, lost = task.unread()
        body, cut = truncate(text, 30_000, keep='tail')
        parts = [status_line(task) + '.']
        if lost:
            parts.append(f'[{lost} characters scrolled away before this was read]')
        parts.append(body if body.strip() else '(nothing new since you last looked)')
        return Output(content='\n\n'.join(parts), display={'task': task.describe()}, truncated=cut)
