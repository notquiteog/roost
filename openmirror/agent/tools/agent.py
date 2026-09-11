"""Handing work to another agent.

Two things make a subagent worth its cost, and the description says both
because a model left to guess uses them for everything or for nothing:

**A fresh context.** A broad search — "everywhere the session id is parsed" —
reads thirty files to find the four that matter. Done here, all thirty stay in
this conversation for the rest of it. Done by an agent, only its report does.

**Parallel work.** Several `agent` calls in one response run at the same time.
Each works on the real tree, so two given overlapping edits would collide — the
read-before-write rule catches it, since one agent's write is a change on disk
to the other, but the fix is to give them work that does not overlap.

The call itself is graded `read`. Starting an agent changes nothing; every
call the agent makes is graded and, where the mode says so, asked about, with
the same policy and the same person as this session's own. Grading the
delegation as well would ask once for the brief and again for every step of
it, which is how an approval prompt becomes something people click through.
"""

from __future__ import annotations

from typing import Any

from openmirror.agent.agents import AgentKind
from openmirror.agent.tools.base import Assessment, Output, Tool, ToolContext, ToolError
from openmirror.protocol.agent import Risk

DESCRIPTION = (
    'Hand a self-contained piece of work to another agent, which does it with a fresh context of '
    'its own and reports back. Worth it for two things: a broad search whose working-out you do '
    'not need to keep ("find everywhere the session id is parsed"), and independent pieces of work '
    'that can happen at once — several agent calls in one response run in parallel, so give them '
    'work that does not overlap. Not worth it for anything you could do in a call or two yourself. '
    'The agent sees none of this conversation: put everything it needs in the task — the goal, '
    'what you already know, and what to hand back. Its final message is what you get.'
)


class AgentTool(Tool):
    name = 'agent'

    def __init__(self, kinds: dict[str, AgentKind]) -> None:
        self.kinds = kinds
        listing = '\n'.join(f'- {k.name}: {k.description}' for k in kinds.values())
        self.description = f'{DESCRIPTION}\n\nAgents:\n{listing}'
        self.input_schema = {
            'type': 'object',
            'properties': {
                'task': {
                    'type': 'string',
                    'description': 'The whole brief. The agent knows nothing you do not tell it here.',
                },
                'agent': {
                    'type': 'string',
                    'enum': sorted(kinds),
                    'description': 'Which agent. Default general.',
                },
                'title': {
                    'type': 'string',
                    'description': 'Three to six words for the person watching, e.g. "find the auth checks".',
                },
                'background': {
                    'type': 'boolean',
                    'description': (
                        'Start it and carry on without waiting. You are told when it finishes, and '
                        'its report comes with that.'
                    ),
                },
            },
            'required': ['task'],
        }

    def _kind(self, args: dict[str, Any]) -> str:
        # `subagent_type` is what a model trained on another harness sends.
        return str(args.get('agent') or args.get('subagent_type') or 'general').strip().lower()

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        task = str(args.get('task') or args.get('prompt') or '').strip()
        if not task:
            return Assessment(
                risk=Risk.READ, summary='',
                invalid='task is required — the whole brief, since the agent sees nothing of this conversation',
            )
        kind = self._kind(args)
        if kind not in self.kinds:
            return Assessment(
                risk=Risk.READ, summary='',
                invalid=f'there is no agent called {kind!r}. There are: {", ".join(sorted(self.kinds))}',
            )
        title = str(args.get('title') or args.get('description') or '').strip()
        shown = title or task.splitlines()[0]
        shown = shown if len(shown) <= 80 else shown[:77] + '...'
        later = ' (in the background)' if args.get('background') else ''
        return Assessment(risk=Risk.READ, summary=f'{kind}: {shown}{later}')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        if ctx.spawn is None:
            raise ToolError('an agent cannot start agents of its own — do the work yourself')
        task = str(args.get('task') or args.get('prompt') or '').strip()
        title = str(args.get('title') or args.get('description') or '').strip()
        return await ctx.spawn(self.kinds[self._kind(args)], task, title, bool(args.get('background')))
