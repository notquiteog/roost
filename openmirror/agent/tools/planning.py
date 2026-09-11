"""Leaving plan mode: the plan, put to a person.

Plan mode is read_only with one way out, and this is it. The agent
investigates with the tools that only look, then calls this with what it
intends to do. The person reads the plan and says yes — choosing, as they do,
how much it may then do unasked — or says what to change.

The answer is a person's, always. The question goes through the same path as
`ask_user`, which no policy answers on anyone's behalf; a mode whose promise
is "you see the plan first" cannot have an exit the agent opens itself. And the
tool is shown to the model only while the session is planning, because a
model that can see a way out of a mode it is not in will try to take it.
"""

from __future__ import annotations

from typing import Any

from openmirror.agent.approval import Mode
from openmirror.agent.tools.base import Assessment, Output, Tool, ToolContext, ToolError
from openmirror.protocol.agent import Risk

APPROVE_ASK = 'Yes — ask me before each change'
APPROVE_EDIT = 'Yes — make the edits without asking'
KEEP_PLANNING = 'No — keep planning'


class ProposePlanTool(Tool):
    name = 'propose_plan'
    description = (
        'Put your plan to the person and ask to go ahead. For plan mode, where nothing that changes '
        'anything can run: investigate first with the tools that read, then call this with the plan — '
        'what you will change, in which files, in what order, and how you will check it worked. Write '
        'it for someone deciding whether to let you start. If they approve, the mode changes and you '
        'carry it out; if not, their answer says what to change.'
    )
    input_schema = {
        'type': 'object',
        'properties': {'plan': {'type': 'string', 'description': 'The plan, in markdown.'}},
        'required': ['plan'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        plan = str(args.get('plan') or '').strip()
        if not plan:
            return Assessment(risk=Risk.READ, summary='', invalid='plan is required')
        first = next((line.strip('#*- ').strip() for line in plan.splitlines() if line.strip('#*- ').strip()), '')
        return Assessment(risk=Risk.READ, summary=first if len(first) <= 100 else first[:97] + '...')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        policy = ctx.policy
        if policy is None or ctx.set_mode is None:
            raise ToolError('there is no plan to approve here')
        if policy.mode is not Mode.PLAN:
            return Output(content='This session is not planning, so there is nothing to approve. Carry on.')

        plan = str(args['plan']).strip()
        previous = policy.previous if policy.previous not in (None, Mode.PLAN, Mode.READ_ONLY) else None

        options: list[str] = []
        back = ''
        if previous is not None and previous not in (Mode.ASK, Mode.AUTO_EDIT):
            back = f'Yes — carry on in {previous.value} mode'
            options.append(back)
        options += [APPROVE_ASK, APPROVE_EDIT, KEEP_PLANNING]

        answer = await ctx.ask(f'The plan:\n\n{plan}\n\nGo ahead?', options, False)

        chosen = {APPROVE_ASK: Mode.ASK, APPROVE_EDIT: Mode.AUTO_EDIT}.get(answer)
        if back and answer == back:
            chosen = previous
        if chosen is None:
            said = '' if answer == KEEP_PLANNING else answer.strip()
            return Output(
                content=(
                    'The plan was not approved.' + (f' They said: {said}' if said else '')
                    + '\n\nYou are still in plan mode. Revise the plan with that in mind — investigate '
                    'further if you need to — and propose it again.'
                ),
                display={'approved': False, 'answer': answer},
            )

        await ctx.set_mode(chosen.value)
        # Directive rather than descriptive. "You can make changes" read to a
        # 12B model as the end of the task: it replied with nothing and the
        # turn ended, the plan approved and not one line of it done.
        return Output(
            content=(
                f'Approved. The session is now in {chosen.value} mode and the editing tools are available. '
                'Start carrying out the plan now, beginning with the first change, and keep the to-do list '
                'current as you go.'
            ),
            display={'approved': True, 'mode': chosen.value},
        )
