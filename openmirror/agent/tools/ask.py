"""Asking the person a question.

A tool rather than a special message type, because it has to be something the
model can choose in the middle of its own reasoning — after reading two files
and discovering they contradict each other, not only at the start of a turn.

It is exempt from every approval policy, including the most permissive one.
Auto-approving a question would answer it on the person's behalf, which is
the exact opposite of what it is for.
"""

from __future__ import annotations

from typing import Any

from openmirror.agent.tools.base import Assessment, Output, Tool, ToolContext
from openmirror.protocol.agent import Risk


class AskUserTool(Tool):
    name = 'ask_user'
    description = (
        'Ask the person a question and wait for their answer. Use this when the request is '
        'genuinely ambiguous and different readings would lead to materially different work, '
        'or before something irreversible where their intent is unclear. '
        'Do not use it for things you can determine yourself by reading the code, or for '
        'choices with an obvious default — make those and say what you chose.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'question': {'type': 'string', 'description': 'The question, in full.'},
            'options': {
                'type': 'array',
                'items': {'type': 'string'},
                'description': 'Suggested answers. The person can always write their own instead.',
            },
            'multi': {'type': 'boolean', 'description': 'Whether several options may be chosen.'},
        },
        'required': ['question'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        question = (args.get('question') or '').strip()
        if not question:
            return Assessment(risk=Risk.READ, summary='', invalid='question is required')
        shown = question if len(question) <= 100 else question[:97] + '...'
        return Assessment(risk=Risk.READ, summary=shown)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        answer = await ctx.ask(
            args['question'],
            [str(o) for o in (args.get('options') or [])],
            bool(args.get('multi')),
        )
        return Output(content=answer, display={'question': args['question'], 'answer': answer})
