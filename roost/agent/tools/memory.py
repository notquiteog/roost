"""Remembering and recalling, as tools the agent can choose.

Recall also happens automatically before each turn, so why a tool as well?
Because automatic recall only ever sees the opening message. An agent that
reads three files and *then* realises it needs to know how this person likes
migrations written has no way to ask for that unless it can.

`remember` is deliberately not automatic here. Writing to someone's long-term
memory is a side effect, so it is a visible, approvable act like any other —
which is also why it carries a real risk grade rather than being waved through
as a read.
"""

from __future__ import annotations

from typing import Any

from roost.agent.tools.base import Assessment, Output, Tool, ToolContext, ToolError
from roost.protocol.agent import Risk


class RememberTool(Tool):
    name = 'remember'
    description = (
        'Store one durable fact about this person or their setup, for future conversations. '
        'Use it for things that stay true: how they like things done, what their tools are, '
        'decisions they have made. Do not use it for anything that was only true today, for '
        'anything you can read from the code, or for secrets and credentials.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'text': {
                'type': 'string',
                'description': 'The fact, written so it still makes sense on its own in a month.',
            }
        },
        'required': ['text'],
    }

    def __init__(self, service: Any, user_id: str) -> None:
        self.service = service
        self.user_id = user_id

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        text = (args.get('text') or '').strip()
        if not text:
            return Assessment(risk=Risk.READ, summary='', invalid='text is required')
        if len(text) > 500:
            return Assessment(risk=Risk.WRITE, summary='', invalid='too long for one fact; keep it under 500 characters')
        shown = text if len(text) <= 90 else text[:87] + '...'
        # A write, not a read: it persists past this session, and a person
        # should get the chance to say no to what is being recorded about them.
        return Assessment(risk=Risk.WRITE, summary=f'remember: {shown}')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        memory = await self.service.remember(
            self.user_id, args['text'], kind='fact', source=ctx.session_id
        )
        if memory is None:
            # Said plainly so the model stops trying rather than rephrasing.
            raise ToolError(
                'Memory is switched off for this user, so nothing was stored. '
                'Do not try again this session.'
            )
        return Output(content=f'Remembered: {memory.text}', display={'id': memory.id})


class RecallTool(Tool):
    name = 'recall'
    description = (
        'Search what you remember about this person from earlier conversations. '
        'Relevant memories are already provided at the start of a turn, so use this only when '
        'something you learned partway through makes you want to check.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'query': {'type': 'string', 'description': 'What you want to know.'},
            'limit': {'type': 'integer', 'description': 'How many to return. Default 5.'},
        },
        'required': ['query'],
    }

    def __init__(self, service: Any, user_id: str) -> None:
        self.service = service
        self.user_id = user_id

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        query = (args.get('query') or '').strip()
        if not query:
            return Assessment(risk=Risk.READ, summary='', invalid='query is required')
        return Assessment(risk=Risk.READ, summary=f'recall {query!r}')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        result = await self.service.recall(
            self.user_id, args['query'], limit=int(args.get('limit') or 5)
        )
        if not result.memories:
            return Output(content=result.reason or 'Nothing relevant is remembered.')

        lines = [f'- {m.text}   ({m.score:.2f})' for m in result.memories]
        return Output(
            content='\n'.join(lines),
            display={'count': len(result.memories)},
        )
