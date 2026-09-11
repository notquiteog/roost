"""The to-do list: the shape of a long piece of work, where a person can see it.

Not for the model's benefit — models track their own work well enough — but
for the person watching one grind through eleven steps with no idea which one
it is on, or whether the thing they asked for is still on the list. The client
pins the latest list above the composer, so it is the one thing that stays in
view while the transcript scrolls past.

This was the `plan` tool until plan *mode* arrived. Two things called "plan",
one of them a list and the other a promise not to touch anything, is one more
than a model can be relied on to keep apart.

Held on the tool rather than in a file, because it belongs to this session and
writing it into the working root would be the agent creating a file nobody
asked for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from openmirror.agent.tools.base import Assessment, Output, Tool, ToolContext
from openmirror.protocol.agent import Risk

STATES = ('todo', 'doing', 'done', 'dropped')

# The names other harnesses use for the same four states, and for the list
# and its fields. A model trained on one of them reaches for its words first,
# and mapping them costs nothing where refusing them costs a turn.
STATE_ALIASES = {
    'pending': 'todo', 'open': 'todo', 'not_started': 'todo',
    'in_progress': 'doing', 'active': 'doing', 'started': 'doing',
    'completed': 'done', 'complete': 'done', 'finished': 'done',
    'cancelled': 'dropped', 'canceled': 'dropped', 'skipped': 'dropped',
}

MARKS = {'todo': '[ ]', 'doing': '[~]', 'done': '[x]', 'dropped': '[-]'}


@dataclass(slots=True)
class Item:
    text: str
    state: str = 'todo'
    at: float = field(default_factory=time.time)


def _normalise(args: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The list as it was meant, or None if there is no list in it at all."""
    raw = args.get('items')
    if raw is None:
        raw = args.get('todos', args.get('steps'))
    if not isinstance(raw, list):
        return None
    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            out.append({'text': str(entry), 'state': 'todo'})
            continue
        text = entry.get('text') or entry.get('content') or entry.get('title') or ''
        state = str(entry.get('state') or entry.get('status') or 'todo').strip().lower()
        out.append({'text': str(text), 'state': STATE_ALIASES.get(state, state)})
    return out


class TodoTool(Tool):
    name = 'todo'
    description = (
        'Keep a to-do list for the work in hand, so the person watching can see what is done and '
        'what is left. Use it for anything that takes more than two or three steps, and not for a '
        'quick question. Send the whole list every time, each item marked todo, doing, done or '
        'dropped. Mark an item doing when you start it and done the moment it is finished — not in '
        'a batch at the end. While you work, exactly one item is doing. Keep items to real steps: '
        '"read the file" is not one.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'items': {
                'type': 'array',
                'description': 'The whole list, every time — not only what changed.',
                'items': {
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string'},
                        'state': {'type': 'string', 'enum': list(STATES)},
                    },
                    'required': ['text'],
                },
            },
        },
        'required': ['items'],
    }

    def __init__(self) -> None:
        self.items: list[Item] = []

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        items = _normalise(args)
        if items is None:
            return Assessment(risk=Risk.READ, summary='', invalid='items must be a list')
        for item in items:
            if not item['text'].strip():
                return Assessment(risk=Risk.READ, summary='', invalid='every item needs text')
            if item['state'] not in STATES:
                return Assessment(
                    risk=Risk.READ, summary='',
                    invalid=f'{item["state"]!r} is not a state — use todo, doing, done or dropped',
                )
        doing = [i for i in items if i['state'] == 'doing']
        if len(doing) > 1:
            return Assessment(
                risk=Risk.READ, summary='',
                invalid=f'{len(doing)} items are marked doing — mark exactly one, so the person '
                        'watching can tell where you are',
            )
        done = sum(1 for i in items if i['state'] == 'done')
        summary = f'{done} of {len(items)} done'
        if doing:
            summary += f' · {doing[0]["text"].strip()}'
        # A read: it changes nothing outside this conversation, and a to-do
        # list that needed approving would be one nobody wrote.
        return Assessment(risk=Risk.READ, summary=summary)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        self.items = [Item(text=i['text'].strip(), state=i['state']) for i in _normalise(args) or []]

        content = self.render() or '(the list is empty)'
        waiting = any(i.state == 'todo' for i in self.items)
        if waiting and not any(i.state == 'doing' for i in self.items):
            # The one nudge worth making: a list with work left and nothing in
            # hand is a list the person cannot read their place in.
            content += '\n\nNothing is marked doing. Mark the item you start next.'

        done = sum(1 for i in self.items if i.state == 'done')
        return Output(
            content=content,
            display={
                'items': [{'text': i.text, 'state': i.state} for i in self.items],
                'done': done,
                'total': len(self.items),
            },
        )

    def render(self) -> str:
        return '\n'.join(f'{MARKS[i.state]} {i.text}' for i in self.items)
