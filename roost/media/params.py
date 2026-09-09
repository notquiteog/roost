"""What you are allowed to turn, and how far.

"As advanced as the user wants" is a UI problem before it is a model problem.
A generator has thirty knobs, a person wants two of them most days, and the
other twenty-eight have to be *there* rather than hidden behind a text field
where a typo becomes a silent default. So a provider describes its parameters
and the client renders them, which means adding a sampler to a backend adds a
dropdown to the page and nothing else has to know.

Three things make this more than a form description.

**The choices are asked of the server, not listed here.** A1111's samplers,
schedulers, upscalers and LoRAs are what that install actually has, and
hard-coding "DPM++ 2M" is how a UI offers a sampler that was removed two
releases ago. `describe()` is therefore async: it is allowed to ask.

**Every parameter says what it does, in a sentence.** Not for decoration —
these are the tooltips, and a person who has never set a CFG scale should be
able to find out what it is without leaving the page. The same sentences go to
the model, so it can pick sensibly rather than copying numbers out of its
training data.

**`advanced` is a disclosure hint, never a restriction.** Everything is
settable from the API whatever this says; it only decides what is behind the
fold on first open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Kind = Literal['int', 'float', 'string', 'text', 'bool', 'enum', 'seed', 'image']


@dataclass(slots=True)
class Param:
    name: str
    label: str
    kind: Kind = 'string'
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    options: list[str] = field(default_factory=list)
    help: str = ''
    # Behind the fold on first open. Not a permission.
    advanced: bool = False
    # For grouping into sections: 'prompt', 'shape', 'sampling', 'quality',
    # 'motion', 'output'. A client that does not know a group shows it last.
    group: str = 'sampling'

    def to_json(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'label': self.label,
            'kind': self.kind,
            'default': self.default,
            'min': self.minimum,
            'max': self.maximum,
            'step': self.step,
            'options': self.options,
            'help': self.help,
            'advanced': self.advanced,
            'group': self.group,
        }


def coerce(params: list[Param], given: dict[str, Any]) -> dict[str, Any]:
    """Clean a request against a schema, dropping what does not belong.

    Silently dropping an unknown key would be wrong for a typo and right for a
    stale client, and there is no way to tell them apart from here — so unknown
    keys are dropped and *reported*, and the caller decides whether to mention
    it. Values out of range are clamped rather than refused: a person dragging
    a slider to 151 steps meant 150, not an error page.
    """
    schema = {p.name: p for p in params}
    out: dict[str, Any] = {}
    for key, value in given.items():
        param = schema.get(key)
        if param is None or value is None or value == '':
            continue
        try:
            if param.kind in ('int', 'seed'):
                value = int(value)
            elif param.kind == 'float':
                value = float(value)
            elif param.kind == 'bool':
                value = bool(value) if not isinstance(value, str) else value.lower() in ('1', 'true', 'yes', 'on')
            elif param.kind == 'enum':
                value = str(value)
                if param.options and value not in param.options:
                    # Kept anyway. The options came from the server a moment
                    # ago and may have changed since; refusing a value the
                    # backend would have accepted is worse than passing it on
                    # and letting the backend say so.
                    pass
            else:
                value = str(value)
        except (TypeError, ValueError):
            continue

        if param.minimum is not None and isinstance(value, (int, float)):
            value = max(param.minimum, value)
        if param.maximum is not None and isinstance(value, (int, float)):
            value = min(param.maximum, value)
        if param.kind == 'int':
            value = int(value)
        out[key] = value
    return out


def unknown_keys(params: list[Param], given: dict[str, Any]) -> list[str]:
    known = {p.name for p in params}
    return sorted(k for k in given if k not in known and given[k] not in (None, ''))


def defaults(params: list[Param]) -> dict[str, Any]:
    return {p.name: p.default for p in params if p.default is not None}


# ---------------------------------------------------------------------------
# The parameters common to every image generator
# ---------------------------------------------------------------------------


def common_image() -> list[Param]:
    """What every image backend has in some form.

    Kept in one place so that the prompt box, the size and the count are the
    same three controls whichever backend is selected — switching provider
    should not move the field you were typing in.
    """
    return [
        Param('prompt', 'Prompt', 'text', group='prompt',
              help='What to make. Longer and more specific beats a list of adjectives.'),
        Param('negative_prompt', 'Negative prompt', 'text', group='prompt', advanced=True,
              help='What to keep out. Ignored by backends that have no negative '
                   'conditioning — OpenAI\'s among them.'),
        Param('n', 'How many', 'int', default=1, minimum=1, maximum=8, step=1, group='output',
              help='Images per request. Every one of them costs.'),
        Param('seed', 'Seed', 'seed', default=-1, group='sampling', advanced=True,
              help='-1 picks a new one each time. Reusing a seed with the same '
                   'prompt and settings reproduces the same image, which is how you '
                   'change one thing and see only that change.'),
    ]
