"""Loading a skill: the instructions for a kind of job, when that job turns up.

See openmirror/agent/skills.py for where skills come from and why only their
names are listed up front.

Reading a skill's own files is part of this tool rather than left to
`read_file`, because a person's skills live in their home directory and a
confined session cannot read there. The window this opens is exactly one
folder wide — the skill's own — and a path that resolves outside it is
refused the same way `read_file` refuses one outside the root.
"""

from __future__ import annotations

from typing import Any

from openmirror.agent.skills import Skill, files_of, render
from openmirror.agent.tools.base import Assessment, Output, Tool, ToolContext, ToolError, truncate
from openmirror.protocol.agent import Risk

DESCRIPTION = (
    'Load a skill: instructions packaged for a particular kind of job, on this machine or in this '
    'project. When a request matches a skill below, load it before you start and follow it — it '
    'knows things about the job that you do not. `file` reads one of the files that come with a '
    'skill, when its instructions point at one.'
)


def _short(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[: limit - 3] + '...'


class SkillTool(Tool):
    name = 'skill'

    def __init__(self, skills: dict[str, Skill]) -> None:
        self.skills = skills
        offered = [s for s in skills.values() if s.model_invocable]
        listing = '\n'.join(f'- {s.name}: {_short(s.description)}' for s in offered[:60])
        self.description = f'{DESCRIPTION}\n\nSkills:\n{listing}'
        self.input_schema = {
            'type': 'object',
            'properties': {
                'name': {'type': 'string', 'enum': sorted(s.name for s in offered)},
                'arguments': {'type': 'string', 'description': 'Anything the skill should be given, as text.'},
                'file': {'type': 'string', 'description': 'A file that comes with the skill, to read instead.'},
            },
            'required': ['name'],
        }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        name = str(args.get('name') or '').strip().lower().lstrip('/')
        if not name:
            return Assessment(risk=Risk.READ, summary='', invalid='name is required')
        skill = self.skills.get(name)
        if skill is None or not skill.model_invocable:
            names = ', '.join(sorted(s.name for s in self.skills.values() if s.model_invocable))
            return Assessment(risk=Risk.READ, summary='', invalid=f'there is no skill {name!r}. There are: {names}')
        file = str(args.get('file') or '').strip()
        return Assessment(risk=Risk.READ, summary=f'{name} ({file})' if file else name)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        skill = self.skills[str(args['name']).strip().lower().lstrip('/')]
        file = str(args.get('file') or '').strip()

        if file:
            if skill.folder is None:
                raise ToolError(f'{skill.name} is a single file; nothing comes with it')
            folder = skill.folder.resolve()
            target = (folder / file).resolve()
            if target != folder and folder not in target.parents:
                raise ToolError(f'{file}: outside the {skill.name} skill')
            if not target.is_file():
                listed = '\n'.join(f'  {f}' for f in files_of(skill)) or '  (none)'
                raise ToolError(f'{file}: not one of the files in {skill.name}. They are:\n{listed}')
            text = target.read_text(encoding='utf-8', errors='replace')
            body, cut = truncate(text, 60_000, keep='head')
            return Output(
                content=f'=== {skill.name}/{file} ===\n{body}',
                truncated=cut,
                display={'skill': skill.name, 'file': file},
            )

        content = render(skill, str(args.get('arguments') or ''))
        files = files_of(skill)
        if files:
            content += '\n\nFiles that come with this skill:\n' + '\n'.join(f'  {f}' for f in files)
        return Output(content=content, display={'skill': skill.name, 'source': skill.source})
