"""Skills: instructions for a kind of job, loaded when that job turns up.

A skill is a folder with a SKILL.md in it — frontmatter naming it and saying
when it applies, a body saying how to do the job, and whatever else the job
needs beside it: a checklist, a template, a script. Only the names and the
one-line descriptions are in front of the model all the time. The body is read
when a request matches, and the files beside it only when the body points at
them. That ordering is the point: twenty skills cost twenty lines of tool
description, not twenty documents of context — which matters twice over on a
small model, where a long prompt is not only expensive but confusing.

Where they come from, a later one winning a name:

    openmirror/skills/                                  shipped with openmirror
    ~/.claude/skills/, ~/.openmirror/skills/            this person's, everywhere
    <root>/.claude/skills/, <root>/.openmirror/skills/  this project's

A flat `commands/<name>.md` beside any of those is a skill too. It is the
older shape of the same idea — a prompt you run by name — so `/name` in the
composer finds both.

Project skills are part of the project, which means somebody else may have
written them. They are treated like AGENTS.md: instructions about how to work
here, read by a model still bound by everything else it has been told. They
cannot widen what the approval policy allows, because nothing in them reaches
the policy.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from openmirror.agent import frontmatter

log = logging.getLogger(__name__)

BUNDLED = Path(__file__).resolve().parent.parent / 'skills'


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path
    # The skill's own folder, for the files that come with it. None for a
    # single-file command, which has nothing beside it.
    folder: Path | None
    source: str
    # False for a skill that should only run because a person asked for it by
    # name — `disable-model-invocation: true`. A deploy is the usual example:
    # a model deciding by itself that now is a good time is not the point.
    model_invocable: bool = True
    argument_hint: str = ''


def _normalise(name: str) -> str:
    return re.sub(r'[^a-z0-9_:-]+', '-', name.strip().lower()).strip('-')


def _load(path: Path, folder: Path | None, source: str) -> Skill | None:
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError as exc:
        log.warning('skill %s could not be read: %s', path, exc)
        return None
    fields, body = frontmatter.split(text)
    name = _normalise(str(fields.get('name') or (folder.name if folder else path.stem)))
    if not name or not body.strip():
        log.warning('skill %s has no name or no body; skipped', path)
        return None

    description = str(fields.get('description') or '').strip()
    if not description:
        # A command file often has no frontmatter at all. Its first line is
        # what it is for, near enough, and better than listing it blank.
        description = next((line.strip('# ').strip() for line in body.splitlines() if line.strip()), '')

    return Skill(
        name=name,
        description=' '.join(description.split())[:500],
        body=body.strip(),
        path=path,
        folder=folder,
        source=source,
        model_invocable=not frontmatter.as_bool(fields.get('disable-model-invocation')),
        argument_hint=str(fields.get('argument-hint') or '').strip(),
    )


def discover(root: Path, *, home: Path | None = None, bundled: bool = True) -> dict[str, Skill]:
    """Every skill this session can use, keyed by name."""
    places: list[tuple[Path, str]] = []
    if home is not None:
        places += [(home / '.claude', 'personal'), (home / '.openmirror', 'personal')]
    places += [(root / '.claude', 'project'), (root / '.openmirror', 'project')]

    found: dict[str, Skill] = {}
    if bundled and BUNDLED.is_dir():
        for skill_md in sorted(BUNDLED.glob('*/SKILL.md')):
            skill = _load(skill_md, skill_md.parent, 'bundled')
            if skill:
                found[skill.name] = skill

    for base, source in places:
        for skill_md in sorted((base / 'skills').glob('*/SKILL.md')):
            skill = _load(skill_md, skill_md.parent, source)
            if skill:
                found[skill.name] = skill
        for command in sorted((base / 'commands').glob('*.md')):
            skill = _load(command, None, source)
            if skill:
                found[skill.name] = skill
    return found


def render(skill: Skill, arguments: str = '') -> str:
    """The skill's instructions, ready for a model to follow."""
    body = skill.body
    arguments = arguments.strip()
    if '$ARGUMENTS' in body:
        body = body.replace('$ARGUMENTS', arguments)
    elif arguments:
        body = f'{body}\n\nArguments: {arguments}'

    head = f'The "{skill.name}" skill.'
    if skill.folder is not None and files_of(skill):
        head += " It comes with files, listed at the end; read one with the skill tool's `file` argument."
    return f'{head}\n\n{body}'


def files_of(skill: Skill, limit: int = 50) -> list[str]:
    """The files that come with a skill, relative to its folder."""
    if skill.folder is None:
        return []
    out = []
    for path in sorted(skill.folder.rglob('*')):
        if path.is_file() and path != skill.path and not any(p.startswith('.') for p in path.relative_to(skill.folder).parts):
            out.append(str(path.relative_to(skill.folder)))
            if len(out) >= limit:
                break
    return out
