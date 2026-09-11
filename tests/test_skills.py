"""Skills, and the frontmatter both they and agent definitions are written in."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmirror.agent import frontmatter
from openmirror.agent.approval import Mode
from openmirror.agent.runtime import build_session
from openmirror.agent.skills import discover
from openmirror.agent.tools.base import ToolContext, ToolError
from openmirror.agent.tools.skill import SkillTool
from openmirror.protocol.agent import TurnStarted
from openmirror.providers.base import StreamDone, StreamText, TextBlock
from tests.test_agent import ScriptedProvider, turn


def skill(folder: Path, name: str, text: str) -> Path:
    (folder / name).mkdir(parents=True, exist_ok=True)
    path = folder / name / 'SKILL.md'
    path.write_text(text)
    return path


def ctx(root: Path) -> ToolContext:
    async def emit(text, stream):
        pass

    async def ask(question, options, multi):
        return ''

    return ToolContext(root=root, cwd=root, emit=emit, ask=ask, session_id='skills')


# -- frontmatter ------------------------------------------------------------


def test_frontmatter_reads_the_shapes_people_actually_write():
    fields, body = frontmatter.split(
        '---\n'
        'name: deploy\n'
        'description: >\n'
        '  Ship the build\n'
        '  to production.\n'
        'tools: [read_file, "git status"]\n'
        'also:\n'
        '  - one\n'
        '  - two\n'
        'disable-model-invocation: true\n'
        '# a comment\n'
        'hint: "quoted: with a colon"\n'
        '---\n'
        'Do the thing.\n'
    )
    assert fields['name'] == 'deploy'
    assert fields['description'] == 'Ship the build to production.'
    assert fields['tools'] == ['read_file', 'git status']
    assert fields['also'] == ['one', 'two']
    assert fields['disable-model-invocation'] is True
    assert fields['hint'] == 'quoted: with a colon'
    assert body == 'Do the thing.'


def test_a_file_with_no_frontmatter_is_all_body():
    assert frontmatter.split('# Title\n\nText.') == ({}, '# Title\n\nText.')


def test_an_unclosed_fence_is_a_horizontal_rule_not_a_header():
    text = '---\nThis is prose under a rule.\n'
    assert frontmatter.split(text) == ({}, text)


def test_a_list_of_names_is_the_same_however_it_is_written():
    assert frontmatter.as_list('read_file, grep') == ['read_file', 'grep']
    assert frontmatter.as_list(['read_file', 'grep']) == ['read_file', 'grep']
    assert frontmatter.as_list('') == []


# -- discovery --------------------------------------------------------------


def test_skills_come_from_every_place_and_the_nearest_wins(tmp_path: Path):
    home, root = tmp_path / 'home', tmp_path / 'project'
    skill(home / '.claude' / 'skills', 'review', '---\ndescription: my own review\n---\nPersonal.\n')
    skill(root / '.openmirror' / 'skills', 'review', '---\ndescription: this project\'s review\n---\nProject.\n')
    (root / '.claude' / 'commands').mkdir(parents=True)
    (root / '.claude' / 'commands' / 'ship.md').write_text('# Ship the build\n\nRun make ship.\n')

    found = discover(root, home=home)
    assert found['review'].description == "this project's review"
    assert found['review'].source == 'project'
    # A command file with no frontmatter is described by its first line.
    assert found['ship'].description == 'Ship the build'
    assert found['ship'].folder is None
    # And what ships with openmirror is there underneath all of it.
    assert {'init', 'review'} <= set(found)
    assert discover(root, home=home, bundled=False)['review'].source == 'project'


def test_no_home_means_no_personal_skills(tmp_path: Path):
    """What a test wants: a result that does not depend on whose machine it is."""
    found = discover(tmp_path)
    assert all(s.source == 'bundled' for s in found.values())


# -- the tool ---------------------------------------------------------------


async def test_the_tool_lists_names_and_loads_the_body_on_demand(tmp_path: Path):
    folder = tmp_path / '.openmirror' / 'skills'
    skill(folder, 'release', '---\ndescription: Cut a release.\n---\nBump the version to $ARGUMENTS.\n')
    (folder / 'release' / 'checklist.md').write_text('- tag it\n')
    tool = SkillTool(discover(tmp_path, bundled=False))

    assert '- release: Cut a release.' in tool.description
    assert 'Bump the version' not in tool.description, 'only names and descriptions up front'

    out = await tool.run({'name': 'release', 'arguments': '2.0'}, ctx(tmp_path))
    assert 'Bump the version to 2.0.' in out.content
    assert 'checklist.md' in out.content

    page = await tool.run({'name': 'release', 'file': 'checklist.md'}, ctx(tmp_path))
    assert '- tag it' in page.content


async def test_a_skill_reads_its_own_files_and_nothing_else(tmp_path: Path):
    skill(tmp_path / '.openmirror' / 'skills', 'safe', '---\ndescription: d\n---\nBody.\n')
    (tmp_path / 'secret.txt').write_text('no')
    tool = SkillTool(discover(tmp_path, bundled=False))
    with pytest.raises(ToolError, match='outside'):
        await tool.run({'name': 'safe', 'file': '../../../secret.txt'}, ctx(tmp_path))


def test_a_person_only_skill_is_not_offered_to_the_model(tmp_path: Path):
    skill(tmp_path / '.openmirror' / 'skills', 'deploy', '---\ndescription: d\ndisable-model-invocation: true\n---\nGo.\n')
    skill(tmp_path / '.openmirror' / 'skills', 'lint', '---\ndescription: lint it\n---\nLint.\n')
    tool = SkillTool(discover(tmp_path, bundled=False))
    assert 'deploy' not in tool.description
    assert tool.assess({'name': 'deploy'}, ctx(tmp_path)).invalid
    assert not tool.assess({'name': 'lint'}, ctx(tmp_path)).invalid

    session = build_session(root=tmp_path, provider=ScriptedProvider([[StreamDone()]]), model='x')
    assert 'deploy' in {c['name'] for c in session.commands()}, 'it still runs from /deploy'


# -- running one by name ----------------------------------------------------


async def test_typing_a_skill_runs_it(tmp_path: Path):
    skill(tmp_path / '.openmirror' / 'skills', 'explain', '---\ndescription: d\n---\nExplain $ARGUMENTS simply.\n')
    provider = ScriptedProvider([[StreamText(text='ok'), StreamDone()]])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.ASK)
    await session.start()
    events = await turn(session, '/explain the loop')

    # The person sees what they typed; the model gets the skill.
    assert [e.text for e in events if isinstance(e, TurnStarted)] == ['/explain the loop']
    sent = provider.seen[0].messages[0].content[0].text
    assert 'Explain the loop simply.' in sent
    assert 'ran /explain the loop' in sent


async def test_a_slash_that_names_nothing_is_just_text(tmp_path: Path):
    provider = ScriptedProvider([[StreamText(text='ok'), StreamDone()]])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.ASK)
    await session.start()
    await turn(session, '/etc/hosts has the wrong address in it')
    first = provider.seen[0].messages[0].content[0]
    assert isinstance(first, TextBlock) and first.text == '/etc/hosts has the wrong address in it'


def test_the_bundled_skills_are_well_formed():
    for found in discover(Path('/nonexistent')).values():
        assert found.description and found.body, found.name
        assert found.model_invocable, found.name
