"""Plan mode, and the to-do list that used to be called `plan`.

Plan mode's promise is that the person sees the whole plan before anything
moves, and that the way out is their answer. Both halves are tested from the
outside: what the model is shown, what is refused, and what an answer does.
"""

from __future__ import annotations

from pathlib import Path

from openmirror.agent.approval import ApprovalPolicy, Decision, Mode
from openmirror.agent.runtime import build_session
from openmirror.agent.tools.base import ToolContext
from openmirror.agent.tools.planning import APPROVE_EDIT, KEEP_PLANNING
from openmirror.agent.tools.todo import TodoTool
from openmirror.protocol.agent import (
    PolicyChanged,
    QuestionAsked,
    Risk,
    ToolCall,
    ToolCompleted,
    ToolDenied,
    ToolProposed,
)
from openmirror.providers.base import StreamDone, StreamText, StreamToolUse
from tests.test_agent import ScriptedProvider, turn


def call(call_id, name, **args):
    return [StreamToolUse(id=call_id, name=name, input=args), StreamDone(stop_reason='tool_use')]


def say(text):
    return [StreamText(text=text), StreamDone()]


async def test_planning_hides_the_editors_and_shows_the_way_out(tmp_path: Path):
    provider = ScriptedProvider([say('looking around'), say('now doing')])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.PLAN)
    await session.start()
    await turn(session, 'add a feature')

    shown = {t.name for t in provider.seen[0].tools}
    assert 'propose_plan' in shown and 'read_file' in shown and 'shell' in shown
    assert not shown & {'write_file', 'edit_file', 'multi_edit', 'apply_patch', 'notebook_edit'}
    assert '## Plan mode' in provider.seen[0].system

    # And the other way round, once it is not planning.
    await session.set_mode(Mode.ASK)
    await turn(session, 'carry on')
    shown = {t.name for t in provider.seen[1].tools}
    assert 'propose_plan' not in shown and 'write_file' in shown
    assert '## Plan mode' not in provider.seen[1].system


async def test_a_write_while_planning_is_refused_without_asking(tmp_path: Path):
    provider = ScriptedProvider([call('c1', 'write_file', path='x.txt', content='x'), say('ok')])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.PLAN)
    await session.start()
    events = await turn(session, 'write it')
    assert not (tmp_path / 'x.txt').exists()
    assert any(isinstance(e, ToolDenied) and 'planning' in e.reason for e in events)
    assert not any(isinstance(e, ToolProposed) and e.needs_approval for e in events)


def test_nothing_that_spends_or_types_a_secret_runs_while_planning():
    policy = ApprovalPolicy(mode=Mode.PLAN)
    for risk in (Risk.WRITE, Risk.EXECUTE, Risk.NETWORK, Risk.DESTRUCTIVE, Risk.PURCHASE, Risk.CREDENTIAL):
        assert policy.decide(ToolCall(id='c', name='x', risk=risk))[0] is Decision.DENY, risk
    assert policy.decide(ToolCall(id='c', name='read_file', risk=Risk.READ))[0] is Decision.ALLOW


async def test_an_approved_plan_changes_the_mode_and_everyone_is_told(tmp_path: Path):
    provider = ScriptedProvider([
        call('c1', 'propose_plan', plan='1. Create hello.txt saying hi.'),
        call('c2', 'write_file', path='hello.txt', content='hi\n'),
        say('Done.'),
    ])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.ASK)
    await session.start()
    await session.set_mode(Mode.PLAN)
    asked = []

    def answer(event):
        if isinstance(event, QuestionAsked):
            asked.append(event)
            session.answer(event.question_id, APPROVE_EDIT)

    events = await turn(session, 'make hello.txt', on=answer)

    assert asked and '1. Create hello.txt' in asked[0].question
    assert KEEP_PLANNING in asked[0].options
    changed = [e for e in events if isinstance(e, PolicyChanged)]
    assert changed and changed[-1].mode == 'auto_edit'
    # auto_edit runs the write without asking, which is what was chosen.
    assert (tmp_path / 'hello.txt').read_text() == 'hi\n'
    assert not any(isinstance(e, ToolProposed) and e.needs_approval for e in events)


async def test_a_plan_turned_down_stays_planning_and_carries_the_answer(tmp_path: Path):
    provider = ScriptedProvider([call('c1', 'propose_plan', plan='Rewrite everything.'), say('I will revise.')])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.PLAN)
    await session.start()

    def answer(event):
        if isinstance(event, QuestionAsked):
            session.answer(event.question_id, 'only touch the tests')

    events = await turn(session, 'fix it', on=answer)
    result = [e for e in events if isinstance(e, ToolCompleted) and e.result.name == 'propose_plan'][0].result
    assert 'only touch the tests' in result.content and 'still in plan mode' in result.content
    assert session.policy.mode is Mode.PLAN


async def test_approving_can_go_back_to_the_mode_before_planning(tmp_path: Path):
    provider = ScriptedProvider([call('c1', 'propose_plan', plan='Do it.'), say('ok')])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.TRUSTED)
    await session.start()
    await session.set_mode(Mode.PLAN)
    assert session.policy.previous is Mode.TRUSTED

    def answer(event):
        if isinstance(event, QuestionAsked):
            assert event.options[0] == 'Yes — carry on in trusted mode'
            session.answer(event.question_id, event.options[0])

    await turn(session, 'go', on=answer)
    assert session.policy.mode is Mode.TRUSTED


async def test_the_way_out_is_there_however_the_session_was_narrowed(tmp_path: Path):
    session = build_session(
        root=tmp_path, provider=ScriptedProvider([say('x')]), model='x', mode=Mode.PLAN, toolset=['files'],
    )
    assert 'propose_plan' in session.tools


# -- the to-do list ---------------------------------------------------------


def ctx(root: Path) -> ToolContext:
    async def emit(text, stream):
        pass

    async def ask(question, options, multi):
        return ''

    return ToolContext(root=root, cwd=root, emit=emit, ask=ask, session_id='todo')


async def test_the_to_do_list_keeps_the_latest_whole_list(tmp_path: Path):
    tool = TodoTool()
    args = {'items': [
        {'text': 'find the bug', 'state': 'done'},
        {'text': 'fix it', 'state': 'doing'},
        {'text': 'add a test'},
    ]}
    assessment = tool.assess(args, ctx(tmp_path))
    assert assessment.risk is Risk.READ
    assert assessment.summary == '1 of 3 done · fix it'
    out = await tool.run(args, ctx(tmp_path))
    assert out.content.splitlines() == ['[x] find the bug', '[~] fix it', '[ ] add a test']
    assert out.display['done'] == 1 and out.display['total'] == 3


def test_two_items_in_hand_at_once_is_refused(tmp_path: Path):
    both = {'items': [{'text': 'a', 'state': 'doing'}, {'text': 'b', 'state': 'doing'}]}
    assert 'exactly one' in TodoTool().assess(both, ctx(tmp_path)).invalid


async def test_another_harnesss_words_for_the_same_list_are_understood(tmp_path: Path):
    """A model trained elsewhere sends `todos`, `content` and `in_progress`."""
    tool = TodoTool()
    args = {'todos': [{'content': 'ship', 'status': 'in_progress'}, {'content': 'celebrate', 'status': 'pending'}]}
    assert not tool.assess(args, ctx(tmp_path)).invalid
    await tool.run(args, ctx(tmp_path))
    assert [(i.text, i.state) for i in tool.items] == [('ship', 'doing'), ('celebrate', 'todo')]


async def test_a_list_with_nothing_in_hand_gets_a_nudge(tmp_path: Path):
    out = await TodoTool().run({'items': [{'text': 'a'}, {'text': 'b'}]}, ctx(tmp_path))
    assert 'Nothing is marked doing' in out.content
