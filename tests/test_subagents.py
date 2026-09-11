"""Subagents: sessions that report to a session.

What has to hold is mostly about where things go rather than what the model
says: a subagent's steps land in the parent's log under the call that started
it, its approvals wait on the same person, it cannot outgrow the brief its
kind gives it, and several of them really do run at once.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
from pathlib import Path

from openmirror.agent import agents
from openmirror.agent.approval import Mode
from openmirror.agent.runtime import build_session
from openmirror.protocol.agent import TaskUpdated, ToolCompleted, ToolDenied, ToolProposed
from openmirror.providers.base import StreamDone, StreamText, StreamToolUse, TextBlock, ToolResultBlock
from tests.test_agent import ScriptedProvider, turn


class Routed:
    """A scripted provider with a separate script per agent.

    Which agent is asking is read off its first message: the main agent's is
    what the person typed, a subagent's is its brief, and each test puts a
    marker in the brief. An entry that is a coroutine function is awaited
    instead of streamed, which is how a test holds an agent mid-thought.
    """

    def __init__(self, scripts: dict[str, list[list]]) -> None:
        self.scripts = scripts
        self.calls = {key: 0 for key in scripts}
        self.seen: dict[str, list] = {key: [] for key in scripts}

    def _who(self, req) -> str:
        first = req.messages[0].content[0] if req.messages else None
        text = first.text if isinstance(first, TextBlock) else ''
        return next((key for key in self.scripts if key != 'main' and key in text), 'main')

    async def stream(self, req):
        who = self._who(req)
        self.seen[who].append(dataclasses.replace(req, messages=list(req.messages)))
        script = self.scripts[who]
        events = script[min(self.calls[who], len(script) - 1)]
        self.calls[who] += 1
        for event in events:
            if callable(event):
                await event()
            else:
                yield event

    async def models(self):
        return [{'id': 'routed'}]


def call(call_id, name, **args):
    return [StreamToolUse(id=call_id, name=name, input=args), StreamDone(stop_reason='tool_use')]


def say(text):
    return [StreamText(text=text), StreamDone()]


def results_in(req) -> list[ToolResultBlock]:
    return [b for m in req.messages for b in m.content if isinstance(b, ToolResultBlock)]


async def test_an_agent_reports_back_and_its_steps_carry_its_call(tmp_path: Path):
    (tmp_path / 'notes.txt').write_text('the answer is 42\n')
    provider = Routed({
        'main': [
            call('a1', 'agent', agent='explore', task='FIND-IT: what does notes.txt say?', title='read the notes'),
            say('It says 42.'),
        ],
        'FIND-IT': [call('r1', 'read_file', path='notes.txt'), say('notes.txt:1 says "the answer is 42".')],
    })
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.ASK)
    await session.start()
    events = await turn(session, 'what do the notes say?')

    # The subagent's read is an event in the session's own log, marked with
    # the call that started it — which is what lets a client nest it.
    read = [e for e in events if isinstance(e, ToolCompleted) and e.result.name == 'read_file']
    assert read and read[0].agent == 'a1'
    assert all(e.session_id == session.id for e in events)
    top = [e for e in events if isinstance(e, ToolCompleted) and e.result.name == 'agent']
    assert top and top[0].agent == '', "the delegation itself is the session's own call"

    # The report reached the main agent as the result of its call.
    assert 'the answer is 42' in results_in(provider.seen['main'][-1])[-1].content
    assert top[0].result.display['tool_calls'] == 1


async def test_an_agent_is_not_given_what_no_agent_gets(tmp_path: Path):
    provider = Routed({
        'main': [call('a1', 'agent', agent='general', task='LOOK: list the folder'), say('done')],
        'LOOK': [say('nothing here')],
    })
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.UNRESTRICTED)
    await session.start()
    await turn(session, 'look')

    names = {t.name for t in provider.seen['LOOK'][0].tools}
    # Its own agents would be a way to spend without limit; a question to the
    # person would come from something they never spoke to; the to-do list and
    # background work belong to the session.
    for never in ('agent', 'ask_user', 'todo', 'tasks', 'propose_plan'):
        assert never not in names, never
    assert {'read_file', 'write_file', 'shell'} <= names
    assert 'You are the general agent' in provider.seen['LOOK'][0].system


async def test_an_explore_agent_changes_nothing_whatever_the_mode(tmp_path: Path):
    provider = Routed({
        'main': [call('a1', 'agent', agent='explore', task='SNOOP: find things'), say('ok')],
        'SNOOP': [call('s1', 'shell', command='touch made-by-explore'), say('could not')],
    })
    # Unrestricted, so the only thing that can stop the command is the kind.
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.UNRESTRICTED)
    await session.start()
    events = await turn(session, 'look around')

    assert not (tmp_path / 'made-by-explore').exists()
    refused = [e for e in events if isinstance(e, ToolDenied) and e.agent == 'a1']
    assert refused and 'read-only' in refused[0].reason
    assert 'write_file' not in {t.name for t in provider.seen['SNOOP'][0].tools}


async def test_a_subagents_approval_waits_on_the_same_person(tmp_path: Path):
    provider = Routed({
        'main': [call('a1', 'agent', agent='general', task='WRITE-IT: create hello.txt'), say('done')],
        'WRITE-IT': [call('w1', 'write_file', path='hello.txt', content='hi\n'), say('created')],
    })
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.ASK)
    await session.start()
    asked = []

    def approve(event):
        if isinstance(event, ToolProposed) and event.needs_approval:
            # Waiting in the session's own table: a reattaching client, and
            # the reaper, both have to be able to see it.
            asked.append((event.agent, session.waiting_on))
            assert session.approve(event.call.id)

    await turn(session, 'make a file', on=approve)
    assert asked == [('a1', 'approval')]
    assert (tmp_path / 'hello.txt').read_text() == 'hi\n'


async def test_several_agents_in_one_response_run_at_once(tmp_path: Path):
    both = asyncio.Event()
    arrived = 0

    async def meet():
        # Each agent waits here for the other. Run one after the other, the
        # first would wait for ever — so finishing at all is the proof.
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            both.set()
        await asyncio.wait_for(both.wait(), timeout=5)

    provider = Routed({
        'main': [
            [
                StreamToolUse(id='a1', name='agent', input={'agent': 'general', 'task': 'ONE: first half'}),
                StreamToolUse(id='a2', name='agent', input={'agent': 'general', 'task': 'TWO: second half'}),
                StreamDone(stop_reason='tool_use'),
            ],
            say('both done'),
        ],
        'ONE': [[meet, StreamText(text='first half done'), StreamDone()]],
        'TWO': [[meet, StreamText(text='second half done'), StreamDone()]],
    })
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.ASK)
    await session.start()
    await turn(session, 'do both halves')

    assert both.is_set()
    contents = [r.content for r in results_in(provider.seen['main'][-1])]
    assert any('first half done' in c for c in contents)
    assert any('second half done' in c for c in contents)


async def test_a_background_agent_reports_at_the_next_request(tmp_path: Path):
    release = asyncio.Event()

    async def hold():
        await asyncio.wait_for(release.wait(), timeout=10)

    provider = Routed({
        'main': [
            call('a1', 'agent', agent='general', task='LATER: count the files', background=True),
            say('started it'),
            say('noted'),
        ],
        'LATER': [[hold, StreamText(text='there are 7 files'), StreamDone()]],
    })
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.ASK)
    await session.start()
    events = await turn(session, 'count them in the background')

    started = [e for e in events if isinstance(e, ToolCompleted) and e.result.name == 'agent'][0]
    assert started.result.display['task'] == 't1'
    assert session.tasks.get('t1').running, 'the turn ended without waiting for it'

    finished = asyncio.Event()
    # Taken now, not when the watcher first runs. A scripted model never
    # waits on anything, so once released the agent can finish in a single
    # step of the loop — before the watcher has run at all.
    since = session.seq

    async def watch():
        async for event in session.events(since=since):
            if isinstance(event, TaskUpdated) and event.task['status'] == 'done':
                finished.set()
                return

    watcher = asyncio.create_task(watch())
    release.set()
    await asyncio.wait_for(finished.wait(), timeout=10)
    await watcher

    await turn(session, 'anything back?')
    delivered = ' '.join(
        b.text for m in provider.seen['main'][-1].messages for b in m.content if isinstance(b, TextBlock)
    )
    assert 'Background agent t1' in delivered and 'there are 7 files' in delivered


async def test_closing_the_session_stops_a_background_agent(tmp_path: Path):
    never = asyncio.Event()

    async def hang():
        await never.wait()

    provider = Routed({
        'main': [call('a1', 'agent', agent='general', task='FOREVER: wait', background=True), say('ok')],
        'FOREVER': [[hang, StreamDone()]],
    })
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.ASK)
    await session.start()
    await turn(session, 'start it')
    task = session.tasks.get('t1')
    assert task.running
    await session.close()
    assert task.status == 'stopped' and task.stopped_by == 'session'


async def test_interrupting_stops_an_agent_in_the_foreground(tmp_path: Path):
    never = asyncio.Event()

    async def hang():
        await never.wait()

    provider = Routed({
        'main': [call('a1', 'agent', agent='general', task='STUCK: wait')],
        'STUCK': [[hang, StreamDone()]],
    })
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.ASK)
    await session.start()
    session.submit('go')
    for _ in range(200):
        if provider.seen['STUCK']:
            break
        await asyncio.sleep(0.01)
    assert session.interrupt()
    for _ in range(200):
        if not session.busy:
            break
        await asyncio.sleep(0.01)
    assert not session.busy


async def test_a_provider_that_reuses_call_ids_does_not_confuse_approvals(tmp_path: Path):
    """Ollama numbers calls per response, so `call_1` comes back every time.
    An approval is filed under the id; two under one would be one answer for
    two questions."""
    provider = ScriptedProvider([
        call('call_1', 'list_dir'),
        call('call_1', 'list_dir', path='.'),
        say('ok'),
    ])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.ASK)
    await session.start()
    events = await turn(session, 'list twice')
    ids = [e.call.id for e in events if isinstance(e, ToolProposed)]
    assert len(ids) == 2 and len(set(ids)) == 2


# -- definitions ------------------------------------------------------------


def test_an_agent_written_for_another_client_loads_with_its_tools_translated(tmp_path: Path):
    folder = tmp_path / '.claude' / 'agents'
    folder.mkdir(parents=True)
    (folder / 'reviewer.md').write_text(
        '---\nname: reviewer\ndescription: Reviews code for bugs.\n'
        'tools: Read, Grep, Bash(git diff:*)\nmodel: sonnet\n---\nYou review code.\n'
    )
    kind = agents.load(tmp_path)['reviewer']
    assert kind.tools == ('read_file', 'grep', 'shell')
    # A vendor's alias means "the session's own model" here: passed on, it
    # would be an error from every provider, that vendor's included.
    assert kind.model == ''
    assert kind.prompt == 'You review code.'
    assert kind.description == 'Reviews code for bugs.'


def test_the_nearest_definition_wins_and_read_only_is_honoured(tmp_path: Path):
    home = tmp_path / 'home'
    root = tmp_path / 'project'
    (home / '.openmirror' / 'agents').mkdir(parents=True)
    (root / '.openmirror' / 'agents').mkdir(parents=True)
    (home / '.openmirror' / 'agents' / 'auditor.md').write_text('---\ndescription: mine\n---\nPersonal.\n')
    (root / '.openmirror' / 'agents' / 'auditor.md').write_text(
        '---\ndescription: the project\'s\nmode: read_only\n---\nProject.\n'
    )
    kinds = agents.load(root, home=home)
    assert kinds['auditor'].description == "the project's"
    assert kinds['auditor'].read_only
    assert {'explore', 'general', 'research'} <= set(kinds)


def test_a_kind_with_nothing_to_work_with_is_not_offered():
    """A research agent in a session with no web tools would spend its whole
    budget discovering that it has nothing to search with."""
    offered = agents.offered(agents.load(Path(os.devnull).parent), {'read_file': object(), 'grep': object()})
    assert 'explore' in offered and 'general' in offered
    assert 'research' in offered, 'read_file and grep are research tools too'
    offered = agents.offered(
        {k.name: k for k in agents.BUILT_IN if k.name == 'research'}, {'shell': object()}
    )
    assert not offered
