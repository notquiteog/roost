"""Making room: the conversation so far, replaced by a summary of it.

The summary is written by the model, so a scripted one stands in for it here.
What is tested is everything around it: what the summariser is sent, what the
conversation looks like afterwards, that the turn in hand is never cut, and
that nothing written from a summary can overwrite a file unread.
"""

from __future__ import annotations

from pathlib import Path

from openmirror.agent.approval import Mode
from openmirror.agent.runtime import build_session
from openmirror.protocol.agent import AgentError, ContextCompacted, ToolCompleted
from openmirror.providers.base import StreamDone, StreamText, StreamToolUse, TextBlock
from tests.test_agent import ScriptedProvider, turn


def say(text):
    return [StreamText(text=text), StreamDone()]


def texts(req) -> list[str]:
    return [b.text for m in req.messages for b in m.content if isinstance(b, TextBlock)]


async def test_compact_replaces_the_history_with_a_summary(tmp_path: Path):
    provider = ScriptedProvider([
        say('Hello back.'),
        say('SUMMARY: they said hello and asked to keep the greeting.'),
        say('Carrying on.'),
    ])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.ASK)
    await session.start()
    await turn(session, 'hello')
    events = await turn(session, '/compact keep the greeting')

    done = [e for e in events if isinstance(e, ContextCompacted)]
    assert done and done[0].reason == 'manual' and done[0].messages_before == 2
    assert done[0].summary.startswith('SUMMARY')

    # The summariser is sent the conversation and the focus, and no tools.
    request = provider.seen[1]
    assert not request.tools
    sent = texts(request)[0]
    assert 'Hello back.' in sent and 'keep the greeting' in sent

    # Afterwards the model starts from the summary, as a proper exchange.
    assert [m.role for m in session.messages] == ['user', 'assistant']
    await turn(session, 'what now?')
    assert [m.role for m in provider.seen[2].messages] == ['user', 'assistant', 'user']
    assert 'SUMMARY' in texts(provider.seen[2])[0]


async def test_automatic_compaction_never_cuts_the_turn_in_hand(tmp_path: Path):
    (tmp_path / 'a.txt').write_text('a\n')
    provider = ScriptedProvider([
        say('x' * 2000),                     # turn 1: big enough to cross the line
        say('SUMMARY: a long reply.'),       # the summary, asked for at turn 2's first step
        [StreamToolUse(id='c1', name='read_file', input={'path': 'a.txt'}), StreamDone(stop_reason='tool_use')],
        say('Read it.'),
    ])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.ASK, compact_at=200)
    await session.start()
    await turn(session, 'say a lot')
    events = await turn(session, 'now read a.txt')

    # Once, not at every step: the turn itself is large, and summarising the
    # summary on each step would be all this turn did.
    compacted = [e for e in events if isinstance(e, ContextCompacted)]
    assert len(compacted) == 1 and compacted[0].reason == 'automatic'
    # The turn's own message came through whole, after the summary pair.
    after = provider.seen[2]
    assert [m.role for m in after.messages] == ['user', 'assistant', 'user']
    assert texts(after)[-1] == 'now read a.txt'
    assert any(isinstance(e, ToolCompleted) and e.result.ok for e in events)


async def test_clear_forgets_the_conversation(tmp_path: Path):
    provider = ScriptedProvider([say('one'), say('fresh')])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.ASK)
    await session.start()
    await turn(session, 'first')
    events = await turn(session, '/clear')
    assert [e.reason for e in events if isinstance(e, ContextCompacted)] == ['cleared']
    assert session.messages == []
    await turn(session, 'second')
    assert texts(provider.seen[-1]) == ['second']


async def test_after_compaction_a_file_must_be_read_again_before_it_is_overwritten(tmp_path: Path):
    (tmp_path / 'notes.txt').write_text('important\n')
    provider = ScriptedProvider([
        [StreamToolUse(id='c1', name='read_file', input={'path': 'notes.txt'}), StreamDone(stop_reason='tool_use')],
        say('Read.'),
        say('SUMMARY: notes.txt was read.'),
        [StreamToolUse(id='c2', name='write_file', input={'path': 'notes.txt', 'content': 'gone\n'}),
         StreamDone(stop_reason='tool_use')],
        say('Refused, as it should be.'),
    ])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.UNRESTRICTED)
    await session.start()
    await turn(session, 'read the notes')
    await turn(session, '/compact')
    events = await turn(session, 'overwrite them')

    write = [e for e in events if isinstance(e, ToolCompleted)][0]
    assert not write.result.ok and 'read it before overwriting' in write.result.content
    assert (tmp_path / 'notes.txt').read_text() == 'important\n'


async def test_an_empty_summary_changes_nothing(tmp_path: Path):
    provider = ScriptedProvider([say('hi'), [StreamDone()]])
    session = build_session(root=tmp_path, provider=provider, model='x', mode=Mode.ASK)
    await session.start()
    await turn(session, 'hello')
    before = list(session.messages)
    events = await turn(session, '/compact')
    assert any(isinstance(e, AgentError) and 'empty summary' in e.message for e in events)
    assert session.messages == before


async def test_there_is_nothing_to_compact_in_a_new_session(tmp_path: Path):
    session = build_session(root=tmp_path, provider=ScriptedProvider([say('x')]), model='x')
    await session.start()
    events = await turn(session, '/compact')
    assert any(isinstance(e, AgentError) and 'nothing to compact' in e.message for e in events)
