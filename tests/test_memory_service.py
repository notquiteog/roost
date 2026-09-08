"""Memory as the agent meets it: consent, recall, and failing safely."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import numpy as np
import pytest

from roost.memory.service import MemoryService, _candidate_facts
from roost.memory.store import MemoryStore
from roost.providers.base import Modality, ProviderInfo
from roost.providers.registry import ProviderRegistry


class FakeEmbedder:
    """Deterministic embeddings: same text in, same vector out, and texts that
    share words come out near each other."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    async def embed(self, texts, model):
        self.calls += 1
        if self.fail:
            raise RuntimeError('embedding backend is down')
        out = []
        for text in texts:
            vec = np.zeros(64, dtype=np.float32)
            for word in text.lower().split():
                vec[hash(word) % 64] += 1.0
            out.append(vec.tolist())
        return out


def build(fail: bool = False):
    tmp = tempfile.TemporaryDirectory()
    store = MemoryStore(Path(tmp.name) / 'm.db')
    registry = ProviderRegistry()
    embedder = FakeEmbedder(fail=fail)
    registry.register(
        ProviderInfo(id='fake', label='Fake', modalities={Modality.EMBEDDING}, local=True),
        {Modality.EMBEDDING: embedder},
    )
    return MemoryService(store, registry, model='fake-embed'), embedder, tmp


# -- consent ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_nothing_is_written_before_opting_in():
    """Off must mean nothing is stored, not stored-and-ignored."""
    svc, embedder, tmp = build()
    with tmp:
        assert await svc.remember('alice', 'I always use tabs') is None
        assert svc.store.count('alice') == 0
        # Not even embedded: the text never left the process.
        assert embedder.calls == 0


@pytest.mark.asyncio
async def test_recall_says_why_it_found_nothing():
    svc, _, tmp = build()
    with tmp:
        result = await svc.recall('alice', 'tabs or spaces')
        assert result.memories == []
        assert 'off' in result.reason


@pytest.mark.asyncio
async def test_remembering_and_recalling_once_enabled():
    svc, _, tmp = build()
    with tmp:
        svc.set_settings('alice', enabled=True, auto_capture=False)
        await svc.remember('alice', 'deploys go out on Thursday afternoons')
        await svc.remember('alice', 'the staging database is called wren')

        result = await svc.recall('alice', 'when do deploys go out')
        assert result.memories
        assert 'Thursday' in result.memories[0].text


@pytest.mark.asyncio
async def test_context_is_empty_rather_than_an_empty_heading():
    """A model shown a blank 'what you remember' section apologises for not
    remembering, which is worse than never mentioning it."""
    svc, _, tmp = build()
    with tmp:
        svc.set_settings('alice', enabled=True, auto_capture=False)
        assert await svc.context_for('alice', 'anything') == ''


@pytest.mark.asyncio
async def test_context_respects_its_budget():
    svc, _, tmp = build()
    with tmp:
        svc.set_settings('alice', enabled=True, auto_capture=False)
        for i in range(10):
            await svc.remember('alice', f'fact number {i} about deploys ' + 'padding ' * 20)

        context = await svc.context_for('alice', 'deploys', budget=200)
        assert len(context) < 600      # the heading plus a line or two, not ten


# -- failing safely ---------------------------------------------------------


@pytest.mark.asyncio
async def test_recall_failure_does_not_break_the_turn():
    """An assistant that refuses to answer because it could not remember is
    worse than one that simply does not remember."""
    svc, _, tmp = build(fail=True)
    with tmp:
        svc.set_settings('alice', enabled=True, auto_capture=False)
        svc.store.add('alice', 'something', np.ones(64, dtype=np.float32))

        result = await svc.recall('alice', 'anything')
        assert result.memories == []
        assert 'unavailable' in result.reason
        assert await svc.context_for('alice', 'anything') == ''


@pytest.mark.asyncio
async def test_a_failed_write_is_not_an_exception():
    svc, _, tmp = build(fail=True)
    with tmp:
        svc.set_settings('alice', enabled=True, auto_capture=False)
        assert await svc.remember('alice', 'I prefer tabs') is None


# -- automatic capture ------------------------------------------------------


@pytest.mark.asyncio
async def test_capture_needs_its_own_switch():
    svc, _, tmp = build()
    with tmp:
        svc.set_settings('alice', enabled=True, auto_capture=False)
        assert await svc.capture('alice', 'I always run the tests before pushing anything') == []

        svc.set_settings('alice', enabled=True, auto_capture=True)
        assert await svc.capture('alice', 'I always run the tests before pushing anything')


def test_what_counts_as_a_fact():
    keep = _candidate_facts(
        'I always run the tests before pushing anything. '
        'My deploy script lives in bin/release for this project. '
        "Please don't touch the generated migration files."
    )
    assert len(keep) == 3


def test_transient_remarks_are_not_kept():
    """A fact that was only true this afternoon is a liability next week."""
    assert _candidate_facts('I am currently working on the parser refactor today') == []
    assert _candidate_facts('right now I prefer to skip the slow tests entirely') == []


def test_fragments_are_not_kept():
    assert _candidate_facts('I use it') == []          # too short to stand alone
    assert _candidate_facts('yes') == []


@pytest.mark.asyncio
async def test_capture_is_capped_per_message():
    """One conversation must not be able to fill memory with opinion."""
    svc, _, tmp = build()
    with tmp:
        svc.set_settings('alice', enabled=True, auto_capture=True)
        text = ' '.join(f'I always do thing number {i} in this project.' for i in range(10))
        assert len(await svc.capture('alice', text)) <= 3


# -- reaching the model -----------------------------------------------------


@pytest.mark.asyncio
async def test_a_remembered_fact_reaches_the_system_prompt():
    """The point of all of it. A memory that never reaches the model is a
    database, not a memory."""
    import tempfile as _tempfile

    from roost.agent.approval import Mode
    from roost.agent.runtime import build_session
    from roost.providers.base import StreamDone, StreamText
    from tests.test_agent import ScriptedProvider

    svc, _, tmp = build()
    with tmp, _tempfile.TemporaryDirectory() as root:
        svc.set_settings('alice', enabled=True, auto_capture=False)
        await svc.remember('alice', 'deploys go out on Thursday afternoons')

        provider = ScriptedProvider([[StreamText(text='ok'), StreamDone()]])
        session = build_session(
            root=root, provider=provider, model='x', mode=Mode.ASK, memory=svc, user_id='alice'
        )
        await session.start()
        session.submit('when do deploys go out')
        await asyncio.wait_for(session._turn, timeout=10)

        system = provider.seen[0].system
        assert 'Thursday' in system, 'the memory never reached the model'
        # Framed as background rather than instruction, so it is not obeyed as
        # though the person had just said it.
        assert 'background' in system.lower()

        # And the tools are offered only when memory is on for this person.
        assert 'remember' in session.tools and 'recall' in session.tools


@pytest.mark.asyncio
async def test_no_memory_tools_when_it_is_switched_off():
    """A model shown a `remember` tool it will always be refused for using
    wastes a step every turn."""
    import tempfile as _tempfile

    from roost.agent.approval import Mode
    from roost.agent.runtime import build_session
    from tests.test_agent import ScriptedProvider

    svc, _, tmp = build()
    with tmp, _tempfile.TemporaryDirectory() as root:
        session = build_session(
            root=root, provider=ScriptedProvider([[]]), model='x', mode=Mode.ASK,
            memory=svc, user_id='alice',
        )
        assert 'remember' not in session.tools
        assert 'recall' not in session.tools
