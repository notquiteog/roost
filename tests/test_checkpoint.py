"""Undo for the agent's own edits.

The cases that matter are the awkward ones: a file the agent created (undo is
a delete), a file somebody edited by hand afterwards (undo must say so), and
an interrupted turn (the one most likely to need undoing).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from openmirror.agent.approval import Mode
from openmirror.agent.checkpoint import CheckpointStore
from openmirror.agent.runtime import build_session
from openmirror.providers.base import StreamDone, StreamText, StreamToolUse
from tests.test_agent import ScriptedProvider


@pytest.fixture()
def store(tmp_path):
    return CheckpointStore(tmp_path / '.checkpoints')


def test_an_edited_file_goes_back(store, tmp_path):
    target = tmp_path / 'a.txt'
    target.write_text('original\n')

    store.begin('t1', 'change it')
    store.record(target)
    target.write_text('changed\n')
    store.commit()

    report = store.restore('1')
    assert target.read_text() == 'original\n'
    assert str(target) in report.restored


def test_undoing_a_created_file_deletes_it(store, tmp_path):
    """The correct undo of "the agent made this" is that it is gone."""
    target = tmp_path / 'new.txt'

    store.begin('t1', 'create it')
    store.record(target)                  # records that it did not exist
    target.write_text('brand new\n')
    store.commit()

    report = store.restore('1')
    assert not target.exists()
    assert str(target) in report.deleted


def test_only_the_first_state_in_a_turn_is_kept(store, tmp_path):
    """A turn that edits one file three times must go back to before the turn,
    not to the second edit."""
    target = tmp_path / 'a.txt'
    target.write_text('v0\n')

    store.begin('t1', 'several edits')
    for version in ('v1', 'v2', 'v3'):
        store.record(target)
        target.write_text(version + '\n')
    store.commit()

    store.restore('1')
    assert target.read_text() == 'v0\n'


def test_a_turn_that_touched_nothing_is_not_listed(store):
    """An undo list full of no-ops is one nobody reads far enough down."""
    store.begin('t1', 'just a question')
    assert store.commit() is None
    assert store.describe() == []


def test_undoing_a_turn_undoes_the_ones_after_it(store, tmp_path):
    """Undoing turn one while leaving turn two produces a tree that never
    existed."""
    target = tmp_path / 'a.txt'
    target.write_text('v0\n')

    for n, content in enumerate(('v1', 'v2', 'v3'), start=1):
        store.begin(f't{n}', f'edit {n}')
        store.record(target)
        target.write_text(content + '\n')
        store.commit()

    assert len(store.checkpoints) == 3
    store.restore('1')
    assert target.read_text() == 'v0\n'
    assert store.checkpoints == []          # nothing left that could be undone


def test_a_hand_edit_since_is_reported_not_silently_clobbered(store, tmp_path):
    target = tmp_path / 'a.txt'
    target.write_text('original\n')

    store.begin('t1', 'edit')
    store.record(target)
    target.write_text('agent wrote this\n')
    store.commit()

    target.write_text('and then a person edited it\n')

    report = store.restore('1')
    assert target.read_text() == 'original\n'
    assert str(target) in report.changed_since


def test_the_agents_own_edit_is_not_reported_as_a_hand_edit(store, tmp_path):
    """The warning compares against what the agent *left*, not what it found.

    Comparing against the before-state makes it fire on every single rewind,
    since the agent changing the file is the reason there is a checkpoint at
    all — and a warning that always fires is one nobody reads.
    """
    target = tmp_path / 'a.txt'
    target.write_text('original\n')

    store.begin('t1', 'edit')
    store.record(target)
    target.write_text('agent wrote this\n')
    store.commit()

    report = store.restore('1')
    assert target.read_text() == 'original\n'
    assert report.restored == [str(target)]
    assert report.changed_since == []


def test_identical_content_is_stored_once(store, tmp_path):
    """Content addressing: thirty turns over one unchanged file is not thirty
    copies of it."""
    a, b = tmp_path / 'a.txt', tmp_path / 'b.txt'
    a.write_text('same bytes\n')
    b.write_text('same bytes\n')

    store.begin('t1', 'two files')
    store.record(a)
    store.record(b)
    store.commit()

    assert len(list(store.blobs.iterdir())) == 1


def test_a_very_large_file_is_skipped(store, tmp_path, monkeypatch):
    """A 200 MB artefact should not cost 200 MB per turn."""
    import openmirror.agent.checkpoint as mod

    monkeypatch.setattr(mod, 'MAX_SNAPSHOT_BYTES', 32)
    big = tmp_path / 'big.bin'
    big.write_bytes(b'x' * 1024)

    store.begin('t1', 'big')
    store.record(big)
    assert store.commit() is None


def test_restoring_an_unknown_checkpoint_raises(store):
    with pytest.raises(KeyError):
        store.restore('nope')


# -- through a real session -------------------------------------------------


@pytest.mark.asyncio
async def test_a_turn_is_undoable_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / 'notes.txt').write_text('keep me\n')
        store = CheckpointStore(root / '.cp')

        session = build_session(
            root=root,
            provider=ScriptedProvider([
                [
                    StreamToolUse(id='c1', name='write_file',
                                  input={'path': 'notes.txt', 'content': 'clobbered\n'}),
                    StreamDone(stop_reason='tool_use'),
                ],
                [StreamText(text='done'), StreamDone()],
            ]),
            model='x',
            mode=Mode.UNRESTRICTED,
        )
        session.checkpoints = store
        await session.start()

        # write_file refuses a file it has not read, so seed the journal.
        from openmirror.agent.tools.files import journal
        journal.note_read(session.id, root / 'notes.txt')

        session.submit('overwrite the notes')
        await asyncio.wait_for(session._turn, timeout=10)

        assert (root / 'notes.txt').read_text() == 'clobbered\n'
        assert len(store.checkpoints) == 1

        store.restore('1')
        assert (root / 'notes.txt').read_text() == 'keep me\n'


@pytest.mark.asyncio
async def test_an_interrupted_turn_is_still_undoable():
    """The turn cut off halfway is the one most likely to have left the tree
    somewhere nobody wanted."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / 'a.txt').write_text('before\n')
        store = CheckpointStore(root / '.cp')

        session = build_session(
            root=root,
            provider=ScriptedProvider([
                [
                    StreamToolUse(id='c1', name='write_file',
                                  input={'path': 'a.txt', 'content': 'after\n'}),
                    StreamToolUse(id='c2', name='shell', input={'command': 'sleep 30'}),
                    StreamDone(stop_reason='tool_use'),
                ],
            ]),
            model='x',
            mode=Mode.UNRESTRICTED,
        )
        session.checkpoints = store
        await session.start()

        from openmirror.agent.tools.files import journal
        journal.note_read(session.id, root / 'a.txt')

        session.submit('write then hang')
        await asyncio.sleep(1.0)
        session.interrupt()
        with pytest.raises(asyncio.CancelledError):
            await session._turn

        assert len(store.checkpoints) == 1
        store.restore('1')
        assert (root / 'a.txt').read_text() == 'before\n'
