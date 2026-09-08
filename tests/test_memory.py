"""The memory store: consent, isolation, and the claim that scrambling is free."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from roost.memory.store import MemoryStore, Settings, _projection, _quantise


@pytest.fixture()
def store():
    with tempfile.TemporaryDirectory() as tmp:
        s = MemoryStore(Path(tmp) / 'memory.db')
        yield s
        s.close()


def vec(seed: int, dim: int = 384) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=dim).astype(np.float32)


# -- consent ----------------------------------------------------------------


def test_memory_is_off_until_asked_for(store):
    assert store.settings('alice') == Settings(enabled=False, auto_capture=False)


def test_auto_capture_cannot_outlive_the_master_switch(store):
    """Turning memory off must turn automatic capture off with it, not leave
    it armed for whenever memory is next enabled."""
    store.set_settings('alice', enabled=True, auto_capture=True)
    assert store.settings('alice').auto_capture

    after = store.set_settings('alice', enabled=False, auto_capture=True)
    assert not after.enabled and not after.auto_capture


# -- the projection ---------------------------------------------------------


def test_scrambling_preserves_similarity_exactly():
    """The whole justification for the key: a permutation and a sign flip are
    orthogonal, so they cost nothing in accuracy."""
    a, b = vec(1), vec(2)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    perm, signs = _projection(b'\x01' * 32, a.shape[0])
    pa, pb = a[perm] * signs, b[perm] * signs

    assert np.dot(a, b) == pytest.approx(float(np.dot(pa, pb)), abs=1e-6)


def test_two_users_scramble_differently(store):
    """A vector stored for one person must be meaningless under another's key."""
    store._ensure_key('alice')
    store._ensure_key('bob')
    a_key = store._db.execute("SELECT key FROM memory_settings WHERE user_id='alice'").fetchone()['key']
    b_key = store._db.execute("SELECT key FROM memory_settings WHERE user_id='bob'").fetchone()['key']
    assert a_key != b_key

    v = vec(3)
    v /= np.linalg.norm(v)
    pa, sa = _projection(a_key, v.shape[0])
    pb, sb = _projection(b_key, v.shape[0])
    # Same vector, two keys: the results should look unrelated.
    assert abs(float(np.dot(v[pa] * sa, v[pb] * sb))) < 0.3


def test_quantisation_error_is_negligible():
    v = vec(4)
    v /= np.linalg.norm(v)
    blob, scale = _quantise(v)
    back = np.frombuffer(blob, dtype=np.int8).astype(np.float32) * scale
    assert float(np.dot(v, back)) == pytest.approx(1.0, abs=0.002)


def test_a_zero_vector_does_not_divide_by_zero():
    blob, scale = _quantise(np.zeros(16, dtype=np.float32))
    assert scale == 1.0
    assert not np.frombuffer(blob, dtype=np.int8).any()


# -- storing and finding ----------------------------------------------------


def test_finds_the_nearest_memory(store):
    target = vec(10)
    store.add('alice', 'the deploy key lives in 1Password', target)
    store.add('alice', 'the cat is called Widget', vec(11))
    store.add('alice', 'staging redeploys on push to main', vec(12))

    # Query with a slightly perturbed version of the first vector.
    query = target + np.random.default_rng(99).normal(scale=0.1, size=target.shape).astype(np.float32)
    hits = store.search('alice', query, limit=1)

    assert hits and hits[0].text == 'the deploy key lives in 1Password'
    assert hits[0].score > 0.8


def test_one_user_cannot_see_another(store):
    shared = vec(20)
    store.add('alice', "alice's secret", shared)
    store.add('bob', "bob's secret", shared)

    # The identical vector, so only the scoping keeps them apart.
    for user, expected in (('alice', "alice's secret"), ('bob', "bob's secret")):
        hits = store.search(user, shared, limit=10)
        assert [h.text for h in hits] == [expected]


def test_weak_matches_are_left_out(store):
    store.add('alice', 'something unrelated', vec(30))
    assert store.search('alice', vec(31), limit=5, min_score=0.5) == []


def test_a_different_embedding_model_is_skipped_not_coerced(store):
    """Mixing dimensions would produce confident nonsense."""
    store.add('alice', 'from the old model', vec(40, dim=128))
    store.add('alice', 'from the new model', vec(41, dim=384))

    hits = store.search('alice', vec(41, dim=384), limit=5, min_score=0.0)
    assert [h.text for h in hits] == ['from the new model']


def test_search_records_that_a_memory_was_used(store):
    store.add('alice', 'remembered', vec(50))
    assert store.list('alice')[0].hits == 0
    store.search('alice', vec(50), limit=1)
    assert store.list('alice')[0].hits == 1


# -- forgetting -------------------------------------------------------------


def test_delete_is_scoped_to_the_owner(store):
    m = store.add('alice', 'private', vec(60))
    assert not store.delete('bob', m.id), 'one user deleted another user\'s memory'
    assert store.delete('alice', m.id)
    assert store.count('alice') == 0


def test_wipe_takes_the_key_with_it(store):
    store.set_settings('alice', enabled=True, auto_capture=True)
    before = store._db.execute("SELECT key FROM memory_settings WHERE user_id='alice'").fetchone()['key']
    store.add('alice', 'one', vec(70))
    store.add('alice', 'two', vec(71))

    assert store.wipe('alice') == 2
    assert store.count('alice') == 0
    # Consent is gone too: a wipe that left them opted in would start
    # refilling immediately.
    assert store.settings('alice') == Settings(enabled=False, auto_capture=False)

    store._ensure_key('alice')
    after = store._db.execute("SELECT key FROM memory_settings WHERE user_id='alice'").fetchone()['key']
    assert after != before, 'a restored backup would still be searchable'
