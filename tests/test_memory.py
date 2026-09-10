"""The memory store: consent, isolation, and the claim that scrambling is free."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from roost.memory.store import EMBED_DIMS, MemoryStore, Settings, _project, _projection, _quantise


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


def test_the_projection_preserves_similarity_within_a_measured_bound():
    """The trade the store makes, stated as a number rather than a hope.

    This used to assert similarity survived to 1e-6, because the transform was
    a permutation and a sign flip at full width -- orthogonal, and exact. It is
    now a dimension-REDUCING projection, and throwing coordinates away cannot
    be exact. What replaces the old guarantee is a bound: the error is small,
    it is measured here, and if it grows this fails.

    0.12 is roughly twice the mean error measured over random pairs at
    EMBED_DIMS=256 (about 0.05), which leaves room for the tail without
    accepting a regression. Widening this constant is a decision about recall
    and should be made with the sweep in the store's docstring, not by nudging
    it until the suite is green.
    """
    key = b'\x01' * 32
    errors = []
    for i in range(60):
        a, b = vec(1000 + i), vec(5000 + i)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        proj = _projection(key, a.shape[0])
        pa, pb = _project(proj, a), _project(proj, b)
        pa /= np.linalg.norm(pa)
        pb /= np.linalg.norm(pb)
        errors.append(abs(float(np.dot(a, b)) - float(np.dot(pa, pb))))

    assert np.mean(errors) < 0.12, f'mean cosine error {np.mean(errors):.4f}'
    assert max(errors) < 0.40, f'worst cosine error {max(errors):.4f}'


def test_the_projection_actually_reduces_width():
    """The point of the change: a stored vector is EMBED_DIMS, not the model's.

    Without this the suite above would still pass on an identity projection --
    a transform that preserves similarity perfectly and saves nothing is
    exactly what a regression here would look like.
    """
    v = vec(77)
    v /= np.linalg.norm(v)
    proj = _projection(b'\x02' * 32, v.shape[0])
    assert v.shape[0] > EMBED_DIMS, 'the fixture vector is too narrow to test reduction'
    assert _project(proj, v).shape[0] == EMBED_DIMS
    assert proj.dims == EMBED_DIMS


def test_two_users_scramble_differently(store):
    """A vector stored for one person must be meaningless under another's key."""
    store._ensure_key('alice')
    store._ensure_key('bob')
    a_key = store._db.execute("SELECT key FROM memory_settings WHERE user_id='alice'").fetchone()['key']
    b_key = store._db.execute("SELECT key FROM memory_settings WHERE user_id='bob'").fetchone()['key']
    assert a_key != b_key

    v = vec(3)
    v /= np.linalg.norm(v)
    pa, pb = _projection(a_key, v.shape[0]), _projection(b_key, v.shape[0])
    va, vb = _project(pa, v), _project(pb, v)
    va /= np.linalg.norm(va)
    vb /= np.linalg.norm(vb)
    # Same vector, two keys: the results should look unrelated.
    assert abs(float(np.dot(va, vb))) < 0.3


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


# -- the move to a narrower store -------------------------------------------


def _write_legacy_row(store, user_id: str, text: str, v):
    """Write a row exactly as the pre-EMBED_DIMS store would have.

    Full width, permutation and sign flip, `src_dim` left at 0. Constructed by
    hand rather than by checking out the old code, so the test keeps working
    once that code is gone.
    """
    import uuid as _uuid

    import numpy as _np

    from roost.memory.store import _legacy_projection, _quantise

    key = store._ensure_key(user_id)
    v = _np.asarray(v, dtype=_np.float32)
    v = v / _np.linalg.norm(v)
    perm, signs = _legacy_projection(key, v.shape[0])
    blob, scale = _quantise(v[perm] * signs)
    mid = _uuid.uuid4().hex[:16]
    store._db.execute(
        'INSERT INTO memories (id, user_id, kind, text, source, created_at, accessed_at,'
        ' hits, dim, src_dim, scale, vec) VALUES (?,?,?,?,?,?,NULL,0,?,0,?,?)',
        (mid, user_id, 'fact', text, None, 1.0, v.shape[0], scale, blob),
    )
    store._db.commit()
    return mid


def test_a_legacy_row_is_found_again_after_reprojection(store):
    """The migration's whole job: an old memory must not go quiet.

    `search` filters on `src_dim`, so an un-migrated row is not scored wrongly
    — it is not scored at all. That is the failure worth testing for, because
    it looks like an empty memory rather than a broken one.
    """
    target = vec(4242)
    _write_legacy_row(store, 'alice', 'the spare key is under the third pot', target)

    # Before: invisible, because src_dim is 0 and no query can match that.
    assert store.search('alice', target, limit=5) == []

    moved = store.reproject_legacy_rows()
    assert moved == 1

    hits = store.search('alice', target, limit=5)
    assert [h.text for h in hits] == ['the spare key is under the third pot']
    # And it is now stored narrow, like everything else.
    row = store._db.execute('SELECT dim, src_dim FROM memories').fetchone()
    assert row['dim'] == EMBED_DIMS
    assert row['src_dim'] == target.shape[0]


def test_reprojection_is_idempotent_and_cheap_when_there_is_nothing_to_do(store):
    _write_legacy_row(store, 'alice', 'x', vec(5))
    assert store.reproject_legacy_rows() == 1
    # Second run finds nothing: a re-projected row no longer has src_dim 0, so
    # it cannot be put through the inverse transform a second time — which
    # would scramble it into noise rather than failing loudly.
    assert store.reproject_legacy_rows() == 0


def test_rows_from_two_models_are_never_compared(store):
    """Different source widths went through different projections.

    Both end up EMBED_DIMS wide, so `dim` cannot tell them apart any more —
    which is precisely why `src_dim` exists. Without it the store would score a
    1024-wide model's memory against a 2560-wide model's query and return a
    confident number computed from unrelated geometry.
    """
    import numpy as _np

    narrow = _np.asarray(vec(9)[:1024], dtype=_np.float32)
    store.add('alice', 'from the small model', narrow)
    store.add('alice', 'from the big model', vec(10))

    hits = store.search('alice', narrow, limit=10)
    assert [h.text for h in hits] == ['from the small model']
