"""Where memories live.

Three decisions shape this file.

**SQLite and a full scan — and this is the decision most likely to need
revisiting.** openmirror is a client on one person's machine: there are no accounts
and there is not going to be a multi-tenant install, so scale here is never
"more users". It is one user, for years, and the goal is to remember
*everything* about them — which is a growth curve, not a ceiling.

At the few thousand memories this was written for, an approximate index buys
nothing measurable and costs a service to run, a schema to keep in sync, and a
second thing that can be down. That is still true at that size. It stops being
true earlier than the original comment implied, and the intended product walks
straight through it.

The conclusion is right and the arithmetic that used to be here was not, so it
is worth replacing rather than deleting. It said a quantised vector was 768
bytes and that a dot product over ten thousand of them was "a couple of
milliseconds". Measured, at the 2560 dimensions of the floor embedder
(`qwen3-embedding:4b`):

    1,000 memories     4.9 ms
   10,000 memories    52.9 ms
   50,000 memories   283.1 ms

Two things were wrong. 768 bytes assumed a 768-dimension model; the floor is
2560, so a vector is 2560 bytes. And "a couple of milliseconds" describes one
matmul over a contiguous block, while `search` below reconstructs each vector
in a Python loop — about twenty-five times the cost at ten thousand.

**So the threshold is far lower than "a million".** It is comfortable to a few
thousand, noticeable at ten thousand, and a real pause at fifty. An agent that
remembers what it read, what it ran, what was said and what it saw reaches
fifty thousand in months rather than years, so this is a live constraint on the
product rather than a distant one.

**Two fixes, and they compose — the width first, then the scan. The width is
now built; the scan is not.**

*Width.* Vectors are stored at the model's full 2560 dimensions. Tern solves
the same problem with a keyed projection down to a fixed narrow width before
storage, which composes with the rotation below because both are orthogonal.
At 256 dimensions fifty thousand vectors are 13 MB rather than 128, and a
million are 256 MB rather than 2.5 GB. This is the half that decides whether
the store fits in memory at all.

*Scan.* Narrower rows do not fix the Python loop, only shrink it. Vectorising
in numpy is not the answer either — scoring from a resident float32 matrix is
fast but costs 512 MB at fifty thousand rows, which defeats the int8
quantisation below. What is left is a scan in C over int8, which is what
`sqlite-vec` is, in process, with no service and nothing extra to be down.

Measured after the projection landed, at 2560-wide source vectors:

              search      database
    5,000     22.2 ms      1.9 MB
   20,000     91.6 ms      7.7 MB

Against 52.9 ms at ten thousand and 283 ms at fifty before it, with a database
about ten times the size. The projection costs 0.24 ms per memory written,
which is nothing beside the embedding call that produced the vector.

**So the scan is still worth doing, and the reason is now clear.** Search did
not fall by the ten times the vectors shrank — because what is left is not
arithmetic. It is SQLite handing back rows and Python building an object per
row, which does not care how wide the vector is. That is the same 80%-of-cost
finding Tern measured on its own scan, and it is what a C implementation
removes and a narrower vector cannot.

**Vectors are quantised to int8.** A quarter of the size for a similarity
error far below the noise floor of the embedding itself. Each vector carries
its own scale, so a quiet vector is not crushed by a loud one.

**Vectors are scrambled with a per-user key.** A permutation and a sign flip:
both orthogonal, so dot products — and therefore cosine similarity — are
preserved *exactly*, with no quality cost whatsoever. What it buys is that a
stolen database is not a pile of embeddings anyone can run through a public
model to recover the text, and vectors from two different users cannot be
meaningfully compared. The key never leaves this table, and because queries
are projected the same way, nothing ever needs unprojecting.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
  id          TEXT PRIMARY KEY,
  user_id     TEXT NOT NULL,
  kind        TEXT NOT NULL,
  -- What the memory is ABOUT, when that is something nameable: a project, a
  -- task, a person. NULL means it is about the user themselves, which is the
  -- commonest case and the reason this is nullable rather than a sentinel.
  --
  -- Separate from `kind` because they answer different questions. `kind` is
  -- what sort of thing this is (a fact, a preference, something that happened);
  -- `subject` is what it attaches to. "Recall what you know about the openmirror
  -- project" needs the second, and a kind alone cannot express it.
  subject     TEXT,
  text        TEXT NOT NULL,
  source      TEXT,
  created_at  REAL NOT NULL,
  accessed_at REAL,
  hits        INTEGER NOT NULL DEFAULT 0,
  dim         INTEGER NOT NULL,
  -- The width the MODEL produced, as opposed to `dim`, which is the width this
  -- store keeps. Two rows are comparable only if both agree: the projection is
  -- derived from the source width, so vectors from a 1024-wide model and a
  -- 2560-wide one went through different maps and scoring one against the
  -- other returns confident nonsense rather than a low score.
  src_dim     INTEGER NOT NULL DEFAULT 0,
  -- Which embedder produced this vector, as the fingerprint the service hands
  -- down: a model name, plus the requested width when the model is
  -- truncatable. `src_dim` cannot do this job -- it separates a 1024-wide
  -- model from a 2560-wide one, but two different models of the SAME width are
  -- indistinguishable by it, and scoring one against the other returns a
  -- confident number computed from unrelated geometry. That is the case this
  -- column exists for.
  --
  -- A row with `src_dim = -1` has had its vector discarded, because its
  -- embedder changed under it, and is waiting to be embedded again from
  -- `text`. Distinct from `src_dim = 0`, which means a pre-projection row.
  embedder    TEXT NOT NULL DEFAULT '',
  scale       REAL NOT NULL,
  vec         BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, kind);
CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(user_id, subject);

-- Which embedder this store's vectors are under. One row, because there is one
-- index: when this stops matching what the service is actually calling, every
-- vector in the store is being scored against a geometry that no longer
-- exists, so the change has to be noticed at open rather than discovered in
-- bad recall.
CREATE TABLE IF NOT EXISTS memory_index_state (
  id       INTEGER PRIMARY KEY CHECK (id = 1),
  embedder TEXT NOT NULL,
  dims     INTEGER NOT NULL,
  set_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_settings (
  user_id      TEXT PRIMARY KEY,
  enabled      INTEGER NOT NULL DEFAULT 0,
  auto_capture INTEGER NOT NULL DEFAULT 0,
  key          BLOB NOT NULL
);
"""


@dataclass(slots=True)
class Memory:
    id: str
    user_id: str
    kind: str
    #: What this is about -- a project, a task -- or None for the user.
    subject: str | None
    text: str
    source: str | None
    created_at: float
    hits: int = 0
    score: float = 0.0


@dataclass(slots=True)
class Settings:
    """What one person has agreed to.

    Both default to off, and `enabled` off means nothing is written at all —
    not written-but-unused. A toggle that only stops the reading would leave a
    record of someone who believed they had opted out.
    """

    enabled: bool = False
    auto_capture: bool = False


#: What the store knows about the state of its own index, for a UI or an
#: operator who wants to see it rather than infer it from recall going quiet.
@dataclass(slots=True)
class IndexState:
    #: The embedder the store's vectors are under, '' if never recorded.
    embedder: str
    #: The width those vectors are stored at -- EMBED_DIMS, unless a store was
    #: written by a build with a different one.
    dims: int
    #: Memories in total, memories that can currently be searched, and
    #: memories whose vector was discarded and not yet re-made.
    total: int
    searchable: int
    pending: int
    #: True when sqlite-vec answers searches, False when the Python scan does.
    accelerated: bool
    set_at: float = 0.0


@dataclass(slots=True)
class IndexReset:
    """What a change of embedder cost."""

    previous: str
    embedder: str
    #: Vectors discarded. The texts are all still there: they are re-embedded,
    #: not lost.
    discarded: int


#: The kinds a memory can be, as the rest of openmirror uses them.
#:
#: Not enforced -- the column is free text and a caller may invent one -- but
#: written down because recall is only as good as the vocabulary it can filter
#: on, and four sessions inventing four spellings of "preference" is how that
#: vocabulary stops being useful.
KINDS = (
    'fact',        # something durable about the person or their setup
    'preference',  # how they like things done
    'project',     # about a piece of ongoing work
    'task',        # about one job, usually short-lived
    'episode',     # something that happened, captured automatically
)

#: Every stored vector is this wide, whatever the model produced.
#:
#: The store used to keep the model's own width -- 2560 for the floor embedder
#: -- which made a row 2560 bytes and a million memories 2.5 GB. At 256 the
#: same million is 256 MB, which is the difference between a store that loads
#: and one that does not.
EMBED_DIMS = 256

#: Sign-flip rounds before the transform. Three is what Tern uses and what the
#: literature suggests: one round leaves structure that the Hadamard transform
#: does not spread, and past three the measured error stops moving.
_ROUNDS = 3


@dataclass(slots=True)
class _Projection:
    """One user's keyed map from a model's width down to EMBED_DIMS."""

    padded: int
    dims: int
    signs: list[np.ndarray]
    pick: np.ndarray
    scale: float


def _projection(key: bytes, src_dim: int) -> _Projection:
    """Derive this user's dimension-reducing projection from their key.

    A subsampled randomised Hadamard transform: pad to a power of two, flip
    signs and run a Walsh-Hadamard transform a few times, then keep a keyed
    subset of the coordinates. The same construction Tern uses, for the same
    two reasons.

    **It reduces dimensions, so unlike the permutation it replaces it is not
    exact.** The old transform was a permutation and a sign flip at full width:
    orthogonal, and cosine survived to 1e-6. This one throws coordinates away,
    and what it preserves is preserved only in expectation -- see
    `tests/test_memory.py` for the measured error. That is the trade being
    made: a bounded similarity error for a tenth of the storage and a tenth of
    the scan.

    **The keying still does its job.** Sign patterns and the surviving axes are
    both derived from the user's key, so a stolen database is not a pile of
    embeddings anyone can run through a public model -- and now it is not even
    the right width to try.
    """
    dims = min(EMBED_DIMS, src_dim)
    padded = 1
    while padded < max(src_dim, dims):
        padded <<= 1

    seed = int.from_bytes(key[:8], 'little')
    rng = np.random.default_rng(seed)
    signs = [
        rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=padded)
        for _ in range(_ROUNDS)
    ]
    # Keyed choice of which axes survive, rather than the leading block:
    # somebody who learned the sign patterns still would not know which
    # coordinates were kept.
    pick = rng.permutation(padded)[:dims]
    # Undoes the shrinkage from keeping only `dims` of `padded` coordinates, so
    # projected vectors have roughly the length the originals did.
    return _Projection(padded=padded, dims=dims, signs=signs, pick=pick,
                       scale=float(np.sqrt(padded / dims)))


def _walsh_hadamard(v: np.ndarray) -> np.ndarray:
    """In-place-ish Walsh-Hadamard transform, normalised to be orthogonal.

    `len(v)` is a power of two by construction. Written with numpy slicing
    rather than a Python loop over coordinates: at 4096 padded width the loop
    version is the dominant cost of storing a memory.
    """
    n = v.shape[0]
    out = v.astype(np.float32, copy=True)
    step = 1
    while step < n:
        # Butterfly: reshape so each row holds one (left, right) pair block,
        # then add and subtract whole blocks at once.
        blocks = out.reshape(-1, 2 * step)
        left = blocks[:, :step].copy()
        right = blocks[:, step:].copy()
        blocks[:, :step] = left + right
        blocks[:, step:] = left - right
        step <<= 1
    return out / np.sqrt(n)


def _project(proj: _Projection, vec: np.ndarray) -> np.ndarray:
    """A model vector, as the EMBED_DIMS floats that get quantised."""
    work = np.zeros(proj.padded, dtype=np.float32)
    n = min(vec.shape[0], proj.padded)
    work[:n] = vec[:n]
    for s in proj.signs:
        work = _walsh_hadamard(work * s)
    return work[proj.pick] * proj.scale


def _legacy_projection(key: bytes, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """The permutation and sign flip rows were stored under before EMBED_DIMS.

    Kept solely so `reproject_legacy_rows` can undo it. Both halves are
    orthogonal and the permutation is invertible, so an old row can be turned
    back into (an int8-quantised approximation of) the model vector it came
    from and pushed through the new projection -- no re-embedding, no provider
    call, no network.
    """
    seed = int.from_bytes(key[:8], 'little')
    rng = np.random.default_rng(seed)
    perm = rng.permutation(dim)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dim)
    return perm, signs


def _pad_to_index_width(blob: bytes) -> bytes:
    """Right-pad a stored vector with zeros to exactly EMBED_DIMS bytes.

    The vec0 table is declared at one width, but `_projection` keeps
    `min(EMBED_DIMS, src_dim)` — so a model narrower than EMBED_DIMS produces a
    narrower vector and would be refused.

    Padding with zeros is **exact**, not an approximation: appending zeros to
    both the stored vector and the query changes neither their dot product nor
    either norm, so the cosine is identical to the unpadded one. The scan path
    and the index therefore agree, which a test asserts.

    Only the index needs this. `memories.vec` keeps the true width, because
    that is the column `reproject_legacy_rows` and the scan read.
    """
    if len(blob) >= EMBED_DIMS:
        return blob[:EMBED_DIMS]
    return blob + b'\x00' * (EMBED_DIMS - len(blob))


def _quantise(vec: np.ndarray) -> tuple[bytes, float]:
    """int8 with a per-vector scale."""
    peak = float(np.abs(vec).max())
    if peak == 0.0:
        return np.zeros(vec.shape, dtype=np.int8).tobytes(), 1.0
    scale = peak / 127.0
    return np.round(vec / scale).astype(np.int8).tobytes(), scale


class MemoryStore:
    def __init__(self, path: str | Path, embedder: str = '') -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        # WAL so a read during a write does not block; this is read far more
        # than it is written.
        self._db.execute('PRAGMA journal_mode=WAL')
        self._db.executescript(SCHEMA)
        # `CREATE TABLE IF NOT EXISTS` does nothing to a table that already
        # exists, so a store written before EMBED_DIMS has no `src_dim` column
        # and every query naming it would raise. Added here rather than in
        # SCHEMA for that reason.
        columns = {r['name'] for r in self._db.execute('PRAGMA table_info(memories)')}
        if 'src_dim' not in columns:
            self._db.execute('ALTER TABLE memories ADD COLUMN src_dim INTEGER NOT NULL DEFAULT 0')
        if 'subject' not in columns:
            self._db.execute('ALTER TABLE memories ADD COLUMN subject TEXT')
            self._db.execute(
                'CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(user_id, subject)')
        if 'embedder' not in columns:
            self._db.execute("ALTER TABLE memories ADD COLUMN embedder TEXT NOT NULL DEFAULT ''")
        self._db.commit()
        # Rows from before the projection are unreachable until this runs --
        # `search` filters on `src_dim`, so they would go quiet rather than
        # wrong, and somebody's memory would empty out with nothing to say why.
        self.reproject_legacy_rows()
        self._vec = self._open_vector_index()
        #: What the last open did about the embedder, for whoever wants to
        #: report it. None when nothing changed.
        self.reset: IndexReset | None = self.use_embedder(embedder) if embedder else None

    def _open_vector_index(self) -> bool:
        """Load sqlite-vec and make sure its table matches the store.

        Optional on purpose. `sqlite-vec` is a loadable extension, and
        `enable_load_extension` is a COMPILE-TIME Python option that not every
        interpreter is built with -- so an install that cannot load it has to
        keep working rather than fail. When it is absent `search` falls back to
        the Python scan, which is correct and slower; the two paths are asserted
        to agree in the tests.

        Measured at fifty thousand memories: 12.9 ms through the index against
        107.6 ms for the scan's arithmetic alone, before the scan also pays for
        SQLite handing back every row. The index returns k rows and no more,
        which is the larger half of the win.
        """
        try:
            import sqlite_vec
        except ImportError:
            log.debug('sqlite-vec is not installed; memory search will scan')
            return False
        if not hasattr(self._db, 'enable_load_extension'):
            log.info('this Python cannot load SQLite extensions; memory search will scan')
            return False
        try:
            self._db.enable_load_extension(True)
            sqlite_vec.load(self._db)
            self._db.enable_load_extension(False)
        except Exception as exc:  # noqa: BLE001
            log.info('sqlite-vec would not load (%s); memory search will scan', exc)
            return False

        # A table built for a different width cannot hold today's vectors, and
        # sqlite-vec will not tell us politely -- every insert would fail. The
        # declared width is in the DDL, so it is read back and the table
        # rebuilt if it disagrees. Cheap: the vectors are all still in
        # `memories`, so a rebuild is a backfill.
        row = self._db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memories_vec'"
        ).fetchone()
        if row is not None:
            found = re.search(r'int8\[(\d+)\]', row['sql'] or '')
            if not found or int(found.group(1)) != EMBED_DIMS:
                log.info('vector index was built for a different width; rebuilding')
                self._db.execute('DROP TABLE memories_vec')
                row = None
        if row is None:
            # `subject` cannot be NULL in a vec0 TEXT metadata column, so a
            # memory about the user rather than about a named thing carries ''.
            # The store's own column stays NULL, which is the honest
            # representation; this is a sentinel for one index's limitation and
            # is translated at both ends.
            self._db.execute(f'''
                CREATE VIRTUAL TABLE memories_vec USING vec0(
                  memory_id TEXT PRIMARY KEY,
                  user_id TEXT partition key,
                  src_dim INTEGER,
                  kind TEXT,
                  subject TEXT,
                  v int8[{EMBED_DIMS}] distance_metric=cosine
                )''')
        self._db.commit()
        self._backfill_vector_index()
        return True

    def _backfill_vector_index(self) -> None:
        """Put every memory that is not in the index into it.

        Runs after a rebuild, after the extension appears on a store that was
        written without it, and after re-projection. A store already in step
        does one anti-join and stops.
        """
        rows = list(self._db.execute(
            'SELECT m.id, m.user_id, m.kind, m.subject, m.src_dim, m.vec FROM memories m'
            ' LEFT JOIN memories_vec v ON v.memory_id = m.id'
            ' WHERE v.memory_id IS NULL AND m.src_dim > 0'
        ))
        if not rows:
            return
        self._db.executemany(
            'INSERT INTO memories_vec(memory_id, user_id, src_dim, kind, subject, v)'
            ' VALUES (?,?,?,?,?,vec_int8(?))',
            [(r['id'], r['user_id'], r['src_dim'], r['kind'], r['subject'] or '',
              _pad_to_index_width(r['vec'])) for r in rows],
        )
        self._db.commit()
        log.info('added %d memories to the vector index', len(rows))

    # -- which embedder the vectors are under -------------------------------

    def embedder(self) -> str:
        """The embedder this store's vectors are under, or '' if unrecorded."""
        row = self._db.execute('SELECT embedder FROM memory_index_state WHERE id = 1').fetchone()
        return row['embedder'] if row else ''

    def use_embedder(self, embedder: str) -> IndexReset | None:
        """Declare which embedder is now in use, wiping the index if it changed.

        This is the one place that protects recall from a changed embedding
        model, and it has to be here rather than left to the operator, because
        the failure mode is silent. A new model's vectors live in a different
        space: the old ones do not *fail* against them, they score. Cosine
        against unrelated geometry returns plausible middling numbers, so a
        store that was not reset answers questions with whichever old memory
        happened to land near the new query. Nothing logs, nothing raises, and
        the recall is quietly wrong rather than quietly empty.

        So a changed embedder discards every vector. The **texts are kept** --
        they are the thing the user actually owns, and they are re-embedded
        afterwards from `pending_reembed`. Until that runs, recall finds
        nothing, which is the honest state and the one the service reports.

        Returns what it discarded, or None when nothing changed: an unrecorded
        store adopts the name (its vectors were made by whatever is configured
        now, which is the only assumption available and does not destroy
        anything), and a matching one is left alone.
        """
        row = self._db.execute(
            'SELECT embedder, dims FROM memory_index_state WHERE id = 1').fetchone()
        if row is not None and row['embedder'] == embedder and int(row['dims']) == EMBED_DIMS:
            return None

        if row is None:
            # First open since this column existed. Stamp the rows that already
            # have usable vectors rather than throwing them away: they were made
            # by whatever is configured now, and a width change was already
            # caught by `src_dim`.
            self._db.execute(
                "UPDATE memories SET embedder = ? WHERE embedder = '' AND src_dim > 0", (embedder,))
            self._db.execute(
                'INSERT INTO memory_index_state (id, embedder, dims, set_at) VALUES (1,?,?,?)',
                (embedder, EMBED_DIMS, time.time()))
            self._db.commit()
            return None

        previous = row['embedder']
        discarded = self.discard_vectors()
        self._db.execute(
            'UPDATE memory_index_state SET embedder = ?, dims = ?, set_at = ? WHERE id = 1',
            (embedder, EMBED_DIMS, time.time()))
        self._db.commit()
        log.warning(
            'embedder changed from %r to %r: discarded %d vector(s); the texts are kept and '
            'will be embedded again', previous or '(unrecorded)', embedder, discarded)
        return IndexReset(previous=previous, embedder=embedder, discarded=discarded)

    # -- manual control ----------------------------------------------------

    def discard_vectors(self, user_id: str | None = None) -> int:
        """Throw away vectors, keep the memories. Returns how many.

        The vector is a derived artefact and the text is not, so "reset the
        index" means this and not `wipe`. A row left behind carries
        `src_dim = -1`, which no search can match and no backfill will pick up,
        so a half-finished reset is unsearchable rather than wrong.
        """
        clause, params = ('', ())
        if user_id is not None:
            clause, params = (' AND user_id = ?', (user_id,))
        cur = self._db.execute(
            "UPDATE memories SET vec = x'', scale = 0, dim = 0, src_dim = -1"
            ' WHERE src_dim > 0' + clause, params)
        if self._vec:
            if user_id is None:
                self._db.execute('DELETE FROM memories_vec')
            else:
                self._db.execute('DELETE FROM memories_vec WHERE user_id = ?', (user_id,))
        self._db.commit()
        return cur.rowcount

    def rebuild_index(self) -> int:
        """Drop the sqlite-vec table and fill it again from `memories`.

        Cheap and non-destructive: `memories.vec` is the record and the index is
        a copy of it, so this costs one pass and fixes an index that was
        interrupted, built by a different build, or never built because the
        extension was missing at the time. Returns the number of vectors in it
        afterwards, and 0 on an install with no extension -- where there is no
        index to rebuild and the scan was always answering.
        """
        if not self._vec:
            return 0
        self._db.execute('DROP TABLE IF EXISTS memories_vec')
        self._db.commit()
        self._vec = self._open_vector_index()
        row = self._db.execute('SELECT COUNT(*) AS n FROM memories_vec').fetchone()
        return int(row['n']) if row else 0

    def index_state(self) -> IndexState:
        row = self._db.execute(
            'SELECT embedder, dims, set_at FROM memory_index_state WHERE id = 1').fetchone()
        counts = self._db.execute(
            'SELECT COUNT(*) AS total,'
            ' SUM(CASE WHEN src_dim > 0 THEN 1 ELSE 0 END) AS live,'
            ' SUM(CASE WHEN src_dim < 0 THEN 1 ELSE 0 END) AS stale'
            ' FROM memories').fetchone()
        return IndexState(
            embedder=row['embedder'] if row else '',
            dims=int(row['dims']) if row else EMBED_DIMS,
            set_at=float(row['set_at']) if row else 0.0,
            total=int(counts['total'] or 0),
            searchable=int(counts['live'] or 0),
            pending=int(counts['stale'] or 0),
            accelerated=bool(self._vec),
        )

    def pending_reembed(self, limit: int = 256) -> list[tuple[str, str, str]]:
        """(id, user_id, text) for memories whose vector was discarded."""
        return [
            (r['id'], r['user_id'], r['text'])
            for r in self._db.execute(
                'SELECT id, user_id, text FROM memories WHERE src_dim < 0'
                ' ORDER BY created_at LIMIT ?', (limit,))
        ]

    def set_vector(self, memory_id: str, vector: list[float] | np.ndarray,
                   embedder: str | None = None) -> bool:
        """Give an existing memory a new vector, under today's embedder.

        What re-embedding after a reset is made of. Quantises and indexes by
        exactly the path `add` takes, so a re-made row is indistinguishable
        from a freshly written one.
        """
        row = self._db.execute(
            'SELECT user_id, kind, subject FROM memories WHERE id = ?', (memory_id,)).fetchone()
        if row is None:
            return False
        blob, scale, proj_dims, src_dim = self._encode(row['user_id'], vector)
        name = self.embedder() if embedder is None else embedder
        self._db.execute(
            'UPDATE memories SET dim = ?, src_dim = ?, scale = ?, vec = ?, embedder = ?'
            ' WHERE id = ?', (proj_dims, src_dim, scale, blob, name, memory_id))
        if self._vec:
            # The row may still be indexed under its old vector if something
            # put it there; one delete keeps this idempotent.
            self._db.execute('DELETE FROM memories_vec WHERE memory_id = ?', (memory_id,))
            self._db.execute(
                'INSERT INTO memories_vec(memory_id, user_id, src_dim, kind, subject, v)'
                ' VALUES (?,?,?,?,?,vec_int8(?))',
                (memory_id, row['user_id'], src_dim, row['kind'], row['subject'] or '',
                 _pad_to_index_width(blob)))
        self._db.commit()
        return True

    def close(self) -> None:
        self._db.close()

    # -- consent ------------------------------------------------------------

    def settings(self, user_id: str) -> Settings:
        row = self._db.execute(
            'SELECT enabled, auto_capture FROM memory_settings WHERE user_id = ?', (user_id,)
        ).fetchone()
        # No row means never asked, which means no.
        if row is None:
            return Settings()
        return Settings(enabled=bool(row['enabled']), auto_capture=bool(row['auto_capture']))

    def set_settings(self, user_id: str, *, enabled: bool, auto_capture: bool) -> Settings:
        self._ensure_key(user_id)
        self._db.execute(
            'UPDATE memory_settings SET enabled = ?, auto_capture = ? WHERE user_id = ?',
            (int(enabled), int(auto_capture and enabled), user_id),
        )
        self._db.commit()
        return self.settings(user_id)

    def _ensure_key(self, user_id: str) -> bytes:
        row = self._db.execute('SELECT key FROM memory_settings WHERE user_id = ?', (user_id,)).fetchone()
        if row is not None:
            return row['key']
        key = os.urandom(32)
        self._db.execute(
            'INSERT INTO memory_settings (user_id, enabled, auto_capture, key) VALUES (?, 0, 0, ?)',
            (user_id, key),
        )
        self._db.commit()
        return key

    # -- writing ------------------------------------------------------------

    def _encode(self, user_id: str, vector: list[float] | np.ndarray) -> tuple[bytes, float, int, int]:
        """One model vector, as the columns a row stores it in.

        Shared by `add` and `set_vector` so a re-embedded row is encoded by the
        same path as a new one -- two copies of this arithmetic is how the
        rebuilt half of a store ends up in a slightly different space from the
        half that was written normally.
        """
        vec = np.asarray(vector, dtype=np.float32)
        # Normalised on the way in, so search is a dot product rather than a
        # dot product and two norms per row.
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        src_dim = int(vec.shape[0])
        proj = _projection(self._ensure_key(user_id), src_dim)
        blob, scale = _quantise(_project(proj, vec))
        return blob, scale, proj.dims, src_dim

    def add(
        self,
        user_id: str,
        text: str,
        vector: list[float] | np.ndarray,
        *,
        kind: str = 'fact',
        subject: str | None = None,
        source: str | None = None,
        embedder: str | None = None,
    ) -> Memory:
        blob, scale, proj_dims, src_dim = self._encode(user_id, vector)
        # Defaults to whatever the store was opened against, so a caller that
        # does not care cannot write a row the search will then filter out.
        name = self.embedder() if embedder is None else embedder

        memory = Memory(
            id=uuid.uuid4().hex[:16],
            user_id=user_id,
            kind=kind,
            subject=subject,
            text=text,
            source=source,
            created_at=time.time(),
        )
        self._db.execute(
            'INSERT INTO memories (id, user_id, kind, subject, text, source, created_at, accessed_at,'
            ' hits, dim, src_dim, embedder, scale, vec) VALUES (?,?,?,?,?,?,?,NULL,0,?,?,?,?,?)',
            (memory.id, user_id, kind, subject, text, source, memory.created_at,
             proj_dims, src_dim, name, scale, blob),
        )
        if self._vec:
            self._db.execute(
                'INSERT INTO memories_vec(memory_id, user_id, src_dim, kind, subject, v)'
                ' VALUES (?,?,?,?,?,vec_int8(?))',
                (memory.id, user_id, src_dim, kind, subject or '', _pad_to_index_width(blob)),
            )
        self._db.commit()
        return memory

    def reproject_legacy_rows(self) -> int:
        """Bring rows written before EMBED_DIMS into the new projection.

        Old rows were stored at the model's full width under a permutation and
        a sign flip. They carry `src_dim = 0`, which the search filters on, so
        without this they would not be *wrong* — they would silently stop being
        found, which is worse. Somebody's memory would quietly empty out and
        nothing would say why.

        **No re-embedding and no provider call.** The old transform is a
        permutation and a sign flip: both invertible, and the permutation is
        known from the key. So an old row can be un-scrambled back into an
        int8-accurate approximation of the model vector it came from, and
        pushed through the new projection locally. The quantisation error it
        carries is about 0.002 of cosine, well under the projection's own.

        Runs once at open. A store with nothing legacy in it does one indexed
        COUNT and returns.
        """
        pending = self._db.execute(
            'SELECT COUNT(*) AS n FROM memories WHERE src_dim = 0'
        ).fetchone()['n']
        if not pending:
            return 0

        done = 0
        for user_id in [
            r['user_id'] for r in
            self._db.execute('SELECT DISTINCT user_id FROM memories WHERE src_dim = 0')
        ]:
            key = self._ensure_key(user_id)
            rows = list(self._db.execute(
                'SELECT id, dim, scale, vec FROM memories WHERE user_id = ? AND src_dim = 0',
                (user_id,),
            ))
            for row in rows:
                src_dim = int(row['dim'])
                stored = np.frombuffer(row['vec'], dtype=np.int8).astype(np.float32) * row['scale']
                if stored.shape[0] != src_dim:
                    # A row whose width disagrees with its own column cannot be
                    # un-scrambled safely. Left alone and reported rather than
                    # guessed at.
                    log.warning('memory %s has dim %d and %d bytes; skipped', row['id'], src_dim, stored.shape[0])
                    continue
                perm, signs = _legacy_projection(key, src_dim)
                original = np.empty(src_dim, dtype=np.float32)
                # stored == original[perm] * signs, and signs are +-1, so
                # multiplying undoes the flip and the scatter undoes the
                # permutation.
                original[perm] = stored * signs

                proj = _projection(key, src_dim)
                blob, scale = _quantise(_project(proj, original))
                self._db.execute(
                    'UPDATE memories SET dim = ?, src_dim = ?, scale = ?, vec = ? WHERE id = ?',
                    (proj.dims, src_dim, scale, blob, row['id']),
                )
                done += 1
        self._db.commit()
        # Every vector just changed, so the index holds the old ones. Emptied
        # rather than updated row by row: the backfill that follows is one
        # statement and cannot leave a mixture behind.
        if getattr(self, '_vec', False):
            self._db.execute('DELETE FROM memories_vec')
            self._db.commit()
            self._backfill_vector_index()
        log.info('re-projected %d memories into %d dimensions', done, EMBED_DIMS)
        return done

    def delete(self, user_id: str, memory_id: str) -> bool:
        cur = self._db.execute('DELETE FROM memories WHERE id = ? AND user_id = ?', (memory_id, user_id))
        # The index holds a copy of the vector. A delete that leaves it behind
        # is a memory somebody removed that still comes back in a search --
        # the same class of bug as an erase that leaves the vectors, and it
        # would not look like a failure from anywhere.
        if self._vec:
            self._db.execute(
                'DELETE FROM memories_vec WHERE memory_id = ? AND user_id = ?', (memory_id, user_id))
        self._db.commit()
        return cur.rowcount > 0

    def wipe(self, user_id: str) -> int:
        """Delete everything, key included.

        The key goes too: leaving it would let a restored backup be searched
        again. A new one is minted if they ever opt back in, which also means
        old vectors could never be compared with new ones.
        """
        cur = self._db.execute('DELETE FROM memories WHERE user_id = ?', (user_id,))
        self._db.execute('DELETE FROM memory_settings WHERE user_id = ?', (user_id,))
        if self._vec:
            self._db.execute('DELETE FROM memories_vec WHERE user_id = ?', (user_id,))
        self._db.commit()
        return cur.rowcount

    # -- reading ------------------------------------------------------------

    def list(self, user_id: str, *, kind: str | None = None, subject: str | None = None,
             limit: int = 200) -> list[Memory]:
        sql = 'SELECT id, user_id, kind, subject, text, source, created_at, hits FROM memories WHERE user_id = ?'
        params: list[object] = [user_id]
        if kind:
            sql += ' AND kind = ?'
            params.append(kind)
        if subject:
            sql += ' AND subject = ?'
            params.append(subject)
        sql += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        return [
            Memory(
                id=r['id'], user_id=r['user_id'], kind=r['kind'], subject=r['subject'], text=r['text'],
                source=r['source'], created_at=r['created_at'], hits=r['hits'],
            )
            for r in self._db.execute(sql, params)
        ]

    def count(self, user_id: str) -> int:
        row = self._db.execute('SELECT COUNT(*) AS n FROM memories WHERE user_id = ?', (user_id,)).fetchone()
        return int(row['n'])

    def _search_indexed(
        self,
        user_id: str,
        projected: np.ndarray,
        *,
        src_dim: int,
        kind: str | None,
        subject: str | None,
        limit: int,
        min_score: float,
    ) -> list[Memory] | None:
        """Nearest memories through sqlite-vec, or None to fall back.

        The query vector is quantised exactly as a stored one is, so the two
        sides are the same kind of thing -- an unquantised query compared
        against quantised rows is a subtly different geometry and would score
        slightly differently from the scan it is meant to replace.

        `distance_metric=cosine` returns a DISTANCE, so the score the rest of
        openmirror speaks in is `1 - distance`.

        Returns None rather than raising if the index cannot answer. A memory
        search that fails because an optional extension misbehaved should
        degrade to the scan, not to an error in the middle of somebody's turn.
        """
        blob, _scale = _quantise(projected)
        blob = _pad_to_index_width(blob)
        clauses = ['user_id = ?', 'src_dim = ?']
        params: list[object] = [user_id, src_dim]
        if kind:
            clauses.append('kind = ?')
            params.append(kind)
        if subject:
            clauses.append('subject = ?')
            params.append(subject)
        params.append(blob)
        params.append(int(limit))
        try:
            rows = list(self._db.execute(
                'SELECT memory_id, distance FROM memories_vec WHERE '
                + ' AND '.join(clauses)
                + ' AND v MATCH vec_int8(?) AND k = ?',
                tuple(params),
            ))
        except sqlite3.Error as exc:
            log.warning('the vector index could not answer (%s); scanning instead', exc)
            return None

        # A NULL distance means the cosine was undefined — one side was a zero
        # vector, which is what an embedding model returns when it had nothing
        # to work with. The scan skips those rows explicitly (`snorm == 0`);
        # this is the same rule, and without it `float(None)` ends the turn with
        # a TypeError rather than a quiet non-match.
        scored = [
            (r['memory_id'], 1.0 - float(r['distance']))
            for r in rows if r['distance'] is not None
        ]
        scored = [(mid, sc) for mid, sc in scored if sc >= min_score]
        if not scored:
            return []

        # The index knows ids and distances and nothing else, so the text comes
        # from `memories` -- k rows, not the whole table, which is the half of
        # the win the arithmetic does not explain.
        by_id = dict(scored)
        rows = self._db.execute(
            'SELECT id, user_id, kind, subject, text, source, created_at, hits FROM memories'
            ' WHERE id IN (' + ','.join('?' * len(by_id)) + ')',
            tuple(by_id.keys()),
        )
        out = [
            Memory(
                id=r['id'], user_id=r['user_id'], kind=r['kind'], subject=r['subject'],
                text=r['text'], source=r['source'], created_at=r['created_at'],
                hits=r['hits'], score=by_id[r['id']],
            )
            for r in rows
        ]
        out.sort(key=lambda m: m.score, reverse=True)
        return out

    def search(
        self,
        user_id: str,
        vector: list[float] | np.ndarray,
        *,
        limit: int = 5,
        min_score: float = 0.25,
        kind: str | None = None,
        subject: str | None = None,
        embedder: str | None = None,
    ) -> list[Memory]:
        """Nearest memories by cosine, scoped to one person by SQL.

        The scan cannot reach another user's rows, and could not use them if it
        did: their vectors are under a different projection, so a match across
        users is not merely forbidden, it is meaningless.

        Scoped to one embedder for the same reason. `src_dim` catches a model of
        a different width; `embedder` catches one of the same width, which is
        the case that would otherwise score and look like an answer.
        """
        query = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm
        src_dim = int(query.shape[0])

        # Only rows the SAME model produced. Filtered in SQL rather than
        # skipped in the loop: a mailbox that has changed embedder is mostly
        # rows that cannot be scored, and pulling them out of the database to
        # discard them is the expensive half of the scan.
        name = self.embedder() if embedder is None else embedder
        clauses = ['user_id = ?', 'src_dim = ?', 'embedder = ?']
        params: list[object] = [user_id, src_dim, name]
        if kind:
            clauses.append('kind = ?')
            params.append(kind)
        if subject:
            # "What do you know about the openmirror project" -- narrowed before the
            # scan rather than filtered after it, so a small subject's memories
            # are not pushed out of the top-k by a large one's.
            clauses.append('subject = ?')
            params.append(subject)
        rows = list(
            self._db.execute(
                'SELECT id, user_id, kind, subject, text, source, created_at, hits, dim, scale, vec'
                ' FROM memories WHERE ' + ' AND '.join(clauses),
                tuple(params),
            )
        )
        if not rows:
            return []

        proj = _projection(self._ensure_key(user_id), src_dim)
        projected = _project(proj, query)
        pnorm = float(np.linalg.norm(projected))
        if pnorm > 0:
            projected = projected / pnorm

        if self._vec:
            found = self._search_indexed(
                user_id, projected, src_dim=src_dim, kind=kind, subject=subject,
                limit=limit, min_score=min_score,
            )
            if found is not None:
                # The vec0 table carries no embedder column, so the one clause
                # it cannot apply is applied here against the rows the scan
                # would have considered. It only ever removes rows the scan
                # would have removed too, so the two paths still agree.
                allowed = {r['id'] for r in rows}
                return self._record_use([m for m in found if m.id in allowed])

        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            stored = np.frombuffer(row['vec'], dtype=np.int8).astype(np.float32)
            snorm = float(np.linalg.norm(stored))
            if snorm == 0.0:
                continue
            # Cosine, explicitly. Both sides are normalised here rather than
            # relying on the stored vector already being unit length: the
            # per-vector quantisation scale means it is only approximately so,
            # and an unnormalised dot product quietly ranks longer vectors
            # higher for no reason anyone intended.
            scored.append((float(np.dot(projected, stored / snorm)), row))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        out: list[Memory] = []
        for score, row in scored[:limit]:
            if score < min_score:
                break
            out.append(
                Memory(
                    id=row['id'], user_id=row['user_id'], kind=row['kind'], subject=row['subject'], text=row['text'],
                    source=row['source'], created_at=row['created_at'], hits=row['hits'], score=score,
                )
            )

        return self._record_use(out)

    def _record_use(self, found: list[Memory]) -> list[Memory]:
        """Mark what a search returned as used, and hand it back.

        Shared by both search paths deliberately. It lived at the tail of the
        scan, and adding the indexed path in front of it silently stopped
        recording anything — hits froze at zero, `accessed_at` never moved, and
        nothing failed. Whatever ranks or prunes memories by use would have
        been reading a column that had quietly stopped being written.
        """
        if found:
            now = time.time()
            self._db.executemany(
                'UPDATE memories SET hits = hits + 1, accessed_at = ? WHERE id = ?',
                [(now, m.id) for m in found],
            )
            self._db.commit()
        return found
