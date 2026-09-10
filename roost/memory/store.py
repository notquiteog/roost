"""Where memories live.

Three decisions shape this file.

**SQLite and a full scan, not a vector database.** Roost is a client on one
person's machine — there are no accounts and there is not going to be a
multi-tenant install — so the store holds one person's memories, which number
in the thousands. At that size an approximate index buys nothing measurable and
costs a service to run, a schema to keep in sync, and a second thing that can
be down.

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
thousand, noticeable at ten thousand, and a real pause at fifty — which a
long-lived daily install reaches in a couple of years, not never.

**And when it arrives, the fix is not a vector database.** Vectorising in numpy
does not help: scoring from a resident float32 matrix is fast but costs 512 MB
at fifty thousand rows, which defeats the int8 quantisation below. The fix that
is already proven in this family is Tern's — a keyed projection down to a fixed
narrow width before storage, which composes with the rotation below because
both are orthogonal. At 256 dimensions fifty thousand vectors are 13 MB rather
than 128, and the scan gets an order of magnitude cheaper without adding a
dependency, a service, or a second thing that can be down.

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

import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
  id          TEXT PRIMARY KEY,
  user_id     TEXT NOT NULL,
  kind        TEXT NOT NULL,
  text        TEXT NOT NULL,
  source      TEXT,
  created_at  REAL NOT NULL,
  accessed_at REAL,
  hits        INTEGER NOT NULL DEFAULT 0,
  dim         INTEGER NOT NULL,
  scale       REAL NOT NULL,
  vec         BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, kind);

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


def _projection(key: bytes, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Derive this user's permutation and sign flip from their key.

    Both are orthogonal transforms, so similarity survives untouched. Derived
    rather than stored because the key is small and the transform is not.
    """
    seed = int.from_bytes(key[:8], 'little')
    rng = np.random.default_rng(seed)
    perm = rng.permutation(dim)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dim)
    return perm, signs


def _quantise(vec: np.ndarray) -> tuple[bytes, float]:
    """int8 with a per-vector scale."""
    peak = float(np.abs(vec).max())
    if peak == 0.0:
        return np.zeros(vec.shape, dtype=np.int8).tobytes(), 1.0
    scale = peak / 127.0
    return np.round(vec / scale).astype(np.int8).tobytes(), scale


class MemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        # WAL so a read during a write does not block; this is read far more
        # than it is written.
        self._db.execute('PRAGMA journal_mode=WAL')
        self._db.executescript(SCHEMA)
        self._db.commit()

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

    def add(
        self,
        user_id: str,
        text: str,
        vector: list[float] | np.ndarray,
        *,
        kind: str = 'fact',
        source: str | None = None,
    ) -> Memory:
        vec = np.asarray(vector, dtype=np.float32)
        # Normalised on the way in, so search is a dot product rather than a
        # dot product and two norms per row.
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm

        perm, signs = _projection(self._ensure_key(user_id), vec.shape[0])
        blob, scale = _quantise(vec[perm] * signs)

        memory = Memory(
            id=uuid.uuid4().hex[:16],
            user_id=user_id,
            kind=kind,
            text=text,
            source=source,
            created_at=time.time(),
        )
        self._db.execute(
            'INSERT INTO memories (id, user_id, kind, text, source, created_at, accessed_at, hits, dim, scale, vec)'
            ' VALUES (?,?,?,?,?,?,NULL,0,?,?,?)',
            (memory.id, user_id, kind, text, source, memory.created_at, vec.shape[0], scale, blob),
        )
        self._db.commit()
        return memory

    def delete(self, user_id: str, memory_id: str) -> bool:
        cur = self._db.execute('DELETE FROM memories WHERE id = ? AND user_id = ?', (memory_id, user_id))
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
        self._db.commit()
        return cur.rowcount

    # -- reading ------------------------------------------------------------

    def list(self, user_id: str, *, kind: str | None = None, limit: int = 200) -> list[Memory]:
        sql = 'SELECT id, user_id, kind, text, source, created_at, hits FROM memories WHERE user_id = ?'
        params: list[object] = [user_id]
        if kind:
            sql += ' AND kind = ?'
            params.append(kind)
        sql += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        return [
            Memory(
                id=r['id'], user_id=r['user_id'], kind=r['kind'], text=r['text'],
                source=r['source'], created_at=r['created_at'], hits=r['hits'],
            )
            for r in self._db.execute(sql, params)
        ]

    def count(self, user_id: str) -> int:
        row = self._db.execute('SELECT COUNT(*) AS n FROM memories WHERE user_id = ?', (user_id,)).fetchone()
        return int(row['n'])

    def search(
        self,
        user_id: str,
        vector: list[float] | np.ndarray,
        *,
        limit: int = 5,
        min_score: float = 0.25,
        kind: str | None = None,
    ) -> list[Memory]:
        """Nearest memories by cosine, scoped to one person by SQL.

        The scan cannot reach another user's rows, and could not use them if it
        did: their vectors are under a different projection, so a match across
        users is not merely forbidden, it is meaningless.
        """
        rows = list(
            self._db.execute(
                'SELECT id, user_id, kind, text, source, created_at, hits, dim, scale, vec'
                ' FROM memories WHERE user_id = ?' + (' AND kind = ?' if kind else ''),
                (user_id, kind) if kind else (user_id,),
            )
        )
        if not rows:
            return []

        query = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm

        perm, signs = _projection(self._ensure_key(user_id), query.shape[0])
        projected = query[perm] * signs

        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            # A vector of a different width is from a different embedding
            # model. Comparing them would return confident nonsense, so they
            # are skipped rather than coerced.
            if row['dim'] != query.shape[0]:
                continue
            stored = np.frombuffer(row['vec'], dtype=np.int8).astype(np.float32) * row['scale']
            scored.append((float(np.dot(projected, stored)), row))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        out: list[Memory] = []
        for score, row in scored[:limit]:
            if score < min_score:
                break
            out.append(
                Memory(
                    id=row['id'], user_id=row['user_id'], kind=row['kind'], text=row['text'],
                    source=row['source'], created_at=row['created_at'], hits=row['hits'], score=score,
                )
            )

        if out:
            now = time.time()
            self._db.executemany(
                'UPDATE memories SET hits = hits + 1, accessed_at = ? WHERE id = ?',
                [(now, m.id) for m in out],
            )
            self._db.commit()
        return out
