"""Undo, for an agent that edits real files.

An autonomous agent with write access will eventually do something you did not
want, and the moment you notice is usually several steps later. Being able to
say "put it back the way it was before that turn" is worth more than any
amount of care beforehand, because care is what fails.

Snapshots are content-addressed: the same bytes are stored once no matter how
many turns touch the file, so a session that edits one file thirty times costs
thirty hashes and a handful of blobs. Restoring is writing bytes back, which
means it works on a file the agent has since deleted, and on one somebody has
edited by hand in between — the second case is why `restore` reports what it
overwrote rather than doing it silently.

**What this does not cover, and cannot.** Only changes made through the file
tools are recorded. A shell command can write anywhere, and snapshotting the
whole filesystem before every command is not a thing anybody wants. So this is
an undo for the agent's own edits, not a transaction over your machine. Where
the working root is a git repository, git remains the better answer and this
says so.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# A file bigger than this is not snapshotted. Source files are kilobytes; a
# 200 MB artefact in the working tree should not quietly cost 200 MB per turn.
MAX_SNAPSHOT_BYTES = 8_000_000


@dataclass(slots=True)
class FileState:
    path: Path
    # None means the file did not exist. Restoring to that state deletes it,
    # which is the correct undo of "the agent created this".
    digest: str | None
    size: int = 0
    # What the file looked like when the turn *finished*, recorded at commit.
    # Needed to tell "a person edited this afterwards" from "the agent edited
    # it", which is the whole point of the warning — comparing against the
    # before-state instead makes it fire on every single rewind, and a warning
    # that always fires is one nobody reads.
    after: str | None = None


@dataclass(slots=True)
class Checkpoint:
    id: str
    turn_id: str
    label: str
    at: float = field(default_factory=time.time)
    files: dict[str, FileState] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.files)


@dataclass(slots=True)
class RestoreReport:
    restored: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    # Files whose current contents differ from what the agent left behind:
    # somebody edited them by hand since. Reported, not silently clobbered.
    changed_since: list[str] = field(default_factory=list)


class CheckpointStore:
    """Snapshots for one session."""

    def __init__(self, root: Path) -> None:
        # Under the session's own directory rather than beside the files, so a
        # snapshot never appears in the tree the agent is working on — where it
        # would be read, grepped, and eventually committed.
        self.root = root
        self.blobs = root / 'blobs'
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.checkpoints: list[Checkpoint] = []
        self._open: Checkpoint | None = None

    # -- recording ----------------------------------------------------------

    def begin(self, turn_id: str, label: str) -> Checkpoint:
        """Open a checkpoint for a turn. Nothing is stored until a file is touched."""
        self._open = Checkpoint(id=f'{len(self.checkpoints) + 1}', turn_id=turn_id, label=label)
        return self._open

    def commit(self) -> Checkpoint | None:
        """Close the open checkpoint, keeping it only if it recorded anything.

        A turn that answered a question without touching a file should not
        appear in the undo list; an undo list full of no-ops is one nobody
        reads far enough down.
        """
        cp = self._open
        self._open = None
        if cp is None or not cp.files:
            return None

        # Record where the turn left each file, so a later rewind can tell
        # whether anybody has touched it since.
        for state in cp.files.values():
            try:
                state.after = (
                    hashlib.sha256(state.path.read_bytes()).hexdigest()
                    if state.path.is_file() else None
                )
            except OSError:
                state.after = None

        self.checkpoints.append(cp)
        return cp

    def record(self, path: Path) -> None:
        """Snapshot a file as it is *now*, before it is changed.

        Called before every write. The first record of a path in a checkpoint
        wins: later ones would capture the agent's own intermediate states,
        and the point is to get back to before the turn.
        """
        if self._open is None:
            return
        key = str(path)
        if key in self._open.files:
            return

        try:
            if not path.exists():
                self._open.files[key] = FileState(path=path, digest=None)
                return
            if path.is_dir():
                return
            size = path.stat().st_size
            if size > MAX_SNAPSHOT_BYTES:
                log.debug('checkpoint: %s is %d bytes, too large to snapshot', path, size)
                return
            data = path.read_bytes()
        except OSError as exc:
            log.debug('checkpoint: could not read %s: %s', path, exc)
            return

        digest = hashlib.sha256(data).hexdigest()
        blob = self.blobs / digest
        if not blob.exists():
            # Written via a temporary file so a crash mid-write cannot leave a
            # truncated blob under a hash that claims to be complete.
            tmp = blob.with_suffix('.partial')
            tmp.write_bytes(data)
            tmp.replace(blob)
        self._open.files[key] = FileState(path=path, digest=digest, size=size)

    # -- restoring ----------------------------------------------------------

    def restore(self, checkpoint_id: str, *, force: bool = False) -> RestoreReport:
        """Put every file back to how it was when the checkpoint opened.

        Restores *this* checkpoint and every one after it, because undoing turn
        three while leaving turns four and five in place produces a tree that
        never existed and that nobody asked for.
        """
        index = next((i for i, c in enumerate(self.checkpoints) if c.id == checkpoint_id), None)
        if index is None:
            raise KeyError(f'no checkpoint {checkpoint_id!r}')

        # Later checkpoints first, so the earliest recorded state of each file
        # is the one that ends up on disk.
        wanted: dict[str, FileState] = {}
        for cp in reversed(self.checkpoints[index:]):
            wanted.update(cp.files)

        report = RestoreReport()
        for key, state in wanted.items():
            path = Path(key)
            try:
                if state.digest is None:
                    if path.exists():
                        path.unlink()
                        report.deleted.append(key)
                    continue

                blob = self.blobs / state.digest
                if not blob.exists():
                    report.skipped[key] = 'the snapshot is missing'
                    continue

                if path.exists():
                    current = hashlib.sha256(path.read_bytes()).hexdigest()
                    if current == state.digest:
                        continue          # already as it was; nothing to do
                    # Compared against what the agent *left*, not what it found.
                    # A difference here means somebody edited the file since,
                    # and their work is about to be overwritten.
                    if state.after is not None and current != state.after:
                        report.changed_since.append(key)

                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(blob, path)
                report.restored.append(key)
            except OSError as exc:
                report.skipped[key] = str(exc)

        # The undone checkpoints go, so the list always describes what could
        # still be undone rather than what once happened.
        del self.checkpoints[index:]
        return report

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                'id': cp.id,
                'turn_id': cp.turn_id,
                'label': cp.label,
                'at': cp.at,
                'files': cp.count,
                'paths': sorted(cp.files)[:20],
            }
            for cp in reversed(self.checkpoints)
        ]
