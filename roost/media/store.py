"""Where generated images and video are kept.

A file and a JSON sidecar beside it, rather than a table. Two reasons, and the
second is the real one:

**A generation is worth nothing without its settings.** The seed, the sampler,
the steps and the exact prompt are what make an image reproducible, and an
image you cannot reproduce is an image you cannot iterate on. So the recipe is
stored next to the result and returned with it.

**The files should survive this program.** A directory of PNGs and JSON is
readable by everything and needs no export feature; a row in a database that
only Roost can open is a hostage. Someone who stops using this should keep
their pictures, and they will, because they are just files.

Nothing is ever overwritten: ids are random, so two generations with identical
settings are two files. Disk is cheaper than losing the one you liked.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

EXTENSIONS = {
    'image/png': '.png',
    'image/jpeg': '.jpg',
    'image/webp': '.webp',
    'video/mp4': '.mp4',
    'video/webm': '.webm',
    'image/gif': '.gif',
}


@dataclass(slots=True)
class Media:
    id: str
    kind: str                 # 'image' or 'video'
    media_type: str
    filename: str
    bytes: int
    created_at: float
    prompt: str = ''
    provider: str = ''
    model: str = ''
    seed: int | None = None
    # Everything that was sent to the generator, so the result can be made
    # again. Kept whole rather than filtered to what this build understands:
    # a parameter added by a future version is still worth having recorded.
    params: dict[str, Any] = field(default_factory=dict)
    # What the caller asked for, when that is not what happened — a provider
    # that revised the prompt, a size it rounded, a model it substituted.
    notes: str = ''

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data['url'] = f'/api/media/{self.id}/file'
        return data


class MediaStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self, created: float) -> Path:
        # Bucketed by month so that a year of daily generation is not one
        # directory with forty thousand entries in it, which every file
        # manager and half the tooling handles badly.
        return self.root / time.strftime('%Y-%m', time.localtime(created))

    def add(
        self,
        data: bytes,
        *,
        kind: str,
        media_type: str,
        prompt: str = '',
        provider: str = '',
        model: str = '',
        seed: int | None = None,
        params: dict[str, Any] | None = None,
        notes: str = '',
    ) -> Media:
        created = time.time()
        ident = uuid.uuid4().hex[:16]
        extension = EXTENSIONS.get(media_type, '.bin')
        directory = self._dir(created)
        directory.mkdir(parents=True, exist_ok=True)

        path = directory / f'{ident}{extension}'
        path.write_bytes(data)

        media = Media(
            id=ident,
            kind=kind,
            media_type=media_type,
            filename=path.name,
            bytes=len(data),
            created_at=created,
            prompt=prompt,
            provider=provider,
            model=model,
            seed=seed,
            params=params or {},
            notes=notes,
        )
        (directory / f'{ident}.json').write_text(json.dumps(asdict(media), indent=2))
        return media

    def path(self, media_id: str) -> Path | None:
        """The file for an id.

        The id is matched as a filename stem rather than joined onto a path, so
        a request for `../../etc/passwd` finds nothing instead of finding
        something.
        """
        if not media_id.isalnum():
            return None
        for candidate in self.root.glob(f'*/{media_id}.*'):
            if candidate.suffix != '.json':
                return candidate
        return None

    def get(self, media_id: str) -> Media | None:
        if not media_id.isalnum():
            return None
        for sidecar in self.root.glob(f'*/{media_id}.json'):
            try:
                return Media(**json.loads(sidecar.read_text()))
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                log.warning('media %s has an unreadable sidecar: %s', media_id, exc)
        return None

    def list(self, *, limit: int = 60, kind: str = '') -> list[Media]:
        """The most recent first, which is the only order a gallery wants."""
        out: list[Media] = []
        for sidecar in sorted(self.root.glob('*/*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                media = Media(**json.loads(sidecar.read_text()))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            if kind and media.kind != kind:
                continue
            out.append(media)
            if len(out) >= limit:
                break
        return out

    def remove(self, media_id: str) -> bool:
        path = self.path(media_id)
        if path is None:
            return False
        path.unlink(missing_ok=True)
        path.with_suffix('.json').unlink(missing_ok=True)
        return True
