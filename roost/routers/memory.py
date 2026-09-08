"""Seeing, correcting and deleting what has been remembered.

Every one of these exists because memory a person cannot inspect is
surveillance rather than a feature. If it can learn about you, you must be
able to read the whole of it, remove one item, and remove all of it — without
asking anyone.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from roost.config import config

router = APIRouter(prefix='/api/memory')

# Set at start-up, and left None when the feature is off for the whole install.
service = None


def _service():
    if service is None:
        # 404 rather than 503: when memory is off for this install, the
        # endpoints genuinely do not exist.
        raise HTTPException(status_code=404, detail='memory is not enabled on this server')
    return service


def _user() -> str:
    return config.default_user


class SettingsBody(BaseModel):
    enabled: bool
    # Passive capture from conversation, which is a bigger step than storing
    # what you explicitly asked to be stored — so it is its own switch.
    auto_capture: bool = False


@router.get('/settings')
async def get_settings() -> dict[str, object]:
    svc = _service()
    settings = svc.settings(_user())
    return {
        'enabled': settings.enabled,
        'auto_capture': settings.auto_capture,
        'count': svc.store.count(_user()),
        'embedding_model': svc.model or None,
    }


@router.put('/settings')
async def put_settings(body: SettingsBody) -> dict[str, object]:
    svc = _service()
    settings = svc.set_settings(_user(), enabled=body.enabled, auto_capture=body.auto_capture)
    return {'enabled': settings.enabled, 'auto_capture': settings.auto_capture}


@router.get('')
async def list_memories(kind: str | None = None, limit: int = 200) -> dict[str, object]:
    svc = _service()
    return {
        'memories': [
            {
                'id': m.id,
                'kind': m.kind,
                'text': m.text,
                'source': m.source,
                'created_at': m.created_at,
                'hits': m.hits,
            }
            for m in svc.store.list(_user(), kind=kind, limit=limit)
        ]
    }


class AddBody(BaseModel):
    text: str


@router.post('')
async def add_memory(body: AddBody) -> dict[str, object]:
    svc = _service()
    memory = await svc.remember(_user(), body.text, kind='fact', source='user')
    if memory is None:
        raise HTTPException(status_code=409, detail='memory is switched off for this user')
    return {'id': memory.id, 'text': memory.text}


class SearchBody(BaseModel):
    query: str
    limit: int = 5


@router.post('/search')
async def search_memories(body: SearchBody) -> dict[str, object]:
    svc = _service()
    result = await svc.recall(_user(), body.query, limit=body.limit)
    return {
        'reason': result.reason,
        'memories': [{'id': m.id, 'text': m.text, 'score': round(m.score, 4), 'kind': m.kind} for m in result.memories],
    }


@router.delete('/{memory_id}')
async def delete_memory(memory_id: str) -> dict[str, bool]:
    svc = _service()
    if not svc.store.delete(_user(), memory_id):
        raise HTTPException(status_code=404, detail='no such memory')
    return {'ok': True}


@router.delete('')
async def wipe() -> dict[str, int]:
    """Delete everything, and the key with it.

    This also switches memory back off: a wipe that left the feature on would
    start refilling from the next message, which is not what anyone means.
    """
    svc = _service()
    return {'deleted': svc.store.wipe(_user())}
