"""Seeing and changing what answers each modality."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from roost.config import config
from roost.providers.base import Modality
from roost.providers.registry import NoProviderError, Route, RouteSet, registry

router = APIRouter(prefix='/api/providers')


@router.get('')
async def list_providers() -> dict[str, Any]:
    return {
        'providers': [
            {
                'id': p.id,
                'label': p.label,
                'local': p.local,
                'base_url': p.base_url,
                'modalities': sorted(m.value for m in p.modalities),
            }
            for p in registry.list()
        ],
        'capabilities': registry.capabilities(local_only=config.local_only),
        'local_only': config.local_only,
    }


@router.get('/{provider_id}/models')
async def list_models(provider_id: str) -> dict[str, Any]:
    for entry in registry.list():
        if entry.id != provider_id:
            continue
        impl, _, _ = registry.resolve(Modality.CHAT, RouteSet(routes={Modality.CHAT: Route(provider=provider_id)}))
        return {'models': await impl.models()}
    raise HTTPException(status_code=404, detail=f'no such provider: {provider_id}')


class PerchConnect(BaseModel):
    host: str
    token: str = ''
    scheme: str = 'http'


@router.post('/perch/connect')
async def connect_perch(body: PerchConnect) -> dict[str, Any]:
    """Add a Perch host as whichever of its services are switched on.

    The probe is the point: Perch turns each service off until asked, so
    registering all five blind would offer a video generator that is not
    there. What comes back says which are live, which is also the honest
    answer to "did that work".
    """
    from roost.providers import perch as perch_mod

    cfg = perch_mod.PerchConfig(host=body.host, token=body.token, scheme=body.scheme)
    available = await perch_mod.probe(cfg)

    if not any(available.values()):
        raise HTTPException(
            status_code=502,
            detail=f'nothing answered at {body.host} on ports {", ".join(str(p) for p in perch_mod.DEFAULT_PORTS.values())}',
        )

    registered = registry.register_perch(cfg, available)
    return {
        'registered': registered,
        'services': available,
        'capabilities': registry.capabilities(local_only=config.local_only),
    }


class RouteUpdate(BaseModel):
    modality: str
    provider: str
    model: str = ''
    options: dict[str, Any] = {}


@router.put('/routes')
async def set_route(body: RouteUpdate) -> dict[str, Any]:
    try:
        modality = Modality(body.modality)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f'unknown modality: {body.modality}') from exc

    defaults = registry.defaults
    defaults.routes[modality] = Route(provider=body.provider, model=body.model, options=body.options)

    # Resolved immediately so a route that cannot work is rejected now rather
    # than at the start of someone's next call.
    try:
        registry.resolve(modality, defaults)
    except NoProviderError as exc:
        del defaults.routes[modality]
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    registry.set_defaults(defaults)
    return {'ok': True, 'routes': {m.value: r.provider for m, r in defaults.routes.items()}}
