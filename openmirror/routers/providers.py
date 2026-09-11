"""Seeing and changing what answers each modality.

The rule that shapes every endpoint here: **a model list always comes from the
provider.** Nothing in this file returns a remembered list, a cached one, or a
list assembled from what a vendor's documentation said when it was written.
Asking costs one request and buys the only answer that is true — a local
install has whatever happens to be pulled, a gateway's catalogue changes
weekly, and a key may not have access to half of what the vendor publishes.
The two services with no listing endpoint at all say so in the response, with
`source: "catalogue"` on every row, so a UI can mark them rather than implying
the server confirmed them.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from openmirror.config import config
from openmirror.net import tor as tor_mod
from openmirror.providers import catalog, connections
from openmirror.providers.base import Modality
from openmirror.providers.connections import Connection, ConnectionStore
from openmirror.providers.registry import NoProviderError, Route, registry

log = logging.getLogger(__name__)

router = APIRouter(prefix='/api/providers')

store = ConnectionStore(config.connections_db)


def _providers() -> list[dict[str, Any]]:
    return [registry.describe(p.id) for p in registry.list()]


@router.get('')
async def list_providers() -> dict[str, Any]:
    return {
        'providers': _providers(),
        'capabilities': registry.capabilities(local_only=config.local_only),
        'local_only': config.local_only,
        'routes': _routes(),
    }


@router.get('/catalog')
async def get_catalog() -> dict[str, Any]:
    """The presets and the embedding models this build knows by name.

    Sent whole rather than searched server-side: it is a few kilobytes, it
    never changes at runtime, and a picker that can filter locally is a picker
    that stays usable while the daemon is busy.
    """
    return {
        'hosts': [
            {
                'id': h.id,
                'label': h.label,
                'adapter': h.adapter,
                'base_url': h.base_url,
                'env_key': h.env_key,
                'modalities': sorted(m.value for m in h.modalities),
                'local': h.local,
                'lists_models': h.lists_models,
                'note': h.note,
            }
            for h in catalog.HOSTS
        ],
        'embedding_models': [
            {
                'id': m.id,
                'label': m.label,
                'dimensions': m.dimensions,
                'served_by': m.served_by,
                'truncatable': m.truncatable,
                'max_input_tokens': m.max_input_tokens,
                'note': m.note,
            }
            for m in catalog.EMBEDDING_MODELS
        ],
    }


@router.get('/{provider_id}/models')
async def list_models(provider_id: str, modality: str = 'chat') -> dict[str, Any]:
    """What this provider can serve, asked of the provider, now.

    `modality` matters on the adapters that serve several: ComfyUI's video
    models are its installed workflow templates and its image models are not,
    and one list would be wrong for both.
    """
    try:
        kind = Modality(modality)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f'unknown modality: {modality}') from exc

    try:
        impl = registry.impl(provider_id, kind)
    except NoProviderError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    lister = getattr(impl, 'models', None)
    if lister is None:
        return {'models': [], 'live': False, 'reason': f'{provider_id} does not list its models'}

    try:
        models = await lister()
    except Exception as exc:  # noqa: BLE001 - a dead endpoint is an answer, not a crash
        # Reported rather than raised. "The provider did not answer" is a
        # thing a picker should show next to the empty list, not a 500 that
        # makes the whole screen fail.
        log.info('%s did not list models: %s', provider_id, exc)
        return {'models': [], 'live': False, 'reason': str(exc)[:300]}

    # Anything the catalogue recognises gets its dimensions attached, because
    # that is the one embedding property that cannot be changed later.
    if kind is Modality.EMBEDDING:
        for row in models:
            known = catalog.describe_embedding(str(row.get('id', '')))
            if known:
                row.setdefault('dimensions', known.dimensions)
                row.setdefault('label', known.label)
                row.setdefault('truncatable', known.truncatable)

    live = not all(m.get('source') == 'catalogue' for m in models) if models else False
    return {'models': models, 'live': live, 'count': len(models)}


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


class ConnectionBody(BaseModel):
    # Either a preset to start from, or every field spelled out. A preset is
    # only ever a set of defaults; each field below overrides it.
    host_id: str = ''
    id: str = ''
    label: str = ''
    adapter: str = ''
    base_url: str = ''
    # Absent means "keep whatever is stored", which is what lets a UI edit a
    # connection it was never allowed to read the key of.
    api_key: str | None = None
    modalities: list[str] | None = None
    local: bool | None = None
    tor: bool = False
    tor_host: str = ''
    tor_port: int = 0
    enabled: bool = True


@router.get('/connections')
async def list_connections() -> dict[str, Any]:
    return {'connections': [c.redacted() for c in store.load()]}


@router.put('/connections')
async def put_connection(body: ConnectionBody) -> dict[str, Any]:
    """Add or change a connection, and register it immediately.

    Registered rather than staged: the point of managing connections at
    runtime is that the next message uses the new one. A connection that
    cannot be built is rejected here rather than written and left broken.
    """
    if body.host_id:
        try:
            conn = connections.from_host(
                body.host_id,
                api_key=body.api_key or '',
                base_url=body.base_url,
                connection_id=body.id,
                modalities=body.modalities,
                local=body.local,
                tor=body.tor,
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        if not (body.id and body.adapter and body.base_url):
            raise HTTPException(
                status_code=400,
                detail='without host_id, an id, an adapter and a base_url are all required',
            )
        conn = Connection(
            id=body.id,
            label=body.label or body.id,
            adapter=body.adapter,
            base_url=body.base_url,
            api_key=body.api_key or '',
            modalities=body.modalities or [],
            local=bool(body.local),
        )

    if body.label:
        conn.label = body.label
    if body.local is not None:
        conn.local = body.local
    conn.tor = body.tor
    conn.tor_host = body.tor_host
    conn.tor_port = body.tor_port
    conn.enabled = body.enabled

    # A key the client did not send is the key already stored. Otherwise
    # editing the Tor toggle on a connection would silently clear its key,
    # since the client is never given it back to send.
    if body.api_key is None:
        existing = store.get(conn.id)
        if existing:
            conn.api_key = existing.api_key

    if conn.adapter == 'perch':
        return await _save_perch(conn)

    if conn.tor:
        ok, why = await tor_mod.probe(tor_mod.resolve(conn.tor_host, conn.tor_port))
        if not ok:
            # Refused rather than saved-and-broken. A connection marked as
            # going over Tor that silently cannot is the exact failure the
            # toggle exists to prevent.
            raise HTTPException(status_code=400, detail=why)

    try:
        info, impls = connections.build(conn)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f'could not build that connection: {exc}') from exc

    store.put(conn)
    if conn.enabled:
        registry.register(info, impls, connection=conn)
    else:
        registry.unregister(conn.id)

    return {'connection': conn.redacted(), 'providers': _providers()}


@router.delete('/connections/{connection_id}')
async def delete_connection(connection_id: str) -> dict[str, Any]:
    conn = store.get(connection_id)
    if not store.remove(connection_id):
        raise HTTPException(status_code=404, detail=f'no such connection: {connection_id}')
    for provider_id in connections.provider_ids(conn) if conn else [connection_id]:
        registry.unregister(provider_id)
    if conn is not None and conn.adapter == 'perch':
        # A saved Perch replaced the one the environment configured, since both
        # register the same ids. Removing the saved one puts the environment's
        # back, rather than leaving an install with PERCH_HOST set and no Perch.
        await _register_env_perch()
    return {'ok': True, 'providers': _providers()}


@router.post('/connections/{connection_id}/test')
async def test_connection(connection_id: str) -> dict[str, Any]:
    """Ask the connection for its models, and report exactly what came back.

    The most useful thing a "test" button can do, because it exercises the
    address, the key and the route in one go — and a list of real model names
    is unambiguous proof it worked in a way a green tick is not.
    """
    conn = store.get(connection_id)
    if conn is None:
        # A provider the environment configured has no stored connection, but
        # it can still be asked for its models — which is all a test is. Before
        # this, the Test button on every `from .env` row answered "no answer".
        return await _test_registered(connection_id)

    started = asyncio.get_running_loop().time()
    try:
        if conn.adapter == 'perch':
            from openmirror.providers import perch as perch_mod

            cfg = perch_mod.from_connection(conn)
            ok, why = await perch_mod.check_token(cfg)
            if not ok:
                return {'ok': False, 'error': why, 'tor': False}
        info, impls = connections.build(conn)
        impl = next(iter(impls.values()))
        await _check(impl)
        models = await impl.models()
    except Exception as exc:  # noqa: BLE001
        return {'ok': False, 'error': str(exc)[:500], 'tor': conn.tor}
    took = int((asyncio.get_running_loop().time() - started) * 1000)
    return {
        'ok': True,
        'models': len(models),
        'sample': [m.get('id') for m in models[:8]],
        'took_ms': took,
        'tor': conn.tor,
        'modalities': sorted(m.value for m in info.modalities),
    }


# ---------------------------------------------------------------------------
# Tor
# ---------------------------------------------------------------------------


@router.get('/tor')
async def tor_status() -> dict[str, Any]:
    """Where the proxy is, and whether anything is behind it.

    Only that something is listening on the port. It does not build a circuit
    and cannot tell Tor from any other SOCKS server, and saying so is better
    than a tick that means less than it looks like it means.
    """
    proxy = tor_mod.resolve()
    ok, detail = await tor_mod.probe(proxy)
    return {
        'proxy': str(proxy),
        'listening': ok,
        'detail': detail,
        'using': [p['id'] for p in _providers() if p['tor']],
        'installed': _socks_installed(),
    }


def _socks_installed() -> bool:
    try:
        import aiohttp_socks  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _routes() -> dict[str, Any]:
    defaults = registry.defaults
    return {
        m.value: {'provider': r.provider, 'model': r.model, 'options': r.options}
        for m, r in defaults.routes.items()
    }


@router.get('/routes')
async def get_routes() -> dict[str, Any]:
    """What is chosen per modality, and what could be.

    Chat and embedding are listed like every other modality on purpose: they
    are one choice each, made independently, and a UI that shows them as one
    setting is a UI that will eventually send someone's whole memory to a
    vendor they picked for chat.
    """
    return {
        'routes': _routes(),
        'available': registry.capabilities(local_only=config.local_only),
        'local_only': config.local_only,
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
    previous = defaults.routes.get(modality)
    defaults.routes[modality] = Route(provider=body.provider, model=body.model, options=body.options)

    # Resolved immediately so a route that cannot work is rejected now rather
    # than at the start of someone's next call.
    try:
        registry.resolve(modality, defaults)
    except NoProviderError as exc:
        if previous is None:
            del defaults.routes[modality]
        else:
            defaults.routes[modality] = previous
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    registry.set_defaults(defaults)
    return {'ok': True, 'routes': _routes()}


# ---------------------------------------------------------------------------
# Perch
# ---------------------------------------------------------------------------


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

    Saved, through the same path the settings dialog uses. It used to register
    into the running process and nothing else, so a Perch connected here was
    gone at the next restart — and a connection that quietly disappears reads
    as a connection that never worked.
    """
    conn = connections.from_host(
        'perch',
        api_key=body.token,
        base_url=f'{body.scheme}://{body.host}',
    )
    result = await _save_perch(conn)
    result['capabilities'] = registry.capabilities(local_only=config.local_only)
    return result


async def _save_perch(conn: Connection) -> dict[str, Any]:
    """Probe a Perch host, check its token, then save and register it.

    In that order, and all three before anything is written. A Perch connection
    saved with a token Perch does not know would be reported as connected and
    then fail every request with a 401 somewhere far from this dialog; one
    saved against a host where nothing answers would offer five services that
    are not there.
    """
    from openmirror.providers import perch as perch_mod

    if conn.tor:
        raise HTTPException(
            status_code=400,
            detail=(
                'Perch connections cannot go over Tor from here yet — its services are '
                'reached directly, normally over the SSH tunnel Perch sets up to loopback. '
                'Leave Tor off for this one.'
            ),
        )

    cfg = perch_mod.from_connection(conn)
    available = await perch_mod.probe(cfg)
    if not any(available.values()):
        ports = ', '.join(str(p) for p in perch_mod.DEFAULT_PORTS.values())
        raise HTTPException(
            status_code=502,
            detail=(
                f'nothing answered at {cfg.host} on ports {ports}. If Perch runs on '
                'another machine, the address is wherever its tunnel lands — its '
                'Connect page shows it.'
            ),
        )

    ok, why = await perch_mod.check_token(cfg)
    if not ok:
        raise HTTPException(status_code=400, detail=why)

    store.put(conn)
    registered: list[str] = []
    if conn.enabled:
        registered = registry.register_perch(cfg, available, connection=conn)
    else:
        for provider_id in perch_mod.PROVIDER_IDS:
            registry.unregister(provider_id)
    return {
        'connection': conn.redacted(),
        'providers': _providers(),
        'registered': registered,
        'services': available,
    }


async def _register_env_perch() -> None:
    """Register the Perch the environment describes, if it describes one."""
    if not config.perch_host:
        return
    from openmirror.providers import perch as perch_mod

    cfg = perch_mod.PerchConfig(
        host=config.perch_host, token=config.perch_token, scheme=config.perch_scheme
    )
    registry.register_perch(cfg, await perch_mod.probe(cfg))


async def _check(impl: Any) -> None:
    """Ask the provider to prove its address and key, where listing cannot.

    Most adapters prove both by listing models, and now raise when the key is
    refused. A few list something local instead — ComfyUI's templates — and
    those carry a `check` that reaches the server, because a Test button that
    never leaves this machine is a green tick for anything.
    """
    check = getattr(impl, 'check', None)
    if check is not None:
        await check()


async def _test_registered(provider_id: str) -> dict[str, Any]:
    """Test a provider that is registered but was never stored."""
    try:
        described = registry.describe(provider_id)
    except NoProviderError as exc:
        raise HTTPException(status_code=404, detail=f'no such connection: {provider_id}') from exc

    started = asyncio.get_running_loop().time()
    try:
        impl = registry.impl(provider_id, Modality(described['modalities'][0]))
        await _check(impl)
        models = await impl.models()
    except Exception as exc:  # noqa: BLE001
        return {'ok': False, 'error': str(exc)[:500], 'tor': False}
    return {
        'ok': True,
        'models': len(models),
        'sample': [m.get('id') for m in models[:8]],
        'took_ms': int((asyncio.get_running_loop().time() - started) * 1000),
        'tor': False,
        'modalities': described['modalities'],
    }
