"""The openmirror server."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from openmirror.agent.manager import manager
from openmirror.config import config
from openmirror.providers.bootstrap import bootstrap
from openmirror.providers.registry import registry
from openmirror.routers import agent as agent_router
from openmirror.routers import autopilot as autopilot_router
from openmirror.routers import browser as browser_router
from openmirror.routers import mcp as mcp_router
from openmirror.routers import media as media_router
from openmirror.routers import memory as memory_router
from openmirror.routers import providers as providers_router
from openmirror.routers import realtime as realtime_router
from openmirror.routers import search as search_router
from openmirror.routers import voice as voice_router

log = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s',
    )
    await bootstrap(config, registry)
    manager.start_reaper()

    if config.mcp_enabled:
        from openmirror.mcp.manager import load_config
        from openmirror.mcp.manager import manager as mcp_manager

        servers = load_config(config.mcp_config)
        if servers:
            failures = await mcp_manager.start(servers)
            for name, why in failures.items():
                log.warning('mcp %s did not start: %s', name, why)
            live = len(mcp_manager.servers)
            if live:
                log.info('mcp: %d server(s), %d tool(s)', live, len(mcp_manager.tools()))

    # Media generation exists whenever something is registered that can make
    # a picture. No switch of its own: an install with no image provider
    # already cannot generate, and a second toggle saying so would only be a
    # way to have the feature off while it looks on.
    from openmirror.media.service import MediaService
    from openmirror.media.store import MediaStore

    media_router.service = MediaService(MediaStore(config.media_dir), registry)
    app.state.media = media_router.service

    if config.memory_enabled:
        from openmirror.memory.service import MemoryService
        from openmirror.memory.store import MemoryStore

        store = MemoryStore(config.memory_db)
        memory_router.service = MemoryService(store, registry, model=config.embed_model)
        app.state.memory = memory_router.service
        log.info('memory store at %s (each user opts in separately)', config.memory_db)

        # A changed embedding model has to be caught here, before anything can
        # be recalled: the old vectors do not fail against a new model, they
        # score, so an unnoticed change answers questions out of unrelated
        # geometry instead of going quiet.
        reset = memory_router.service.sync_embedder()
        if reset is not None:
            # Re-embedded in the background. It is a provider call per batch and
            # start-up must not wait on a network, so recall is briefly empty
            # and says so rather than blocking the server from listening.
            app.state.memory_reembed = asyncio.create_task(memory_router.service.reembed())
    else:
        app.state.memory = None
    # openmirror as an MCP server, for other clients. Built after memory and
    # media so it shares this process's services rather than opening a second
    # handle on the same SQLite file.
    if config.mcp_serve:
        loopback = config.host in ('127.0.0.1', 'localhost', '::1')
        if not loopback and not config.mcp_serve_token:
            # Refused rather than warned. This endpoint hands out a person's
            # memory, and an unauthenticated one on a network interface is a
            # different class of mistake from an unauthenticated agent endpoint.
            log.error(
                'OPENMIRROR_MCP_SERVE is on and this server is bound to %s, but no '
                'OPENMIRROR_MCP_SERVE_TOKEN is set. The MCP endpoint will not be served. '
                'Set a token, or bind to loopback.',
                config.host,
            )
        else:
            from openmirror.mcp.server import from_config

            mcp_router.server = await from_config(app=app)
            log.info(
                'serving MCP at /mcp (scope: %s, token: %s)',
                config.mcp_serve_scope, 'yes' if config.mcp_serve_token else 'no',
            )

    log.info('workspace: %s   approval: %s', config.workspace, config.approval_mode)
    if not config.auth_token and config.host not in ('127.0.0.1', 'localhost', '::1'):
        # Worth saying loudly: this process runs commands on the machine.
        log.warning('listening on %s with no OPENMIRROR_TOKEN set', config.host)
    yield

    # Cancelled rather than awaited: it is a rebuild of derived data and the
    # rows it did not reach are still pending, so the next start-up finishes
    # the job. Holding shutdown open for a provider call would not.
    task = getattr(app.state, 'memory_reembed', None)
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    manager.stop_reaper()
    await manager.close_all()

    from openmirror.mcp.manager import manager as mcp_manager

    await mcp_manager.stop()
    await mcp_router.shutdown()

    # The shared Chromium the headless search backend keeps, if anything
    # searched. Reaped on idle anyway; closed here so a stop does not wait for
    # the idle timer.
    from openmirror.agent.headless_search import shutdown as close_search_browser

    await close_search_browser()


app = FastAPI(title='openmirror', version='0.1.0', lifespan=lifespan)

app.include_router(agent_router.router)
app.include_router(agent_router.http)
app.include_router(voice_router.router)
app.include_router(providers_router.router)
app.include_router(memory_router.router)
app.include_router(media_router.router)
app.include_router(search_router.router)
app.include_router(browser_router.router)
# Mounted always, and 404s until start-up decides this install serves MCP —
# the same shape as the memory router, for the same reason: a feature that is
# off should look absent rather than broken.
app.include_router(mcp_router.router)
app.include_router(realtime_router.router)
app.include_router(autopilot_router.router)
app.include_router(autopilot_router.http)


STATIC = Path(__file__).parent / 'static'
app.mount('/static', StaticFiles(directory=STATIC), name='static')


@app.get('/')
async def index() -> FileResponse:
    return FileResponse(STATIC / 'index.html')


@app.get('/healthz')
async def healthz() -> dict[str, object]:
    return {
        'ok': True,
        'service': 'openmirror',
        'providers': [p.id for p in registry.list()],
        'capabilities': registry.capabilities(local_only=config.local_only),
    }


def main() -> None:
    import uvicorn

    uvicorn.run('openmirror.main:app', host=config.host, port=config.port, log_level=config.log_level.lower())


if __name__ == '__main__':
    main()
