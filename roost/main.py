"""The Roost server."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from roost.agent.manager import manager
from roost.config import config
from roost.providers.bootstrap import bootstrap
from roost.providers.registry import registry
from roost.routers import agent as agent_router
from roost.routers import autopilot as autopilot_router
from roost.routers import media as media_router
from roost.routers import memory as memory_router
from roost.routers import providers as providers_router
from roost.routers import realtime as realtime_router
from roost.routers import search as search_router
from roost.routers import voice as voice_router

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
        from roost.mcp.manager import load_config
        from roost.mcp.manager import manager as mcp_manager

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
    from roost.media.service import MediaService
    from roost.media.store import MediaStore

    media_router.service = MediaService(MediaStore(config.media_dir), registry)
    app.state.media = media_router.service

    if config.memory_enabled:
        from roost.memory.service import MemoryService
        from roost.memory.store import MemoryStore

        store = MemoryStore(config.memory_db)
        memory_router.service = MemoryService(store, registry, model=config.embed_model)
        app.state.memory = memory_router.service
        log.info('memory store at %s (each user opts in separately)', config.memory_db)
    else:
        app.state.memory = None
    log.info('workspace: %s   approval: %s', config.workspace, config.approval_mode)
    if not config.auth_token and config.host not in ('127.0.0.1', 'localhost', '::1'):
        # Worth saying loudly: this process runs commands on the machine.
        log.warning('listening on %s with no ROOST_TOKEN set', config.host)
    yield

    manager.stop_reaper()
    await manager.close_all()

    from roost.mcp.manager import manager as mcp_manager

    await mcp_manager.stop()


app = FastAPI(title='Roost', version='0.1.0', lifespan=lifespan)

app.include_router(agent_router.router)
app.include_router(agent_router.http)
app.include_router(voice_router.router)
app.include_router(providers_router.router)
app.include_router(memory_router.router)
app.include_router(media_router.router)
app.include_router(search_router.router)
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
        'service': 'roost',
        'providers': [p.id for p in registry.list()],
        'capabilities': registry.capabilities(local_only=config.local_only),
    }


def main() -> None:
    import uvicorn

    uvicorn.run('roost.main:app', host=config.host, port=config.port, log_level=config.log_level.lower())


if __name__ == '__main__':
    main()
