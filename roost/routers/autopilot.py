"""Starting an autonomous run, and watching it.

Two endpoints and a deliberate split. Starting is HTTP, because it is a
decision with consequences and wants a clear answer — including a refusal,
with a reason, when the machine cannot give the agent a screen of its own.
Watching is a websocket carrying JPEG frames, separate from the agent socket
that carries what it is doing.

The split is what makes both usable. The narration is small, ordered and
replayable to a client that reattaches; the video is large, continuous and
worthless five seconds later. Putting them on one channel would mean either
storing megabytes of stale pictures for replay, or losing the narration when
someone reloads the page.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from roost.agent.approval import Mode
from roost.agent.autopilot import AUTOPILOT_PROMPT, Run, autopilot, frames
from roost.agent.manager import manager
from roost.config import config
from roost.providers.registry import NoProviderError
from roost.routers import media as media_router
from roost.routers import memory as memory_router

log = logging.getLogger(__name__)

router = APIRouter()
http = APIRouter(prefix='/api/autopilot')


class Start(BaseModel):
    goal: str
    root: str | None = None
    provider: str | None = None
    model: str | None = None
    # `trusted` is the honest default for a run nobody is answering: it runs
    # commands and network calls unattended and still stops for anything
    # destructive, for money and for secrets. Naming it here rather than
    # inheriting the install default is deliberate — the install default is
    # tuned for someone sitting in front of it.
    mode: str = 'trusted'
    # Force the shared screen. Off by default and never chosen automatically:
    # it means taking the person's mouse, and that should be something they
    # asked for in words.
    share_screen: bool = False


@http.post('')
async def start(body: Start) -> dict[str, object]:
    if not body.goal.strip():
        raise HTTPException(status_code=400, detail='a goal is required')

    if not config.desktop_enabled:
        raise HTTPException(
            status_code=503,
            detail='autopilot drives the screen, and desktop control is off on this install. '
                   'Set ROOST_DESKTOP=true.',
        )

    from roost.routers.agent import _resolve_chat

    try:
        impl, model, provider_id = await _resolve_chat(body.provider, body.model)
    except NoProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # The stage is decided before anything starts, so that "this will use your
    # real mouse" is something the person is told rather than something they
    # discover when the cursor moves.
    stage_cfg = _stage_config(share_screen=body.share_screen)

    try:
        session = await manager.create(
            root=body.root or config.workspace,
            provider=impl,
            model=model,
            mode=Mode(body.mode),
            title=body.goal[:60],
            memory=memory_router.service,
            media=media_router.service,
            user_id=config.default_user,
            cfg=stage_cfg,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if session.stage is None:
        await manager.close(session.id)
        raise HTTPException(
            status_code=503,
            detail='no screen could be made for it to work on. Install Xvfb for a display of '
                   'its own, or pass share_screen to let it drive the one you are looking at.',
        )

    if session.stage.shares_pointer and not body.share_screen:
        await manager.close(session.id)
        raise HTTPException(
            status_code=409,
            detail='the only screen available is the one you are looking at, so running this '
                   'would take your mouse. Install Xvfb to give it a display of its own, or '
                   'pass share_screen if sharing the pointer is what you want.',
        )

    run = autopilot.start(
        Run(
            id=session.id,
            goal=body.goal,
            session_id=session.id,
            shares_pointer=session.stage.shares_pointer,
            stage=session.stage.kind,
        )
    )

    # The prompt is what makes this autonomous, not the policy. It is appended
    # to the session's own so the project's conventions and the working root
    # still apply.
    session.system_prompt = f'{session.system_prompt}\n\n## Working unattended\n\n{AUTOPILOT_PROMPT}'
    session.submit(f'{body.goal}\n\n(You are running unattended. Begin.)')

    return {
        **run.to_json(),
        'model': model,
        'provider': provider_id,
        'policy': session.policy.describe(),
        'screen': session.stage.describe(),
    }


def _stage_config(*, share_screen: bool):
    """The config an autopilot session is built with.

    A shallow copy with the stage forced, rather than mutating the global: two
    runs at once must not be able to change each other's screen, and a run
    that asked for the shared screen must not leave the install set that way
    for the next one.
    """
    import copy

    cfg = copy.copy(config)
    cfg.desktop_enabled = True
    cfg.desktop_stage = 'shared' if share_screen else 'virtual'
    return cfg


@http.get('')
async def list_runs() -> dict[str, object]:
    live = []
    for run in autopilot.list():
        session = manager.get(str(run['session']))
        if session is None or session.closed:
            continue
        live.append({**run, 'busy': session.busy, 'waiting_on': session.waiting_on})
    return {'runs': live}


@http.delete('/{run_id}')
async def stop(run_id: str) -> dict[str, bool]:
    run = autopilot.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail='no such run')
    session = manager.get(run.session_id)
    if session is not None:
        session.interrupt()
        await manager.close(run.session_id, 'autopilot stopped')
    autopilot.forget(run_id)
    return {'ok': True}


@router.websocket('/ws/autopilot')
async def watch(
    ws: WebSocket,
    run: str = Query(..., description='The run to watch.'),
    fps: float = Query(1.4, ge=0.2, le=5.0),
    token: str | None = Query(None),
) -> None:
    if config.auth_token and token != config.auth_token:
        await ws.close(code=4401, reason='unauthorised')
        return

    await ws.accept()

    entry = autopilot.get(run)
    session = manager.get(entry.session_id) if entry else None
    if entry is None or session is None or session.stage is None:
        await ws.send_json({'type': 'error', 'message': 'no such run, or it has finished'})
        await ws.close(code=4404)
        return

    entry.watchers += 1
    await ws.send_json({'type': 'screen.ready', **entry.to_json(), 'screen': session.stage.describe()})

    async def keep_reading() -> None:
        """Drain whatever the client sends, so a close is noticed promptly.

        Without this the send loop only discovers a closed socket when a frame
        fails, which on a slow interval is a second of capturing for nobody.
        """
        try:
            while True:
                await ws.receive()
        except (WebSocketDisconnect, RuntimeError):
            pass

    reader = asyncio.create_task(keep_reading())

    try:
        async for frame in frames(session.stage, interval=1.0 / fps):
            if reader.done():
                break
            await ws.send_bytes(frame)
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:  # noqa: BLE001
        log.exception('autopilot frame stream failed')
    finally:
        entry.watchers = max(0, entry.watchers - 1)
        reader.cancel()
