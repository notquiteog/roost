"""The agent over HTTP and a websocket.

Sessions are created over HTTP and *attached to* over the websocket, rather
than being owned by it. The distinction is the whole point: a socket is a
window onto work that is happening anyway, so closing it pauses your view and
nothing else. Reattaching passes the last sequence number seen and the session
replays what was missed.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from roost.agent.approval import Mode
from roost.agent.manager import manager
from roost.config import config
from roost.providers.base import Modality
from roost.providers.registry import NoProviderError, Route, RouteSet, registry
from roost.routers import memory as memory_router

log = logging.getLogger(__name__)

router = APIRouter()
http = APIRouter(prefix='/api/sessions')


async def _resolve_chat(provider: str | None, model: str | None) -> tuple[object, str, str]:
    """Pick the chat provider and a concrete model. Raises NoProviderError."""
    routes = RouteSet(routes={Modality.CHAT: Route(provider=provider, model=model or '')}) if provider else None
    impl, route, info = registry.resolve(Modality.CHAT, routes)

    chosen = model or route.model or config.default_chat_model
    if not chosen:
        # Asking the provider is better than guessing a name: a local install
        # has whatever happens to be pulled, and that is not knowable from here.
        available = await impl.models()
        if not available:
            raise NoProviderError(f'{info.id} reports no models')
        chosen = available[0]['id']
    return impl, chosen, info.id


class CreateSession(BaseModel):
    root: str | None = None
    model: str | None = None
    provider: str | None = None
    mode: str | None = None
    title: str = ''


@http.post('')
async def create_session(body: CreateSession) -> dict[str, object]:
    try:
        impl, model, provider_id = await _resolve_chat(body.provider, body.model)
    except NoProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        session = await manager.create(
            root=body.root or config.workspace,
            provider=impl,
            model=model,
            mode=Mode(body.mode or config.approval_mode),
            title=body.title,
            memory=memory_router.service,
            user_id=config.default_user,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        'id': session.id,
        'title': session.title,
        'root': str(session.root),
        'model': model,
        'provider': provider_id,
        'policy': session.policy.describe(),
        'tools': sorted(session.tools),
    }


@http.get('')
async def list_sessions() -> dict[str, object]:
    return {'sessions': manager.list()}


@http.get('/{session_id}/checkpoints')
async def list_checkpoints(session_id: str) -> dict[str, object]:
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='no such session')
    if session.checkpoints is None:
        return {'enabled': False, 'checkpoints': []}
    return {'enabled': True, 'checkpoints': session.checkpoints.describe()}


class Restore(BaseModel):
    checkpoint: str


@http.post('/{session_id}/restore')
async def restore_checkpoint(session_id: str, body: Restore) -> dict[str, object]:
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='no such session')
    if session.checkpoints is None:
        raise HTTPException(status_code=404, detail='checkpoints are not enabled')
    if session.busy:
        # Rewinding under a running turn would race the very writes it is
        # trying to undo.
        raise HTTPException(status_code=409, detail='the session is working — interrupt it first')

    try:
        report = session.checkpoints.restore(body.checkpoint)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        'restored': report.restored,
        'deleted': report.deleted,
        'skipped': report.skipped,
        'changed_since': report.changed_since,
    }


@http.delete('/{session_id}')
async def close_session(session_id: str) -> dict[str, bool]:
    if not await manager.close(session_id):
        raise HTTPException(status_code=404, detail='no such session')
    return {'ok': True}


@router.websocket('/ws/agent')
async def agent_socket(
    ws: WebSocket,
    session: str | None = Query(None, description='Session to attach to.'),
    since: int = Query(0, description='Last event sequence number already seen.'),
    root: str | None = Query(None),
    model: str | None = Query(None),
    provider: str | None = Query(None),
    mode: str | None = Query(None),
    token: str | None = Query(None),
) -> None:
    if config.auth_token and token != config.auth_token:
        # Refused before accepting, so an unauthenticated caller never holds a socket.
        await ws.close(code=4401, reason='unauthorised')
        return

    await ws.accept()

    agent = manager.get(session) if session else None

    if session and agent is None:
        # Almost always a page that outlived the daemon: sessions live in
        # memory, so restarting the daemon invalidates every id a client is
        # holding. Said plainly, because "no such session" reads like data loss
        # and is usually just a restart.
        await ws.send_json({
            'type': 'error',
            'message': (
                f'Session {session} is gone — the daemon has been restarted since this page '
                'last connected. Sessions do not survive a restart. Pick another from the '
                'sidebar, or start a new one.'
            ),
            'retryable': False,
        })
        await ws.close(code=4404)
        return

    if agent is None:
        # No session named: make one, so a trivial client stays trivial.
        try:
            impl, chosen, _ = await _resolve_chat(provider, model)
            agent = await manager.create(
                root=root or config.workspace,
                provider=impl,
                model=chosen,
                mode=Mode(mode or config.approval_mode),
                memory=memory_router.service,
                user_id=config.default_user,
            )
        except (NoProviderError, ValueError) as exc:
            await ws.send_json({'type': 'error', 'message': str(exc), 'retryable': False})
            await ws.close(code=4404)
            return

    async def pump_out() -> None:
        try:
            async for event in agent.events(since=since):
                await ws.send_json(event.model_dump(mode='json'))
        except (WebSocketDisconnect, RuntimeError):
            pass

    outbound = asyncio.create_task(pump_out())

    try:
        while True:
            command = await ws.receive_json()
            kind = command.get('type')

            if kind == 'turn.submit':
                try:
                    agent.submit(command.get('text', ''), command.get('attachments') or [])
                except RuntimeError as exc:
                    await ws.send_json({'type': 'error', 'message': str(exc), 'retryable': False})
            elif kind == 'tool.approve':
                agent.approve(command.get('call_id', ''), bool(command.get('remember')))
            elif kind == 'tool.deny':
                agent.deny(command.get('call_id', ''), command.get('reason', ''))
            elif kind == 'question.answer':
                agent.answer(command.get('question_id', ''), command.get('answer', ''))
            elif kind == 'policy.set':
                # The approval mode is a live control, not a property of the
                # session's birth: the thing you learn while watching an agent
                # work is exactly how much you trust it. It takes effect from
                # the next decision — a call already in flight was decided
                # under the old rule, and re-deciding it retroactively would
                # be a lie about what ran.
                try:
                    agent.policy.mode = Mode(command.get('mode', ''))
                except ValueError:
                    await ws.send_json({
                        'type': 'error',
                        'message': f'unknown approval mode: {command.get("mode")!r}',
                        'retryable': False,
                    })
                else:
                    log.info('session %s: approval mode set to %s', agent.id, agent.policy.mode.value)
                    await ws.send_json({
                        'type': 'policy.changed',
                        'mode': agent.policy.mode.value,
                        'policy': agent.policy.describe(),
                    })
            elif kind == 'turn.interrupt':
                agent.interrupt()
            elif kind == 'session.close':
                await manager.close(agent.id)
                break
            elif kind == 'ping':
                await ws.send_json({'type': 'pong', 'seq': agent.seq})
            else:
                await ws.send_json({'type': 'error', 'message': f'unknown command: {kind}', 'retryable': False})

    except WebSocketDisconnect:
        # Detach only. The session, and any turn it is running, carries on.
        log.info('session %s: client detached at seq %d', agent.id, agent.seq)
    except Exception:  # noqa: BLE001
        log.exception('agent socket failed')
    finally:
        outbound.cancel()
