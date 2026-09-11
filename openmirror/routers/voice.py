"""The voice call over a websocket.

Audio and control share one socket, split by frame type: binary is audio,
text is JSON. That keeps the audio path free of base64, which at 16 kHz is a
third more bytes for nothing.

Outbound audio carries the utterance it belongs to in a 12-byte ASCII prefix.
A client needs it: when a barge-in cancels an utterance, whatever it has
already buffered has to be dropped, and a raw stream gives it no way to know
which samples those were.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from openmirror.config import config
from openmirror.protocol.voice import AudioFormat, VoiceReady
from openmirror.providers.base import Modality
from openmirror.providers.registry import NoProviderError, Route, RouteSet, pick_model, registry
from openmirror.voice.pipeline import VoiceConfig, VoiceSession
from openmirror.voice.vad import VadConfig

log = logging.getLogger(__name__)

router = APIRouter()

UTTERANCE_PREFIX = 12


@router.websocket('/ws/voice')
async def voice_socket(ws: WebSocket, token: str | None = Query(None)) -> None:
    if config.auth_token and token != config.auth_token:
        await ws.close(code=4401, reason='unauthorised')
        return

    await ws.accept()

    # The first message must be voice.start: it carries the per-call provider
    # choices, and building the pipeline before knowing them would mean
    # tearing it down again.
    try:
        opening = await asyncio.wait_for(ws.receive_json(), timeout=30)
    except (TimeoutError, WebSocketDisconnect):
        await ws.close(code=4408, reason='no voice.start')
        return

    if opening.get('type') != 'voice.start':
        await ws.send_json({'type': 'voice.error', 'message': 'expected voice.start', 'fatal': True})
        await ws.close(code=4400)
        return

    def pick(modality: Modality, named: str | None):
        routes = RouteSet(routes={modality: Route(provider=named)}) if named else None
        return registry.resolve(modality, routes)

    try:
        stt, stt_route, stt_info = pick(Modality.STT, opening.get('stt'))
        tts, tts_route, tts_info = pick(Modality.TTS, opening.get('tts'))
        llm, llm_route, llm_info = pick(Modality.CHAT, opening.get('llm'))
    except NoProviderError as exc:
        await ws.send_json({'type': 'voice.error', 'message': str(exc), 'fatal': True})
        await ws.close(code=4404)
        return

    fmt = AudioFormat(**(opening.get('format') or {}))
    llm_model = opening.get('model') or llm_route.model or config.default_chat_model
    if not llm_model:
        # Same rule as the agent socket: not the first model, the first one
        # that can actually hold a conversation. Tools are not required here —
        # a voice call without an agent attached only has to talk.
        models = await llm.models()
        llm_model = pick_model(models, Modality.CHAT)

    cfg = VoiceConfig(
        sample_rate=fmt.sample_rate,
        stt_model=stt_route.model or config.default_stt_model,
        tts_model=tts_route.model or config.default_tts_model,
        tts_voice=opening.get('voice') or tts_route.options.get('voice') or config.default_tts_voice,
        llm_model=llm_model,
        language=opening.get('language'),
        vad=VadConfig(sample_rate=fmt.sample_rate),
    )

    # A voice call that can act on the machine is opt-in per call: consenting
    # to be listened to is not the same as consenting to have commands run.
    agent = None
    if opening.get('agent_session_id'):
        from openmirror.agent.approval import Mode
        from openmirror.agent.runtime import build_session

        agent = build_session(
            root=config.workspace,
            provider=llm,
            model=llm_model,
            mode=Mode(config.approval_mode),
            session_id=opening['agent_session_id'],
        )
        await agent.start()

    # Sends are best-effort. The common way to end a call is for the client to
    # vanish, and the teardown that follows still emits — a cancelled synthesis
    # raises `speech.cancelled` — so a send on a dead socket is expected rather
    # than exceptional. Letting it raise turns a normal hang-up into a
    # traceback and skips the rest of the cleanup.
    async def emit_event(event) -> None:
        try:
            await ws.send_json(event.model_dump(mode='json'))
        except (WebSocketDisconnect, RuntimeError):
            pass

    async def emit_audio(utterance_id: str, chunk: bytes) -> None:
        try:
            await ws.send_bytes(utterance_id.ljust(UTTERANCE_PREFIX)[:UTTERANCE_PREFIX].encode('ascii') + chunk)
        except (WebSocketDisconnect, RuntimeError):
            pass

    session = VoiceSession(
        stt=stt, tts=tts, llm=llm, config=cfg, emit_event=emit_event, emit_audio=emit_audio, agent=agent
    )

    await ws.send_json(
        VoiceReady(
            format=fmt,
            output_format=AudioFormat(sample_rate=getattr(tts, 'output_sample_rate', 24_000)),
            stt=stt_info.id,
            tts=tts_info.id,
            llm=llm_info.id,
        ).model_dump(mode='json')
    )

    try:
        while True:
            message = await ws.receive()

            if message.get('type') == 'websocket.disconnect':
                break

            if (audio := message.get('bytes')) is not None:
                await session.feed(audio)
                continue

            if (text := message.get('text')) is None:
                continue

            import json

            try:
                command = json.loads(text)
            except json.JSONDecodeError:
                continue

            kind = command.get('type')
            if kind == 'voice.stop':
                break
            elif kind == 'voice.interrupt':
                await session.interrupt()
            elif kind == 'voice.mute':
                await session.mute(bool(command.get('muted', True)))
            elif kind == 'voice.text':
                await session.say(command.get('text', ''))

    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.exception('voice socket failed')
    finally:
        await session.close()
