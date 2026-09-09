"""Perch, as one thing you connect rather than five.

Perch is a set of separately authenticated ports on a machine with a GPU, and
each one sits on the port its own backend is conventionally found on: Ollama
11434, whisper.cpp 8080, Stable Diffusion 7860, ComfyUI 8188, Kokoro 8880.
That is deliberate on Perch's side and useful on ours — every one of them
speaks a shape something else already speaks, so a client written against the
real thing needs neither a new adapter nor a new port number.

What they do need is to stop being five separate pieces of configuration.
One host, one token; this expands that into the five providers underneath and
probes which are actually switched on, because on Perch each service is off
until someone asks for it and a UI offering video against a dead port is
worse than one that does not offer it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from roost.providers.base import Modality, ProviderInfo
from roost.providers.ollama import OllamaProvider
from roost.providers.openai_compat import OpenAICompatProvider

# Perch's defaults: each backend's own conventional port, not a block of
# consecutive ones. A tunnelled install may map them elsewhere, which is what
# the per-service port overrides on PerchConfig are for — and since these are
# popular numbers, the probe's identity check matters more here than it would
# with a private range.
DEFAULT_PORTS = {
    'chat': 11434,   # Ollama
    'voice': 8080,   # whisper.cpp
    'image': 7860,   # Stable Diffusion / A1111
    'video': 8188,   # ComfyUI
    'audio': 8880,   # Kokoro
}

# Which of our modalities each Perch service answers for.
SERVICE_MODALITIES: dict[str, set[Modality]] = {
    'chat': {Modality.CHAT, Modality.EMBEDDING},
    'voice': {Modality.STT},
    'audio': {Modality.TTS},
    'image': {Modality.IMAGE},
    'video': {Modality.VIDEO},
}


@dataclass(slots=True)
class PerchConfig:
    host: str = '127.0.0.1'
    token: str = ''
    scheme: str = 'http'
    ports: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_PORTS))
    # Perch is normally reached over an SSH reverse tunnel, so the default
    # assumption is loopback and the default posture is that this is local
    # hardware the operator owns.
    local: bool = True

    def url(self, service: str) -> str:
        return f'{self.scheme}://{self.host}:{self.ports.get(service, DEFAULT_PORTS[service])}'


async def probe(cfg: PerchConfig, timeout: float = 3.0) -> dict[str, bool]:  # noqa: ASYNC109 - passed to aiohttp.ClientTimeout, which is the right mechanism
    """Which Perch services are up.

    `/healthz` is unauthenticated on every Perch service and returns the
    service id, so this says nothing about whether the token is any good —
    only whether something is listening. That is the right question here: a
    bad token is a clear 401 later, whereas a service that is switched off
    should simply not appear.
    """

    async def one(service: str) -> tuple[str, bool]:
        try:
            # transport-exempt: a Perch connection has no Tor toggle to
            # honour. PerchConfig is its own shape rather than a stored
            # Connection, so nothing here has ever promised to route through a
            # proxy — and Perch is normally reached over an SSH tunnel to
            # loopback, where there is nothing for Tor to hide. If that
            # changes, this probe and all five providers built below need a
            # transport, not just whichever one someone remembers.
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(f'{cfg.url(service)}/healthz') as resp:
                    if resp.status != 200:
                        return service, False
                    body = await resp.json()
                    # Guards against something unrelated holding the port.
                    return service, body.get('service') == 'perch'
        except (TimeoutError, aiohttp.ClientError, ValueError):
            return service, False

    results = await asyncio.gather(*(one(s) for s in DEFAULT_PORTS))
    return dict(results)


def build(cfg: PerchConfig, available: dict[str, bool] | None = None) -> dict[Modality, Any]:
    """Expand one Perch connection into a provider per modality.

    Services reported down are left out rather than registered and allowed to
    fail at call time, so that "what can this install do" is answerable
    without making five requests.
    """
    up = available or {s: True for s in DEFAULT_PORTS}
    out: dict[Modality, Any] = {}

    if up.get('chat'):
        # Ollama's native API rather than the OpenAI compatibility layer, since
        # Perch exposes both and the native one also reports what is pulled and
        # what is resident.
        out[Modality.CHAT] = OllamaProvider(cfg.url('chat'), cfg.token, provider_id='perch:chat')
        out[Modality.EMBEDDING] = out[Modality.CHAT]

    if up.get('voice'):
        # whisper.cpp, served on OpenAI's transcription path.
        out[Modality.STT] = OpenAICompatProvider(f"{cfg.url('voice')}/v1", cfg.token, provider_id='perch:voice')

    if up.get('audio'):
        # Kokoro, on OpenAI's speech path.
        out[Modality.TTS] = OpenAICompatProvider(f"{cfg.url('audio')}/v1", cfg.token, provider_id='perch:audio')

    if up.get('image'):
        from roost.providers.a1111 import A1111Provider

        out[Modality.IMAGE] = A1111Provider(cfg.url('image'), cfg.token, provider_id='perch:image')

    if up.get('video'):
        from roost.providers.comfyui import ComfyUIProvider

        out[Modality.VIDEO] = ComfyUIProvider(cfg.url('video'), cfg.token, provider_id='perch:video')

    return out


def describe(cfg: PerchConfig, available: dict[str, bool]) -> list[ProviderInfo]:
    """One entry per live service, for the connection screen."""
    out: list[ProviderInfo] = []
    for service, ok in available.items():
        if not ok:
            continue
        out.append(
            ProviderInfo(
                id=f'perch:{service}',
                label=f'Perch · {service}',
                modalities=SERVICE_MODALITIES[service],
                local=cfg.local,
                base_url=cfg.url(service),
            )
        )
    return out
