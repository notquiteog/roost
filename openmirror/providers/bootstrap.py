"""Turning configuration into a live registry at start-up.

Order matters here in one respect: Perch is probed before anything else and
registered first, so that an install with a GPU behind it defaults to its own
hardware rather than to whichever hosted key happened to be in the
environment. Choosing a paid remote model when a local one was sitting there
is the wrong default even when it is the faster one.
"""

from __future__ import annotations

import logging

from openmirror.config import Config
from openmirror.providers import connections
from openmirror.providers.anthropic import AnthropicProvider
from openmirror.providers.base import Modality, ProviderInfo
from openmirror.providers.connections import ConnectionStore
from openmirror.providers.ollama import OllamaProvider
from openmirror.providers.openai_compat import OpenAICompatProvider
from openmirror.providers.registry import ProviderRegistry, Route, RouteSet

log = logging.getLogger(__name__)


def _default_routes(cfg: Config) -> RouteSet:
    """The install's own choice per modality.

    Only chat and embedding are pinned from configuration, and they are pinned
    separately. Everything else falls through to "first provider that can do
    it", which prefers local hardware — see `ProviderRegistry.resolve`.

    Naming a provider that is not registered is left to fail at resolution
    rather than dropped here, because the failure names the missing provider
    and being quietly served by a different one is the outcome this design
    exists to prevent.
    """
    routes = RouteSet(local_only=cfg.local_only)
    if cfg.chat_provider:
        routes.routes[Modality.CHAT] = Route(provider=cfg.chat_provider, model=cfg.default_chat_model)
    if cfg.embed_provider:
        options = {'dimensions': cfg.embed_dimensions} if cfg.embed_dimensions else {}
        routes.routes[Modality.EMBEDDING] = Route(
            provider=cfg.embed_provider, model=cfg.embed_model, options=options
        )
    return routes


async def bootstrap(cfg: Config, registry: ProviderRegistry) -> list[str]:
    """Register every provider the configuration describes. Returns their ids."""
    registered: list[str] = []

    if cfg.perch_host:
        from openmirror.providers import perch as perch_mod

        pc = perch_mod.PerchConfig(host=cfg.perch_host, token=cfg.perch_token, scheme=cfg.perch_scheme)
        available = await perch_mod.probe(pc)
        ids = registry.register_perch(pc, available)
        registered += ids
        live = [s for s, ok in available.items() if ok]
        down = [s for s, ok in available.items() if not ok]
        log.info('perch at %s: %s up%s', cfg.perch_host, ', '.join(live) or 'nothing', f'; {", ".join(down)} off' if down else '')

    if cfg.ollama_url:
        impl = OllamaProvider(cfg.ollama_url, provider_id='ollama')
        registry.register(
            ProviderInfo(
                id='ollama',
                label='Ollama',
                modalities={Modality.CHAT, Modality.EMBEDDING},
                local=True,
                base_url=cfg.ollama_url,
            ),
            {Modality.CHAT: impl, Modality.EMBEDDING: impl},
        )
        registered.append('ollama')

    if cfg.anthropic_key:
        registry.register(
            ProviderInfo(
                id='anthropic',
                label='Anthropic',
                modalities={Modality.CHAT},
                local=False,
                base_url=cfg.anthropic_url,
            ),
            {Modality.CHAT: AnthropicProvider(cfg.anthropic_url, cfg.anthropic_key)},
        )
        registered.append('anthropic')

    if cfg.openai_key:
        from openmirror.providers.sora import SoraProvider

        impl = OpenAICompatProvider(cfg.openai_url, cfg.openai_key, provider_id='openai')
        # Video is a different API behind the same key, so it is a different
        # object — see the note in openmirror.providers.connections.build. An
        # install configured by environment gets Sora for the same reason one
        # configured through the UI does.
        video = SoraProvider(cfg.openai_url, cfg.openai_key, provider_id='openai', timeout=3600)
        modalities = {
            Modality.CHAT: impl,
            Modality.EMBEDDING: impl,
            Modality.STT: impl,
            Modality.TTS: impl,
            Modality.IMAGE: impl,
            Modality.VIDEO: video,
        }
        registry.register(
            ProviderInfo(
                id='openai',
                label='OpenAI',
                modalities=set(modalities),
                local=False,
                base_url=cfg.openai_url,
            ),
            modalities,
        )
        registered.append('openai')

    if cfg.openwebui_url:
        # Open WebUI is a provider in its own right: it serves OpenAI's shapes
        # at /api/v1, so every connection configured over there — including
        # ones openmirror has no adapter for — is reachable through this one entry.
        base = cfg.openwebui_url.rstrip('/')
        impl = OpenAICompatProvider(f'{base}/api/v1', cfg.openwebui_key, provider_id='openwebui')
        registry.register(
            ProviderInfo(
                id='openwebui',
                label='Open WebUI',
                modalities={Modality.CHAT, Modality.EMBEDDING},
                # Whether this is local depends on what Open WebUI is pointed
                # at, which cannot be known from here. Assumed remote, because
                # the failure that matters is calling something remote while
                # believing it local.
                local=False,
                base_url=base,
            ),
            {Modality.CHAT: impl, Modality.EMBEDDING: impl},
        )
        registered.append('openwebui')

    # Connections added through the UI, which outlive the process and can be
    # changed without restarting it. Registered last so that a connection
    # someone made deliberately replaces an environment-configured one with
    # the same id — the newer decision is the one they can see and edit.
    stored = ConnectionStore(cfg.connections_db).load()
    for conn in stored:
        if not conn.enabled:
            continue
        try:
            # `register` rather than `build`: a stored Perch is one connection
            # and several providers, and only registering it knows that.
            ids = await connections.register(conn, registry)
        except Exception as exc:  # noqa: BLE001
            # One bad connection must not stop the daemon. It is reported and
            # left unregistered, which is also what the UI shows.
            log.warning('connection %s could not be built: %s', conn.id, exc)
            continue
        for provider_id in ids:
            if provider_id not in registered:
                registered.append(provider_id)

    registry.set_defaults(_default_routes(cfg))

    if not registered:
        log.warning(
            'no providers configured — set PERCH_HOST, OLLAMA_BASE_URL, ANTHROPIC_API_KEY, '
            'OPENAI_API_KEY or OPENWEBUI_BASE_URL'
        )
    else:
        log.info('providers: %s', ', '.join(registered))

    return registered
