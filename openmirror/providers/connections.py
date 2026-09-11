"""Provider connections that outlive the process, and the Tor toggle on each.

Before this, a provider was whatever the environment said at start-up. That is
right for an install managed by a file, and useless for the case this exists
for: someone adding Groq at four in the afternoon, discovering their key is
wrong, and fixing it without editing `.env` and restarting a daemon that is in
the middle of a build.

So a connection is data. It is written to disk, rebuilt into a live provider
on start-up, and can be added, edited and removed while the server runs.

**The key is a secret in a file, and treated like one.** The file is written
0600 and re-read rather than cached, and no API response ever carries a key
back out — only whether one is set. Nothing is encrypted, and pretending
otherwise would be worse than saying so: anything that can read the file can
read the key, exactly as with `.env`.

**Tor is per connection and never inferred.** An install may have Arti running
for something else entirely; routing a model's traffic through it because it
happened to be there would be a decision about where someone's prompts go,
made on their behalf. Turning it on and having the proxy be absent is an
error at the point of use, never a quiet direct connection — see
`openmirror.net.tor`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openmirror.net.transport import Transport
from openmirror.providers.base import Modality, ProviderInfo
from openmirror.providers.catalog import HOSTS_BY_ID

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Connection:
    """One configured endpoint."""

    id: str
    label: str
    adapter: str
    base_url: str
    api_key: str = ''
    # What this connection is offered for. Stored rather than derived from the
    # adapter, because one adapter serves several kinds of service and a
    # gateway may serve more than its vendor documents.
    modalities: list[str] = field(default_factory=list)
    # A claim about topology: is the hardware behind this yours? It decides
    # whether `local_only` will accept the connection, so it is asked rather
    # than guessed — a remote Ollama looks exactly like a local one from here.
    local: bool = False
    tor: bool = False
    enabled: bool = True
    # Where the SOCKS proxy is for this connection, when it is not where the
    # environment says. Empty is the normal case.
    tor_host: str = ''
    tor_port: int = 0
    # Ties a connection back to the preset it was made from, so the UI can
    # show the right notes and the right embedding hints.
    host_id: str = ''

    def transport(self, timeout: int = 600) -> Transport:
        return Transport(timeout=timeout, tor=self.tor, tor_host=self.tor_host, tor_port=self.tor_port)

    def redacted(self) -> dict[str, Any]:
        """Safe to send to a client: everything except the key itself."""
        data = asdict(self)
        data.pop('api_key', None)
        data['has_key'] = bool(self.api_key)
        return data


class ConnectionStore:
    """The connections file. Read on every access, written whole."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> list[Connection]:
        if not self.path.is_file():
            return []
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            # Never fatal. A corrupt connections file must not stop a daemon
            # whose environment-configured providers are perfectly fine.
            log.warning('could not read %s (%s); no stored connections', self.path, exc)
            return []

        out: list[Connection] = []
        for entry in raw.get('connections', []):
            try:
                out.append(Connection(**entry))
            except TypeError:
                # A field this version does not know: skip the one entry
                # rather than the file, so a downgrade loses one connection.
                log.warning('skipping a connection this version cannot read: %s', entry.get('id'))
        return out

    def save(self, connections: list[Connection]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({'version': 1, 'connections': [asdict(c) for c in connections]}, indent=2)
        # Written to a temporary file in the same directory and renamed, so a
        # crash halfway cannot leave a half-written file where the keys were.
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(payload)
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    def put(self, connection: Connection) -> list[Connection]:
        """Add or replace by id. Returns the new set."""
        current = [c for c in self.load() if c.id != connection.id]
        current.append(connection)
        self.save(current)
        return current

    def remove(self, connection_id: str) -> bool:
        current = self.load()
        kept = [c for c in current if c.id != connection_id]
        if len(kept) == len(current):
            return False
        self.save(kept)
        return True

    def get(self, connection_id: str) -> Connection | None:
        return next((c for c in self.load() if c.id == connection_id), None)


# ---------------------------------------------------------------------------
# Building live providers
# ---------------------------------------------------------------------------


def build(conn: Connection) -> tuple[ProviderInfo, dict[Modality, Any]]:
    """Turn a stored connection into something the registry can route to.

    The return is a mapping **per modality**, and that is the part that earns
    its keep. One object usually serves several — the OpenAI adapter does
    chat, embeddings, speech both ways and images — but several services need
    a different object for different modalities, and the two reasons are worth
    naming because both produced a real failure before this was per-modality:

    **A different API behind the same key.** OpenAI's video endpoint is not
    OpenAI-shaped chat; Google's Veo is not Google's embeddings. Folding those
    into one adapter would advertise Sora on every OpenAI-*shaped* gateway —
    Groq, Together, LM Studio — none of which serves it.

    **The same ids meaning different things.** Kling and MiniMax use one model
    name for an image model and a video model, so the endpoint cannot be
    chosen from the id and must be fixed when the object is built. Hence two
    instances of one class, which is why this returns a mapping rather than a
    single impl.

    Whichever it is, the Tor toggle covers all of them: every object here is
    handed the same transport.
    """
    transport = conn.transport()
    wanted = {Modality(m) for m in conn.modalities if m in set(Modality)}
    ident = conn.id
    # Media generation is minutes, not seconds. The transport's timeout covers
    # one HTTP call — a poll, which is fast — and this covers the whole job.
    long_running = 3600

    def one(cls: Any, **kw: Any) -> Any:
        return cls(conn.base_url, conn.api_key, provider_id=ident, transport=transport, **kw)

    impls: dict[Modality, Any] = {}

    if conn.adapter == 'anthropic':
        from openmirror.providers.anthropic import AnthropicProvider

        impls = {Modality.CHAT: one(AnthropicProvider)}
    elif conn.adapter == 'ollama':
        from openmirror.providers.ollama import OllamaProvider

        shared = one(OllamaProvider)
        impls = {Modality.CHAT: shared, Modality.EMBEDDING: shared}
    elif conn.adapter == 'voyage':
        from openmirror.providers.voyage import VoyageProvider

        impls = {Modality.EMBEDDING: one(VoyageProvider)}
    elif conn.adapter == 'google':
        from openmirror.providers.google import GoogleProvider
        from openmirror.providers.google_media import GoogleMediaProvider
        from openmirror.providers.openai_compat import OpenAICompatProvider

        impls = {
            # Chat through Google's OpenAI-compatible endpoint, which sits under
            # the same base URL and takes the same key. The catalogue has always
            # offered this host for chat, and without this entry the narrowing
            # below dropped it silently — a Gemini connection could embed and
            # draw and could not be talked to.
            Modality.CHAT: OpenAICompatProvider(
                f'{conn.base_url.rstrip("/")}/openai', conn.api_key, provider_id=ident, transport=transport
            ),
            Modality.EMBEDDING: one(GoogleProvider),
            Modality.IMAGE: one(GoogleMediaProvider, kind='image', timeout=long_running),
            Modality.VIDEO: one(GoogleMediaProvider, kind='video', timeout=long_running),
        }
    elif conn.adapter == 'a1111':
        from openmirror.providers.a1111 import A1111Provider

        impls = {Modality.IMAGE: one(A1111Provider)}
    elif conn.adapter == 'comfyui':
        from openmirror.providers.comfyui import ComfyUIProvider

        shared = one(ComfyUIProvider)
        impls = {Modality.IMAGE: shared, Modality.VIDEO: shared}
    elif conn.adapter in ('replicate', 'fal', 'stability', 'runway', 'luma'):
        from openmirror.providers.fal import FalProvider
        from openmirror.providers.luma import LumaProvider
        from openmirror.providers.replicate import ReplicateProvider
        from openmirror.providers.runway import RunwayProvider
        from openmirror.providers.stability import StabilityProvider

        classes = {'replicate': ReplicateProvider, 'fal': FalProvider, 'stability': StabilityProvider,
                   'runway': RunwayProvider, 'luma': LumaProvider}
        shared = one(classes[conn.adapter], timeout=long_running)
        impls = {Modality.IMAGE: shared, Modality.VIDEO: shared}
    elif conn.adapter in ('kling', 'minimax'):
        from openmirror.providers.kling import KlingProvider
        from openmirror.providers.minimax import MiniMaxProvider

        cls = KlingProvider if conn.adapter == 'kling' else MiniMaxProvider
        impls = {
            Modality.IMAGE: one(cls, kind='image', timeout=long_running),
            Modality.VIDEO: one(cls, kind='video', timeout=long_running),
        }
    elif conn.adapter == 'openrouter':
        from openmirror.providers.openrouter import OpenRouterImages, OpenRouterProvider, OpenRouterVideo

        # An instance per modality even for the four OpenAI-shaped ones,
        # because each lists its models at a different address — one list for
        # all four is four hundred chat models in the dictation picker.
        impls = {
            modality: one(OpenRouterProvider, kind=modality)
            for modality in (Modality.CHAT, Modality.EMBEDDING, Modality.STT, Modality.TTS)
        }
        impls[Modality.IMAGE] = one(OpenRouterImages)
        impls[Modality.VIDEO] = one(OpenRouterVideo, timeout=long_running)
    elif conn.adapter == 'bfl':
        from openmirror.providers.bfl import BFLProvider

        impls = {Modality.IMAGE: one(BFLProvider, timeout=long_running)}
    elif conn.adapter == 'ideogram':
        from openmirror.providers.ideogram import IdeogramProvider

        impls = {Modality.IMAGE: one(IdeogramProvider)}
    elif conn.adapter == 'perch':
        # Every service Perch could offer, assumed up. That is what building
        # means everywhere else in this function — the shape of a connection,
        # without touching the network. Registering one goes through
        # `register` below, which probes first and gives each live service its
        # own provider id; this is for the callers that only want the shape.
        # No transport is handed over because a Perch connection refuses Tor
        # when it is saved rather than carrying a toggle it cannot honour.
        from openmirror.providers import perch as perch_mod

        impls = perch_mod.build(perch_mod.from_connection(conn))
    else:
        from openmirror.providers.openai_compat import OpenAICompatProvider
        from openmirror.providers.sora import SoraProvider

        shared = one(OpenAICompatProvider)
        impls = {
            Modality.CHAT: shared,
            Modality.EMBEDDING: shared,
            Modality.STT: shared,
            Modality.TTS: shared,
            Modality.IMAGE: shared,
            # Only reached when a connection is actually ticked for video,
            # which the catalogue does for OpenAI itself and for nothing else
            # that copied its chat shape.
            Modality.VIDEO: one(SoraProvider, timeout=long_running),
        }

    # What was asked for, narrowed to what the adapter can actually do. A
    # connection ticked for video against an image-only host would otherwise
    # be registered and fail at the first call.
    served = set(impls)
    modalities = (wanted & served) or served
    impls = {m: impl for m, impl in impls.items() if m in modalities}

    info = ProviderInfo(
        id=conn.id,
        label=conn.label or conn.id,
        modalities=set(modalities),
        local=conn.local,
        base_url=conn.base_url,
    )
    return info, impls


async def register(conn: Connection, registry: Any) -> list[str]:
    """Register one stored connection. Returns the provider ids it became.

    One connection is usually one provider, and then this is `build` plus
    `register`. Perch is the exception it exists for: one host, one token, and
    up to five providers — one per live service, each under its own id, so
    chat can be routed at Perch while images go somewhere else. Probing is what
    decides which, which is why this is async where `build` is not.
    """
    if conn.adapter == 'perch':
        from openmirror.providers import perch as perch_mod

        cfg = perch_mod.from_connection(conn)
        available = await perch_mod.probe(cfg)
        return registry.register_perch(cfg, available, connection=conn)

    info, impls = build(conn)
    registry.register(info, impls, connection=conn)
    return [info.id]


def provider_ids(conn: Connection) -> list[str]:
    """Every provider id a connection may have registered, for removing it."""
    if conn.adapter == 'perch':
        from openmirror.providers import perch as perch_mod

        return list(perch_mod.PROVIDER_IDS)
    return [conn.id]


def from_host(
    host_id: str,
    *,
    api_key: str = '',
    base_url: str = '',
    connection_id: str = '',
    modalities: list[str] | None = None,
    local: bool | None = None,
    tor: bool = False,
) -> Connection:
    """A connection from a catalogue preset, with everything overridable."""
    host = HOSTS_BY_ID.get(host_id)
    if host is None:
        raise KeyError(f'no such preset: {host_id}')
    # A key already in the environment is used when none was typed, so
    # connecting a service the install was already configured for does not
    # mean finding the key again.
    key = api_key or (os.getenv(host.env_key, '') if host.env_key else '')
    return Connection(
        id=connection_id or host_id,
        label=host.label,
        adapter=host.adapter,
        base_url=base_url or host.base_url,
        api_key=key,
        modalities=modalities if modalities is not None else sorted(m.value for m in host.modalities),
        local=host.local if local is None else local,
        tor=tor,
        host_id=host_id,
    )
