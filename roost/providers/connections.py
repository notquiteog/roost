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
`roost.net.tor`.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from roost.net.transport import Transport
from roost.providers.base import Modality, ProviderInfo
from roost.providers.catalog import HOSTS_BY_ID

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

    One object usually serves several modalities — the OpenAI adapter does
    chat, embeddings, speech both ways and images — so it is built once and
    referenced from each, which is also what makes a single Tor toggle cover
    all of them.
    """
    transport = conn.transport()
    wanted = {Modality(m) for m in conn.modalities if m in set(Modality)}

    if conn.adapter == 'anthropic':
        from roost.providers.anthropic import AnthropicProvider

        impl: Any = AnthropicProvider(conn.base_url, conn.api_key, provider_id=conn.id, transport=transport)
        served = {Modality.CHAT}
    elif conn.adapter == 'ollama':
        from roost.providers.ollama import OllamaProvider

        impl = OllamaProvider(conn.base_url, conn.api_key, provider_id=conn.id, transport=transport)
        served = {Modality.CHAT, Modality.EMBEDDING}
    elif conn.adapter == 'voyage':
        from roost.providers.voyage import VoyageProvider

        impl = VoyageProvider(conn.base_url, conn.api_key, provider_id=conn.id, transport=transport)
        served = {Modality.EMBEDDING}
    elif conn.adapter == 'google':
        from roost.providers.google import GoogleProvider

        impl = GoogleProvider(conn.base_url, conn.api_key, provider_id=conn.id, transport=transport)
        served = {Modality.EMBEDDING}
    elif conn.adapter == 'a1111':
        from roost.providers.a1111 import A1111Provider

        impl = A1111Provider(conn.base_url, conn.api_key, provider_id=conn.id, transport=transport)
        served = {Modality.IMAGE}
    elif conn.adapter == 'comfyui':
        from roost.providers.comfyui import ComfyUIProvider

        impl = ComfyUIProvider(conn.base_url, conn.api_key, provider_id=conn.id, transport=transport)
        served = {Modality.IMAGE, Modality.VIDEO}
    else:
        from roost.providers.openai_compat import OpenAICompatProvider

        impl = OpenAICompatProvider(conn.base_url, conn.api_key, provider_id=conn.id, transport=transport)
        served = {Modality.CHAT, Modality.EMBEDDING, Modality.STT, Modality.TTS, Modality.IMAGE}

    # What was asked for, narrowed to what the adapter can actually do. A
    # connection ticked for video against an OpenAI-shaped host would
    # otherwise be registered and fail at the first call.
    modalities = (wanted & served) or served
    impls = {m: impl for m in modalities}

    info = ProviderInfo(
        id=conn.id,
        label=conn.label or conn.id,
        modalities=set(modalities),
        local=conn.local,
        base_url=conn.base_url,
    )
    return info, impls


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
