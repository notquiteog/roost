"""Which provider answers which modality, for whom.

The brief was that a person can choose local, OpenAI or Anthropic separately
for every kind of generation. That is this file. A route is a
(modality → provider, model) pair, resolved in three layers:

    per-user override  →  workspace default  →  first provider that can do it

so an admin sets a sane default once and a user who wants their dictation to
stay on their own hardware changes one route without touching the rest.

The privacy posture is enforced here rather than in the UI, because a toggle
that only hides an option is not a guarantee. `local_only` on a route set
means a remote provider is refused at resolution time even if something
downstream asks for one by name, and the refusal is an error rather than a
silent fallback — quietly sending audio to a hosted service because the local
one was down is precisely the failure this is meant to prevent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from openmirror.providers.base import Modality, ProviderInfo

log = logging.getLogger(__name__)


class NoProviderError(RuntimeError):
    """No provider can serve this modality under the current constraints."""


class PrivacyRefusal(NoProviderError):
    """A provider exists but sending to it would break the caller's own rule."""


@dataclass(slots=True)
class Route:
    provider: str
    model: str = ''
    # Per-route options: a voice for TTS, a size for images, a sampler.
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RouteSet:
    """One person's, or one workspace's, choice per modality."""

    routes: dict[Modality, Route] = field(default_factory=dict)
    # Off by default. When on, every modality must resolve to a provider
    # marked local, and there is no fallback out of it.
    local_only: bool = False

    def get(self, modality: Modality) -> Route | None:
        return self.routes.get(modality)


@dataclass(slots=True)
class _Entry:
    info: ProviderInfo
    # One object may implement several capabilities — OpenAICompatProvider
    # does chat, embeddings, speech both ways and images — so instances are
    # stored per modality rather than per provider.
    impls: dict[Modality, Any]
    # The stored connection this was built from, when it came from one.
    # Environment-configured providers have none, which is also how the UI
    # knows which entries it may edit and which are pinned by the install.
    connection: Any = None


class ProviderRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._defaults = RouteSet()

    # -- registration -------------------------------------------------------

    def register(self, info: ProviderInfo, impls: dict[Modality, Any], connection: Any = None) -> None:
        # Registering the same id twice replaces it, so reconfiguring a
        # connection at runtime does not need a restart or a deregister call.
        self._entries[info.id] = _Entry(info=info, impls=dict(impls), connection=connection)
        carried = ' over tor' if connection is not None and getattr(connection, 'tor', False) else ''
        log.info(
            'provider %s registered for %s%s', info.id, ','.join(sorted(m.value for m in impls)), carried
        )

    def unregister(self, provider_id: str) -> None:
        self._entries.pop(provider_id, None)

    def register_perch(self, cfg: Any, available: dict[str, bool]) -> list[str]:
        """Register one Perch host as its live services.

        Each service becomes its own provider id rather than one 'perch'
        entry, so someone can route chat at Perch and images at a hosted
        service without the two decisions being tangled.
        """
        from openmirror.providers import perch as perch_mod

        impls = perch_mod.build(cfg, available)
        registered: list[str] = []
        for info in perch_mod.describe(cfg, available):
            service = info.id.split(':', 1)[1]
            mine = {m: impl for m, impl in impls.items() if m in perch_mod.SERVICE_MODALITIES[service]}
            if not mine:
                continue
            self.register(info, mine)
            registered.append(info.id)
        return registered

    # -- inspection ---------------------------------------------------------

    def list(self) -> list[ProviderInfo]:
        return [e.info for e in self._entries.values()]

    def providers_for(self, modality: Modality, *, local_only: bool = False) -> list[ProviderInfo]:
        return [
            e.info
            for e in self._entries.values()
            if modality in e.impls and (not local_only or e.info.local)
        ]

    def connection(self, provider_id: str) -> Any:
        """The stored connection behind a provider, or None if it came from the
        environment. Used to decide whether a UI may offer to edit it."""
        entry = self._entries.get(provider_id)
        return entry.connection if entry else None

    def impl(self, provider_id: str, modality: Modality) -> Any:
        """The object serving one modality for one provider, without routing.

        For asking a specific provider what models it has — a question about
        that connection rather than about which one should answer next.
        """
        entry = self._entries.get(provider_id)
        if entry is None:
            raise NoProviderError(f'no such provider: {provider_id}')
        if modality not in entry.impls:
            raise NoProviderError(f'{provider_id} is not registered for {modality.value}')
        return entry.impls[modality]

    def describe(self, provider_id: str) -> dict[str, Any]:
        """Everything a client should know about one provider, keys excluded."""
        entry = self._entries.get(provider_id)
        if entry is None:
            raise NoProviderError(f'no such provider: {provider_id}')
        conn = entry.connection
        return {
            'id': entry.info.id,
            'label': entry.info.label,
            'local': entry.info.local,
            'base_url': entry.info.base_url,
            'modalities': sorted(m.value for m in entry.impls),
            'editable': conn is not None,
            'tor': bool(getattr(conn, 'tor', False)),
            'host_id': getattr(conn, 'host_id', ''),
            'has_key': bool(getattr(conn, 'api_key', '')),
        }

    def capabilities(self, *, local_only: bool = False) -> dict[str, list[str]]:
        """What this install can do, by modality — the answer a UI needs to
        grey out what is genuinely unavailable rather than offering it and
        failing later."""
        out: dict[str, list[str]] = {}
        for modality in Modality:
            found = self.providers_for(modality, local_only=local_only)
            if found:
                out[modality.value] = [p.id for p in found]
        return out

    # -- defaults -----------------------------------------------------------

    def set_defaults(self, routes: RouteSet) -> None:
        self._defaults = routes

    @property
    def defaults(self) -> RouteSet:
        return self._defaults

    # -- resolution ---------------------------------------------------------

    def resolve(self, modality: Modality, user_routes: RouteSet | None = None) -> tuple[Any, Route, ProviderInfo]:
        """Pick the provider for this modality, honouring the user's rules.

        Raises rather than falling back when a named provider is missing: a
        person who chose a specific model wants to hear that it is gone, not
        to be quietly served by another one.
        """
        local_only = (user_routes.local_only if user_routes else False) or self._defaults.local_only

        route = (user_routes.get(modality) if user_routes else None) or self._defaults.get(modality)

        if route is not None:
            entry = self._entries.get(route.provider)
            if entry is None or modality not in entry.impls:
                raise NoProviderError(
                    f'{modality.value}: provider {route.provider!r} is not registered for it'
                )
            if local_only and not entry.info.local:
                raise PrivacyRefusal(
                    f'{modality.value}: {route.provider!r} is remote and this session is set to local only'
                )
            return entry.impls[modality], route, entry.info

        # Nothing chosen: take the first that can do it. Locals are preferred
        # even when local_only is off, so the default behaviour of an install
        # with a GPU attached is to use it.
        candidates = sorted(
            (e for e in self._entries.values() if modality in e.impls and (not local_only or e.info.local)),
            key=lambda e: (not e.info.local, e.info.id),
        )
        if not candidates:
            hint = ' (no local provider is configured for it)' if local_only else ''
            raise NoProviderError(f'{modality.value}: nothing is configured to handle it{hint}')

        chosen = candidates[0]
        return chosen.impls[modality], Route(provider=chosen.info.id), chosen.info


# What a model has to be able to do to serve each modality. Ollama is the only
# backend that reports this today; everything else returns bare ids, which is
# why an empty capability list means "no idea, assume it can" rather than "no".
_NEEDED = {
    Modality.CHAT: ('completion', 'tools'),
    Modality.EMBEDDING: ('embedding',),
}


def can_serve(model: dict[str, Any], modality: Modality) -> bool:
    """Whether a listed model can serve this modality, as far as anyone knows."""
    declared = model.get('capabilities') or []
    if not declared:
        return True
    wanted = _NEEDED.get(modality)
    if not wanted:
        return True
    return any(c in declared for c in wanted)


def pick_model(models: list[dict[str, Any]], modality: Modality, *, need_tools: bool = False) -> str:
    """The best of a provider's models for a job, when nobody named one.

    Two failures this exists to stop, both seen on a real install:

    An embedding model chosen for chat. `/api/tags` lists everything that is
    pulled and the first entry is alphabetical, so an install with
    `qwen3-embedding:4b` on it answered every conversation with an HTTP 400
    saying that model does not support chat — which reads as a broken daemon
    rather than as a bad default.

    A chat model that cannot call tools chosen for an agent session. It
    connects, it talks, and it can do nothing at all, which is a much more
    confusing failure than not starting.
    """
    usable = [m for m in models if m.get('id') and can_serve(m, modality)]
    if not usable:
        return ''
    if need_tools:
        with_tools = [m for m in usable if 'tools' in (m.get('capabilities') or [])]
        # Only when something declared it. A provider that reports no
        # capabilities at all must not be narrowed to nothing.
        if with_tools:
            usable = with_tools
    return str(usable[0]['id'])


# Process-wide registry. Deliberately a module global: providers are
# configuration, every request reads the same set, and threading one through
# every call site buys nothing.
registry = ProviderRegistry()
