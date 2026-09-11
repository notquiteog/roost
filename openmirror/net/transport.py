"""How every provider opens a socket.

Providers used to build their own `aiohttp.ClientSession` each, which was fine
until two things had to be true of all of them at once: that a connection can
be routed through Tor, and that the routing is per *connection* rather than
per install. Six adapters each growing their own copy of that logic is six
places for it to be subtly different — and the one that gets it wrong is the
one that quietly connects directly.

So a provider is handed a transport and asks it for a session. The transport
owns the timeout, the proxy and nothing else; it deliberately does not know
what is being sent, which is what lets the same object serve chat, embeddings,
speech and image generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from openmirror.net import tor as tor_mod

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Transport:
    """The socket half of a provider connection.

    `tor` is opt-in per connection and never inferred. An install may have the
    proxy running for something else entirely, and turning it on for a model
    server because it happens to be there would be a decision made on the
    operator's behalf about where their prompts go.
    """

    timeout: int = 600
    tor: bool = False
    # Where the proxy is, when it is not where the environment says. Empty
    # means "ask openmirror.net.tor", which is what almost every connection does.
    tor_host: str = ''
    tor_port: int = 0
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def proxy(self) -> tor_mod.TorProxy | None:
        return tor_mod.resolve(self.tor_host, self.tor_port) if self.tor else None

    def session(self, timeout: float | None = None) -> aiohttp.ClientSession:
        """A session for one call. Closed by the caller, as `async with`."""
        total = float(timeout or self.timeout)
        if not self.tor:
            # transport-exempt: this module *is* the transport — the one place
            # a session is meant to be constructed, and what every adapter is
            # required to come through instead of doing this itself.
            return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=total))

        proxy = self.proxy
        assert proxy is not None
        total = tor_mod.timeout_for(total)
        # transport-exempt: the same, for the proxied case. If a call ever
        # needs a session that is not built here, that is the bug the guard in
        # tests/test_tor.py exists to catch.
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=total),
            connector=tor_mod.connector(proxy, timeout=total),
        )

    def describe(self) -> dict[str, Any]:
        """What a UI shows about how this connection is carried."""
        return {'tor': self.tor, 'proxy': str(self.proxy) if self.tor else None, 'timeout': self.timeout}


# Used by anything constructed without one — tests, and the providers built
# before this existed. Shared because it holds no per-call state.
DIRECT = Transport()


def transport_for(*, tor: bool = False, timeout: int = 600, tor_host: str = '', tor_port: int = 0) -> Transport:
    if not tor and timeout == 600 and not tor_host and not tor_port:
        return DIRECT
    return Transport(timeout=timeout, tor=tor, tor_host=tor_host, tor_port=tor_port)
