"""Routing a provider's traffic through Tor.

The rules here are lifted deliberately from cryptostore's `server/config/tor.js`,
because the two projects run on the same machines and an operator who moved
their proxy should not have to move it twice. Same precedence, same default,
same diagnosis when nothing answers:

    TOR_SOCKS_HOST / TOR_HOST, else 127.0.0.1
    TOR_SOCKS_PORT / TOR_PORT, else 9150

**9150, not 9050 — do not "fix" this back.** The C tor daemon's SocksPort
defaults to 9050 and Arti's to 9150, and the projects this shares a machine
with moved to Arti. An operator who never pinned a port and gets 9050 here
sees `ECONNREFUSED 127.0.0.1` — an error naming loopback rather than the model
server, which reads as "the model is down" and sends them to debug a machine
that was never dialled. `probe` therefore checks the *other* implementation's
port before concluding nothing is listening, so the message can say which one
it found. It is never dialled as a fallback: silently retrying on a port any
unprivileged process can bind would route traffic to whatever won it, and do it
invisibly, which is the worst possible failure for a feature whose whole
purpose is controlling where traffic goes.

**Names are resolved by the proxy, never here.** `socks5h`, always. Resolving
`some-host.example` locally and then dialling the result through Tor leaks the
lookup to whatever DNS this machine uses, which is the thing the toggle was
switched on to prevent.

**Two minutes is a floor, not a default.** A request over Tor runs on two
clocks — building the circuit, then waiting for bytes — and a hosted model can
take a while on either. A caller asking for less gets this; a caller asking
for more keeps what it asked for.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)

DEFAULT_HOST = '127.0.0.1'

# Arti. See the module docstring before changing it.
DEFAULT_PORT = 9150

# The C tor daemon's default. Never dialled — it exists only so a failed probe
# can say "the thing answering on 9050 is C tor, and this defaults to Arti on
# 9150" instead of the useless truth that nothing answered.
LEGACY_PORT = 9050

# The shortest timeout anything reached through Tor is allowed to have.
MIN_TIMEOUT = 120


class TorUnavailable(RuntimeError):
    """Tor was asked for and cannot be provided.

    Always raised rather than falling back to a direct connection. A toggle
    that quietly sends traffic the ordinary way when the proxy is missing is
    worse than one that fails: the operator believes they are on Tor.
    """


@dataclass(frozen=True, slots=True)
class TorProxy:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    @property
    def url(self) -> str:
        # socks5h: the proxy resolves. See the module docstring.
        return f'socks5h://{self.host}:{self.port}'

    def __str__(self) -> str:
        return f'{self.host}:{self.port}'


def resolve(host: str = '', port: int | str = 0) -> TorProxy:
    """Where the SOCKS proxy is: what was passed, else the environment, else Arti.

    Explicit arguments win so that one connection can be pointed at a
    different proxy from the rest, which is the only way to run two circuits
    on one machine.
    """
    chosen_host = (
        str(host).strip()
        or os.getenv('TOR_SOCKS_HOST', '').strip()
        or os.getenv('TOR_HOST', '').strip()
        or DEFAULT_HOST
    )
    raw_port = str(port or '').strip() or os.getenv('TOR_SOCKS_PORT', '').strip() or os.getenv('TOR_PORT', '').strip()
    try:
        chosen_port = int(raw_port) if raw_port else DEFAULT_PORT
    except ValueError:
        log.warning('tor: %r is not a port number, using %d', raw_port, DEFAULT_PORT)
        chosen_port = DEFAULT_PORT
    return TorProxy(host=chosen_host, port=chosen_port)


def timeout_for(requested: float | None) -> float:
    """A timeout for something reached through Tor: the caller's, or the floor."""
    try:
        asked = float(requested or 0)
    except (TypeError, ValueError):
        asked = 0.0
    return asked if asked > MIN_TIMEOUT else float(MIN_TIMEOUT)


def connector(proxy: TorProxy, *, timeout: float | None = None):
    """An aiohttp connector that dials through the proxy.

    `aiohttp_socks` is an optional dependency: an install that never turns Tor
    on should not have to carry it. The error names the extra to install
    rather than surfacing an ImportError, because the person seeing this is an
    operator who ticked a box, not someone reading a stack trace.
    """
    try:
        from aiohttp_socks import ProxyConnector
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise TorUnavailable(
            'Tor routing needs the socks extra: pip install "roost[tor]" '
            '(or pip install aiohttp-socks). Nothing was sent.'
        ) from exc

    return ProxyConnector.from_url(proxy.url, rdns=True, connect_timeout=timeout_for(timeout))


async def probe(proxy: TorProxy | None = None, *, timeout: float = 5.0) -> tuple[bool, str]:  # noqa: ASYNC109 - a socket connect deadline, not a request budget
    """Whether something is listening, and what to say when it is not.

    Only that a TCP connection can be made — it does not build a circuit and
    does not prove the proxy is Tor rather than any other SOCKS server. That
    is the honest limit of a cheap check, and enough to tell an operator
    whether the box they ticked has anything behind it.
    """
    proxy = proxy or resolve()

    async def listening(port: int) -> bool:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(proxy.host, port), timeout=timeout
            )
        except (OSError, TimeoutError):
            return False
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
        return True

    if await listening(proxy.port):
        return True, f'a SOCKS proxy is listening on {proxy}'

    # Before concluding nothing is there, look at the other implementation's
    # port — the answer is almost always "you are running the other one".
    if proxy.port == DEFAULT_PORT and await listening(LEGACY_PORT):
        return False, (
            f'nothing is listening on {proxy}, but something is on port {LEGACY_PORT} — '
            'that is the C tor daemon\'s default, and this defaults to Arti\'s 9150. '
            f'Set TOR_SOCKS_PORT={LEGACY_PORT} if that is the proxy you meant.'
        )
    if proxy.port == LEGACY_PORT and await listening(DEFAULT_PORT):
        return False, (
            f'nothing is listening on {proxy}, but something is on port {DEFAULT_PORT} — '
            'that is Arti\'s default. Set TOR_SOCKS_PORT=9150, or unset it.'
        )

    return False, (
        f'nothing is listening on {proxy}. Start Arti (or the tor daemon), or point '
        'TOR_SOCKS_HOST and TOR_SOCKS_PORT at wherever the proxy actually is.'
    )
