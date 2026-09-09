"""Routing a provider through Tor: precedence, the default, and the diagnosis.

No socket and no database. The rules are the ones cryptostore's proxy config
settled, and they are pinned here for the same reason they are pinned there:
the failure they prevent is silent. An operator who moved their proxy once and
finds half their traffic still dialling the old port has no error telling them
so — only something that does not work.
"""

from __future__ import annotations

import pytest

from roost.net import tor
from roost.net.transport import Transport


def test_the_default_is_arti_not_the_c_daemon(monkeypatch):
    """9150, not 9050. Do not "fix" this back — see roost/net/tor.py."""
    for name in ('TOR_SOCKS_HOST', 'TOR_HOST', 'TOR_SOCKS_PORT', 'TOR_PORT'):
        monkeypatch.delenv(name, raising=False)
    proxy = tor.resolve()
    assert (proxy.host, proxy.port) == ('127.0.0.1', 9150)


@pytest.mark.parametrize(
    ('env', 'expected'),
    [
        ({'TOR_SOCKS_HOST': '10.0.0.5', 'TOR_SOCKS_PORT': '9051'}, ('10.0.0.5', 9051)),
        # The older names still work: an install configured before the
        # rename should not quietly start dialling loopback.
        ({'TOR_HOST': '10.0.0.6', 'TOR_PORT': '9052'}, ('10.0.0.6', 9052)),
        # The newer names win where both are set.
        (
            {'TOR_SOCKS_HOST': 'a', 'TOR_HOST': 'b', 'TOR_SOCKS_PORT': '1', 'TOR_PORT': '2'},
            ('a', 1),
        ),
    ],
)
def test_where_the_proxy_comes_from(monkeypatch, env, expected):
    for name in ('TOR_SOCKS_HOST', 'TOR_HOST', 'TOR_SOCKS_PORT', 'TOR_PORT'):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    proxy = tor.resolve()
    assert (proxy.host, proxy.port) == expected


def test_an_explicit_address_beats_the_environment(monkeypatch):
    """So two connections can use two circuits on one machine."""
    monkeypatch.setenv('TOR_SOCKS_HOST', '10.0.0.9')
    monkeypatch.setenv('TOR_SOCKS_PORT', '9999')
    proxy = tor.resolve('127.0.0.1', 9150)
    assert (proxy.host, proxy.port) == ('127.0.0.1', 9150)


def test_an_unparseable_port_falls_back_rather_than_raising(monkeypatch):
    monkeypatch.setenv('TOR_SOCKS_PORT', 'not-a-port')
    assert tor.resolve().port == tor.DEFAULT_PORT


def test_names_are_resolved_by_the_proxy():
    """socks5h, always.

    Resolving a hostname here and dialling the result through Tor leaks the
    lookup to whatever DNS this machine uses — which is the thing the toggle
    was switched on to prevent.
    """
    assert tor.resolve().url.startswith('socks5h://')


def test_two_minutes_is_a_floor_not_a_default():
    assert tor.timeout_for(None) == tor.MIN_TIMEOUT
    assert tor.timeout_for(30) == tor.MIN_TIMEOUT
    assert tor.timeout_for(600) == 600
    assert tor.timeout_for('nonsense') == tor.MIN_TIMEOUT


async def test_the_probe_names_the_other_implementation(monkeypatch):
    """"Nothing is listening" is useless when the answer is "you run the other one".

    The port that is refused names loopback, not the model server, so without
    this the symptom reads as "the model is down" and sends someone to debug a
    machine that was never dialled.
    """
    listening = {tor.LEGACY_PORT}

    async def fake_open(host, port):
        if port not in listening:
            raise ConnectionRefusedError
        raise ConnectionRefusedError   # never reached for the default port

    monkeypatch.setattr('asyncio.open_connection', fake_open)
    ok, why = await tor.probe(tor.TorProxy('127.0.0.1', tor.DEFAULT_PORT), timeout=0.1)
    assert ok is False
    assert '9150' in why


def test_a_direct_transport_has_no_proxy():
    assert Transport().proxy is None
    assert Transport().describe()['tor'] is False


def test_a_tor_transport_says_where_it_goes():
    described = Transport(tor=True, tor_host='127.0.0.1', tor_port=9150).describe()
    assert described == {'tor': True, 'proxy': '127.0.0.1:9150', 'timeout': 600}
