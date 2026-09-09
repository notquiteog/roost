"""Routing a provider through Tor: precedence, the default, and the diagnosis.

The second half of this file is the guard that keeps adapters from opening
their own connections, and it has been through enough rounds of "that check
was passing without checking anything" to be worth a note on how it is
verified. Two properties are not expressible as tests, because testing them
means damaging the test file itself, so they are checked by hand and recorded
here:

**The floors are structural.** They live in `_outbound_calls` rather than in a
test of their own, so every caller inherits them. The verification is a
*two-fault* plant, because any single fault trips either arrangement: delete
both `test_the_guard_notices…` and `test_a_dead_match_set…`, and blind the
walk in the same run. All four remaining checks then fail, and the message
names the cause — "the scan found only 0 outbound calls… every check that
filters this list is about to pass on an empty one" — rather than asserting
that an empty list was not empty. Under the earlier arrangement, where a
sibling test held the floors, that same run was clean.

**A plant proves nothing unless it applied and still parses.** A substitution
that silently matched nothing reports a pass; one that breaks the file reports
a failure. Both are uninformative, and they are the same mistake in opposite
directions — reading "it passed" as "it was checked", and "it failed" as "it
was caught". So each plant asserts its substitution applied, that the result
parses, and that the failure *names the planted line*.


No socket and no database. The rules are the ones cryptostore's proxy config
settled, and they are pinned here for the same reason they are pinned there:
the failure they prevent is silent. An operator who moved their proxy once and
finds half their traffic still dialling the old port has no error telling them
so — only something that does not work.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

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


# -- every adapter actually uses the transport it was given ------------------


# Every way this codebase has of opening an outbound connection. Wider than
# aiohttp on purpose: a guard you can step around by importing httpx is not a
# guard.
_OPENS_A_CONNECTION = {
    'aiohttp.ClientSession',
    'aiohttp.request',
    'httpx.Client',
    'httpx.AsyncClient',
    'httpx.get',
    'httpx.post',
    'httpx.put',
    'httpx.delete',
    'httpx.request',
    'requests.Session',
    'requests.get',
    'requests.post',
    'requests.put',
    'requests.delete',
    'requests.request',
    'urlopen',
    'urllib.request.urlopen',
}

_EXEMPT = 'transport-exempt:'

# An exemption nobody can review is one the next reader assumes was
# load-bearing. Short enough to write in passing, long enough to be a reason.
_MIN_REASON = 25

# What the scan must find to be believed. See the vacuity test below: the
# failure mode of a check like this is not being wrong, it is looking at
# nothing and reporting success.
_MIN_FILES = 30
_MIN_CALLS = 3


def _package_root() -> Path:
    root = Path(__file__).resolve().parent.parent / 'roost'
    assert root.is_dir(), f'the package is not where this test thinks it is: {root}'
    return root


def _dotted(node: ast.AST) -> str:
    """`aiohttp.ClientSession` from the call's func expression, or ''."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        return ''
    return '.'.join(reversed(parts))


def _exemption_for(lines: list[str], index: int) -> str | None:
    """The exemption covering the call on `lines[index]`, if there is one.

    Looked for on the line itself and then in the contiguous comment block
    directly above it, stopping at the first line that is not a comment. The
    block matters: every reason worth writing here ran to three or four
    sentences, and a rule that forces those onto one line is a rule that
    produces worse reasons. Stopping at the first non-comment line is what
    keeps a comment attached to something else — the function above, say —
    from exempting a call below it.
    """
    candidates = [lines[index]]
    for above in range(index - 1, -1, -1):
        stripped = lines[above].strip()
        if not stripped.startswith('#'):
            break
        candidates.append(stripped)

    for line in candidates:
        if _EXEMPT in line:
            return line.split(_EXEMPT, 1)[1].strip()
    return None


def _scan(root: Path):
    """(file, line, text, exemption) for everything under `root` that opens a socket.

    Parsed rather than grepped. A regex over source text matches the words in
    a comment explaining why something cannot be done, or in a docstring
    describing the bug this guard exists to prevent — and a check that cries
    wolf is one that gets deleted rather than fixed. The syntax tree contains
    only what the file actually *does*.

    Takes a root so the guard can be pointed at a known tree and checked, in
    `test_the_guard_flags_exactly_what_it_should`. A checker nobody has run
    against a planted fault is a checker nobody has tested.
    """
    scanned = 0
    for path in sorted(root.rglob('*.py')):
        source = path.read_text()
        lines = source.splitlines()
        scanned += 1
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:  # pragma: no cover - would fail the suite anyway
            raise AssertionError(f'{path} does not parse: {exc}') from exc

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _dotted(node.func) not in _OPENS_A_CONNECTION:
                continue
            index = node.lineno - 1
            rel = path.relative_to(root.parent)
            yield rel, node.lineno, lines[index].strip(), _exemption_for(lines, index)

    _scan.last_file_count = scanned


def _outbound_calls():
    """Everything in this package that opens a socket, with the scan checked.

    Both floors are asserted *here* rather than in a test of their own, and
    that placement is the point. Every check below filters this list and
    asserts the result is empty — which is exactly what an empty input
    produces. Leaving the vacuity check in a sibling test means those
    assertions are only meaningful because something else happens to be
    watching, and an assertion that depends on another test to mean anything
    is not one you can rely on when the other test is what changes.
    """
    found = list(_scan(_package_root()))
    assert _scan.last_file_count >= _MIN_FILES, (
        f'only {_scan.last_file_count} python files were scanned, which is not this package — '
        'the guard is looking in the wrong place and would pass by having nothing to check'
    )
    assert len(found) >= _MIN_CALLS, (
        f'the scan found only {len(found)} outbound calls in a package that certainly has '
        'more — the match set has probably stopped matching, and every check that filters '
        'this list is about to pass on an empty one'
    )
    return found


def test_the_guard_flags_exactly_what_it_should(tmp_path):
    """Point the guard at a file whose every line has a known verdict.

    The reason this exists rather than a hand-run check: half of a guard's
    behaviour is what it must *not* flag, and "it did not flag the comment" is
    indistinguishable from "the file was never read". A peer hit exactly that
    one level up — a `sed` meant to plant a fault silently matched nothing, so
    a pass meant nothing had been planted rather than nothing had been caught.

    So the decoys and the real calls live in the *same file*. If the scan did
    not read it, the expected offenders are missing and the test fails; if a
    decoy trips, there is an extra one. Neither half can pass by absence.
    """
    (tmp_path / 'subject.py').write_text(
        textwrap.dedent(
            '''
            """Module docstring mentioning aiohttp.ClientSession() in prose.

            And httpx.AsyncClient() again, because this is exactly where the
            words appear in the real package — in comments explaining the bug.
            """
            import aiohttp

            TEMPLATE = """
            import aiohttp
            aiohttp.ClientSession()
            """

            # Never write requests.get('...') in this module.

            def exempted():
                # transport-exempt: reaches a thing that is not a model server,
                # explained at length so the reason test is satisfied too.
                return aiohttp.ClientSession()

            def bare():
                return aiohttp.ClientSession()
            '''
        ).lstrip()
    )

    found = list(_scan(tmp_path))
    flagged = {number: exemption for _, number, _, exemption in found}

    # Exactly two calls are real code. The docstring, the template string and
    # the comment are not, and a scan that read the file at all must see the
    # difference.
    assert len(found) == 2, (
        f'expected the two real calls and nothing else, got {sorted(flagged)} — '
        'a decoy tripped, or the file was never read'
    )

    lines = (tmp_path / 'subject.py').read_text().splitlines()
    exempted_line = next(i for i, ln in enumerate(lines, 1) if 'return aiohttp' in ln)
    bare_line = next(i for i, ln in enumerate(lines, 1) if i > exempted_line and 'return aiohttp' in ln)

    assert flagged[exempted_line], 'the exempted call should carry its reason'
    assert flagged[bare_line] is None, 'the bare call should carry none'


def test_an_exemption_does_not_reach_past_the_code_above_it(tmp_path):
    """A comment attached to one thing must not excuse the next thing down."""
    (tmp_path / 'subject.py').write_text(
        textwrap.dedent(
            '''
            import aiohttp

            # transport-exempt: this belongs to the helper directly below it and
            # must not reach past the def line into the function after it.
            def helper():
                return None

            def sneaky():
                return aiohttp.ClientSession()
            '''
        ).lstrip()
    )

    found = list(_scan(tmp_path))
    assert len(found) == 1
    assert found[0][3] is None, 'the exemption reached past the code it was attached to'


def test_the_guard_notices_when_it_is_looking_at_nothing(tmp_path, monkeypatch):
    """The floors in `_outbound_calls` are the thing under test here.

    A guard's failure mode is not being wrong. It is being vacuous — scanning
    an empty directory, or matching nothing because the pattern broke — and
    reporting success, which is indistinguishable from passing right up until
    the day it was meant to catch something. So both floors are exercised
    against a tree that would trip them, rather than trusted because they are
    written down.
    """
    monkeypatch.setattr('tests.test_tor._package_root', lambda: tmp_path)
    with pytest.raises(AssertionError, match='looking in the wrong place'):
        _outbound_calls()

    # Enough files to clear the first floor, so the second is what answers.
    for n in range(_MIN_FILES + 1):
        (tmp_path / f'm{n}.py').write_text('x = 1\n')
    with pytest.raises(AssertionError, match='stopped matching'):
        _outbound_calls()


def test_a_dead_match_set_does_not_pass_quietly(monkeypatch):
    """The other half: the right files, and a pattern that recognises nothing."""
    monkeypatch.setattr('tests.test_tor._OPENS_A_CONNECTION', {'nothing.at.all'})
    with pytest.raises(AssertionError, match='stopped matching'):
        _outbound_calls()


def test_nothing_opens_its_own_connection_without_saying_why():
    """The bug this exists to prevent was live.

    `OllamaProvider` accepted a transport and then built its own
    `aiohttp.ClientSession` in every one of its methods. A connection with the
    Tor toggle on registered, probed, showed a "tor" tag in the UI — and sent
    every message, every model listing and every embedding straight out. That
    is strictly worse than not having the feature, because the operator
    believes they are on Tor.

    It covers the whole package rather than the adapters, because the other
    shape of this bug is a call that never went near an adapter at all — an
    evaluation helper, a health check, a token counter — reaching a model
    server on a bare client.
    """
    offenders = [
        f'{path}:{number}: {text}'
        for path, number, text, exemption in _outbound_calls()
        if exemption is None
    ]
    assert not offenders, (
        'these open a connection without going through a transport and without '
        'saying why. If it reaches a model server, use self.transport. If it '
        'genuinely must not, mark it `# transport-exempt: <reason>`:\n  '
        + '\n  '.join(offenders)
    )


def test_an_exemption_gives_a_reason_worth_reading():
    """A marker with nothing after it is a silenced check, not a decision."""
    thin = [
        f'{path}:{number}: {exemption!r}'
        for path, number, _, exemption in _outbound_calls()
        if exemption is not None and len(exemption) < _MIN_REASON
    ]
    assert not thin, (
        'these exemptions are too short to review — say what this reaches and '
        f'why a transport would be wrong for it (at least {_MIN_REASON} '
        'characters):\n  ' + '\n  '.join(thin)
    )


async def test_a_tor_connection_refuses_rather_than_going_direct(monkeypatch):
    """Missing SOCKS support is an error at the point of use.

    Falling back to a direct connection would be the same lie in a different
    place: the request succeeds, the operator sees no error, and the traffic
    went the ordinary way.
    """
    import builtins

    real_import = builtins.__import__

    def no_socks(name, *args, **kwargs):
        if name == 'aiohttp_socks':
            raise ImportError('not installed')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_socks)

    with pytest.raises(tor.TorUnavailable, match='roost\\[tor\\]'):
        Transport(tor=True).session()
