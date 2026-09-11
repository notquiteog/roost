"""The headless search backend, against a results page this test controls.

Pointing these at a live engine would test the engine. So a small HTTP server
here serves pages shaped like the real ones — including the two things that
were actually wrong in production and would have been invisible to a mock:

  * **The redirect wrapper.** Bing links every result through
    `bing.com/ck/a?...&u=a1<base64>`. The blocklist that exists to drop the
    engine's own navigation was therefore dropping every result on the page,
    and the tool reported that the web had no answer.
  * **The title that is a stylesheet.** Startpage's result anchors contain an
    injected `<style>`, and `textContent` swept it up — so a result's title
    came back as `.css-rwvbo4{display:-webkit-box...}`.

Both are asserted below. The browser is real; only the internet is not.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from openmirror.agent.headless_search import (
    ENGINE_ORDER,
    ENGINES,
    Engine,
    HeadlessSearcher,
    SearchUnavailable,
)

playwright = pytest.importorskip('playwright', reason='the headless backend needs the browser extra')


# -- a results page, shaped like the real ones -------------------------------


def _bing_link(target: str) -> str:
    """Bing's click wrapper, built the way Bing builds it."""
    encoded = base64.urlsafe_b64encode(target.encode()).decode().rstrip('=')
    return f'/ck/a?!&&p=deadbeef&u=a1{encoded}'


PAGES: dict[str, str] = {
    # Plain markup, rendered server-side.
    '/plain': """
      <html><body><div id="results">
        <div class="result"><a href="https://example.com/one"><h3>The first result</h3></a>
          <p class="description">What the first one says.</p></div>
        <div class="result"><a href="https://example.org/two"><h3>The second result</h3></a>
          <p class="description">What the second one says.</p></div>
      </div></body></html>
    """,
    # Every link wrapped in the engine's own redirect, as Bing does.
    '/wrapped': f"""
      <html><body><div id="results">
        <div class="result"><a href="{_bing_link('https://example.com/real-one')}">
          <h3>A wrapped result</h3></a><p class="description">Behind a redirect.</p></div>
        <div class="result"><a href="{_bing_link('https://example.org/real-two')}">
          <h3>Another wrapped result</h3></a><p class="description">Also behind one.</p></div>
        <div class="result"><a href="/settings"><h3>Not a result at all</h3></a></div>
      </div></body></html>
    """,
    # An anchor carrying an injected stylesheet, as Startpage does.
    '/styled': """
      <html><body><div id="results">
        <div class="result"><a href="https://example.com/styled">
          <style>.css-rwvbo4{display:-webkit-box;-webkit-line-clamp:2}</style>
          <h3>The real title</h3></a><p class="description">A normal snippet.</p></div>
      </div></body></html>
    """,
    # Results that arrive after the document does, as every JS engine does.
    '/late': """
      <html><body><div id="results"></div><script>
        setTimeout(() => {
          document.getElementById('results').innerHTML =
            '<div class="result"><a href="https://example.com/late"><h3>Arrived late</h3></a>' +
            '<p class="description">After a delay.</p></div>';
        }, 1200);
      </script></body></html>
    """,
    # A challenge page: renders, has no results on it.
    '/challenge': '<html><head><title>Are you a robot?</title></head><body><p>Prove it.</p></body></html>',
    # Markup that matches none of the selectors, to exercise the fallback.
    '/unknown-markup': """
      <html><body><ol>
        <li class="mystery-2026"><a href="https://example.com/fallback"><h2>Found anyway</h2></a>
          <p>The selectors did not match this.</p></li>
      </ol></body></html>
    """,
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - the stdlib's name
        path = self.path.split('?')[0]
        body = PAGES.get(path, '<html><body>nothing here</body></html>')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        return  # the test output is not a web server log


@pytest.fixture(scope='module')
def site():
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()


def engine_for(base: str, path: str, **kw) -> Engine:
    return Engine(
        name='test-engine',
        url=f'{base}{path}?q={{q}}',
        ready=kw.get('ready', '.result'),
        groups=kw.get('groups', (
            {'result': '.result', 'link': 'a[href]', 'snippet': '.description'},
        )),
        # The page's own host, so a link back to it is navigation rather than a
        # result — which is exactly the rule that broke Bing.
        blocked=kw.get('blocked', ('127.0.0.1',)),
    )


async def search(base: str, path: str, count: int = 5, patience: float = 8.0, **kw):
    live = HeadlessSearcher(engine_for(base, path, **kw), patience=patience)
    try:
        return await live.search('anything', count)
    finally:
        await live.close()


# -- the catalog -------------------------------------------------------------


def test_every_engine_in_the_chain_exists():
    """`auto` walks ENGINE_ORDER by name. A typo there is a KeyError at the
    first search rather than at import, which is the worst place for it."""
    assert set(ENGINE_ORDER) <= set(ENGINES)
    for name in ENGINE_ORDER:
        engine = ENGINES[name]
        assert engine.ready and engine.url.count('{q}') == 1
        # `ready` may only name result rows. A container exists in the page
        # shell before any result does, so waiting on one returns immediately
        # and the extraction runs against an empty list — which is how a live
        # engine came back as "no results" while showing six.
        for selector in engine.ready.split(','):
            assert selector.strip() not in ('#b_results', '#main', '#search', '#rso'), (
                f'{name}: `ready` names a container, not a row'
            )


def test_naming_an_unknown_engine_is_refused():
    with pytest.raises(SearchUnavailable):
        HeadlessSearcher('askjeeves')


def test_auto_walks_the_chain_and_a_name_does_not():
    """An explicit engine never falls through. Somebody who chose Startpage for
    what it does not log has not agreed to be sent to Google instead."""
    assert [e.name for e in HeadlessSearcher('auto').engines] == list(ENGINE_ORDER)
    assert [e.name for e in HeadlessSearcher('bing').engines] == ['bing']


# -- extraction, in a real browser -------------------------------------------


@pytest.mark.asyncio
async def test_plain_results_are_read(site):
    rows = await search(site, '/plain')
    assert [r['url'] for r in rows] == ['https://example.com/one', 'https://example.org/two']
    assert rows[0]['title'] == 'The first result'
    assert rows[0]['snippet'] == 'What the first one says.'


@pytest.mark.asyncio
async def test_a_redirect_wrapper_is_unwrapped(site):
    """The bug this file exists for. Without unwrapping, every result on the
    page is a link to the engine's own host — and the blocklist drops all of
    them, leaving a search that reports the web is empty."""
    rows = await search(site, '/wrapped')
    assert [r['url'] for r in rows] == [
        'https://example.com/real-one', 'https://example.org/real-two',
    ]
    # The engine's own navigation is still dropped.
    assert not any('/settings' in r['url'] for r in rows)


@pytest.mark.asyncio
async def test_a_stylesheet_is_never_a_title(site):
    rows = await search(site, '/styled')
    assert len(rows) == 1
    assert rows[0]['title'] == 'The real title'
    assert 'css-' not in rows[0]['title']


@pytest.mark.asyncio
async def test_results_that_arrive_late_are_waited_for(site):
    """A slow render must not be reported as an empty web. The extraction keeps
    re-reading for a few seconds rather than taking the first answer."""
    rows = await search(site, '/late')
    assert [r['title'] for r in rows] == ['Arrived late']


@pytest.mark.asyncio
async def test_unknown_markup_falls_back_to_headings(site):
    """Result markup changes without notice. A scraper pinned to one class name
    fails silently, which a model reads as "there is no answer"."""
    rows = await search(site, '/unknown-markup')
    assert [r['url'] for r in rows] == ['https://example.com/fallback']
    assert rows[0]['title'] == 'Found anyway'


@pytest.mark.asyncio
async def test_a_page_with_no_results_says_so(site):
    with pytest.raises(SearchUnavailable) as caught:
        await search(site, '/challenge', patience=0.5)
    # The title is in the message: "no results" and "it served a challenge" are
    # different problems and the person reading the log has to tell them apart.
    assert 'robot' in str(caught.value).lower()


@pytest.mark.asyncio
async def test_the_count_is_a_cap(site):
    rows = await search(site, '/plain', count=1)
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_the_chain_moves_on_when_an_engine_gives_nothing(site):
    """One engine refusing is not an empty web."""
    live = HeadlessSearcher('auto', patience=0.5)
    live.engines = (engine_for(site, '/challenge'), engine_for(site, '/plain'))
    try:
        rows = await live.search('anything', 5)
        assert [r['url'] for r in rows] == ['https://example.com/one', 'https://example.org/two']
    finally:
        await live.close()


@pytest.mark.asyncio
async def test_when_every_engine_refuses_the_error_names_each_one(site):
    live = HeadlessSearcher('auto', patience=0.5)
    live.engines = (engine_for(site, '/challenge'), engine_for(site, '/challenge'))
    try:
        with pytest.raises(SearchUnavailable) as caught:
            await live.search('anything', 5)
        message = str(caught.value)
        assert message.count('test-engine') == 2
        assert 'OPENMIRROR_SEARCH_ENGINE' in message
    finally:
        await live.close()


@pytest.mark.asyncio
async def test_searches_share_one_browser(site):
    """`research` issues five searches at once. Five browsers rather than five
    pages is about a gigabyte."""
    live = HeadlessSearcher(engine_for(site, '/plain'))
    try:
        results = await asyncio.gather(*(live.search(f'q{i}', 2) for i in range(4)))
        assert all(len(rows) == 2 for rows in results)
        assert live._browser is not None and live._browser.is_connected()
        # One browser, four contexts, all closed again.
        assert live._browser.contexts == []
    finally:
        await live.close()


@pytest.mark.asyncio
async def test_closing_is_idempotent(site):
    live = HeadlessSearcher(engine_for(site, '/plain'))
    await live.search('anything', 1)
    await live.close()
    await live.close()
    assert live._browser is None
