"""Searching the web with a real browser, because the keyless paths are shut.

Every search engine worth querying now answers a plain HTTP request with an
anomaly page. `web.py` says so in the comment above its DuckDuckGo fallback,
and that comment is the whole argument for this file: the options had narrowed
to "hold an API key" or "do not search", and an install that wanted neither a
third party nor a self-hosted SearxNG had nothing.

A headless browser is the third option. It runs the page's JavaScript, carries
a plausible fingerprint, and gets the results a person would get, with no key
and no account. It costs a Chromium — about a second to start and a few hundred
megabytes while it runs — which is why the browser is shared between searches
and closed when nothing has used it for a while.

Three things here are deliberate and easy to get wrong.

**A fresh context every time, never the agent's profile.** `agent/browser.py`
keeps a persistent profile you have logged into by hand, and pointing that at a
search engine would attach every query to those accounts — and put the agent's
search traffic in the same cookie jar as its signed-in shopping. Each search
gets its own throwaway context instead: no cookies in, none kept.

**The extraction has a fallback that does not name a single CSS class.** Result
markup changes without notice and a scraper pinned to `.result__a` breaks
silently — it returns nothing, which a model reads as "the web has no answer".
So each engine gets its selectors, and behind them is a generic pass over
headings that link somewhere else, which is what a results page has been for
twenty years.

**More than one engine, tried in order.** Measured, not assumed: as of writing,
DuckDuckGo answers a headless Chromium with an error page on all three of its
endpoints, Google serves a challenge, and Startpage and Bing both answer
normally. An engine being blocked is the ordinary case rather than the
exceptional one, so the default `auto` walks a list — least-tracking first —
and only fails when every one of them has refused. Naming an engine explicitly
turns that off: somebody who chose Startpage for what it does not log has not
consented to quietly falling through to Google.

**What `ready` may name.** Only the result rows, never the container they land
in. The container exists in the page shell before any result does, so waiting
on `#b_results` returns immediately and extraction runs against an empty list —
which is precisely the silent "no results" this file exists to avoid. It cost a
live run to find, and the extraction retries for a few seconds regardless, so a
slow render is not the same as an empty page.

**Consent dialogs are declined, not accepted.** Where a banner is in the way
the only buttons this clicks are the ones that refuse: "Reject all", "Only
necessary". Accepting on someone's behalf is not this code's decision to make,
and a search does not need the cookies anyway.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus

log = logging.getLogger(__name__)

# Long enough that a `research` call's five searches share one browser, short
# enough that a machine is not holding a Chromium open all afternoon because
# somebody asked one question at lunchtime.
IDLE_TIMEOUT = 180.0
PAGE_TIMEOUT = 30_000
# How long to keep re-reading the page before calling it empty. Results arrive
# after `domcontentloaded` on every engine here, and a fixed sleep would be
# either too short for a slow network or wasted on a fast one.
EXTRACT_PATIENCE = 8.0
EXTRACT_INTERVAL = 0.4

# A current desktop Chrome. Not a disguise — Playwright's own default advertises
# HeadlessChrome, which several engines serve a blank page to, and being served
# a blank page is not a security property worth keeping.
USER_AGENT = (
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/127.0.0.0 Safari/537.36'
)

# Clicked when a consent wall is in the way. Refusals only, in the languages a
# default install is likely to be served.
DECLINE = (
    'reject all', 'reject non-essential', 'only necessary', 'necessary only',
    'essential only', 'only essential', 'decline', 'refuse all', 'disagree',
    'alles ablehnen', 'tout refuser', 'rechazar todo',
)


@dataclass(frozen=True)
class Engine:
    name: str
    # `{q}` is replaced with the URL-encoded query.
    url: str
    # What says the results have arrived. A comma-joined list, so a layout
    # change that renames one container does not cost the whole wait.
    ready: str
    groups: tuple[dict[str, str], ...] = ()
    # Links to the engine itself are navigation, not results.
    blocked: tuple[str, ...] = ()
    extra: dict[str, str] = field(default_factory=dict)


ENGINES: dict[str, Engine] = {
    'duckduckgo': Engine(
        name='duckduckgo',
        url='https://duckduckgo.com/?q={q}&ia=web',
        ready='article[data-testid="result"], .result, a.result__a',
        groups=(
            {
                'result': 'article[data-testid="result"]',
                'link': 'a[data-testid="result-title-a"], h2 a',
                'snippet': '[data-result="snippet"], [data-testid="result-snippet"]',
            },
            {'result': '.result', 'link': 'a.result__a', 'snippet': '.result__snippet'},
        ),
        blocked=('duckduckgo.com', 'duck.com'),
    ),
    'bing': Engine(
        name='bing',
        url='https://www.bing.com/search?q={q}',
        ready='li.b_algo',
        groups=({'result': 'li.b_algo', 'link': 'h2 a', 'snippet': '.b_caption p, p'},),
        blocked=('bing.com', 'microsoft.com', 'msn.com'),
    ),
    'google': Engine(
        name='google',
        url='https://www.google.com/search?q={q}&num=20',
        ready='div.g, div#rso div[data-ved] h3',
        groups=(
            {'result': 'div.g, div[data-hveid] div[data-ved]', 'link': 'a:has(h3)',
             'snippet': 'div[data-sncf], div[style*="-webkit-line-clamp"]'},
        ),
        blocked=('google.com', 'gstatic.com', 'googleusercontent.com', 'googleadservices.com'),
    ),
    # Startpage serves Google's results without Google's consent wall, which
    # makes it the useful middle option rather than a fourth copy of the same.
    'startpage': Engine(
        name='startpage',
        url='https://www.startpage.com/sp/search?query={q}',
        # `.result` rather than the `.w-gl__result` this used to say: Startpage
        # now builds with hashed class names (`result css-o7i03b`) and the
        # stable half is the word. The title is an `a` wrapping an `h3`, not an
        # `h3` containing an `a`, which is why the link selector is the anchor.
        ready='.result',
        groups=(
            {'result': '.result',
             'link': 'a[href]',
             'snippet': '[class*="description"], [data-testid="result-snippet"], p'},
        ),
        blocked=('startpage.com',),
    ),
}

# What `auto` walks, least-tracking first. Startpage leads because it answers
# and does not log; DuckDuckGo is second on the same principle even though it is
# currently refusing, since that can change back and costs one attempt to find
# out; Bing and Google are last because they are the two that know who asked.
ENGINE_ORDER = ('startpage', 'duckduckgo', 'bing', 'google')

# Pulls results out of whatever the page turned out to be. Runs in the page, so
# it is plain JS with no build step; `cfg` is the engine's selectors plus a
# limit.
EXTRACT_JS = r"""
(cfg) => {
  const out = [];
  const seen = new Set();
  const clean = (s, n) => (s || '').replace(/\s+/g, ' ').trim().slice(0, n);

  const blocked = (href) => {
    try {
      const host = new URL(href, location.href).hostname.toLowerCase();
      return cfg.blocked.some((b) => host === b || host.endsWith('.' + b));
    } catch { return true; }
  };

  // Engines wrap their own results in a click-tracking redirect, so every
  // result on a Bing page is a link to bing.com. Without this the blocklist
  // above — which exists to drop the engine's own navigation — silently
  // discards every result on the page, and the tool reports that the web has
  // no answer. Unwrapping also means the model gets a URL it can fetch rather
  // than one that expires.
  const unwrap = (href) => {
    try {
      const u = new URL(href, location.href);
      if (u.pathname.toLowerCase().includes('/ck/a')) {   // Bing
        const raw = (u.searchParams.get('u') || '').replace(/^a1/, '');
        const b64 = raw.replace(/-/g, '+').replace(/_/g, '/');
        if (b64) {
          const decoded = atob(b64 + '='.repeat((4 - (b64.length % 4)) % 4));
          if (/^https?:/i.test(decoded)) return decoded;
        }
      }
      if (u.pathname === '/url' && u.searchParams.get('q')) return u.searchParams.get('q');  // Google
      if (u.searchParams.get('uddg')) return u.searchParams.get('uddg');                     // DuckDuckGo
    } catch { /* not a URL this can parse; leave it as it was */ }
    return href;
  };

  // What a title may not be. Both of these were observed on a live results
  // page: an anchor whose textContent swept up an injected <style> block, and
  // one whose text was escaped markup for a favicon placeholder. A model shown
  // `.css-rwvbo4{display:-webkit-box...}` as the name of a page cannot tell
  // which result it is looking at.
  const junk = (text) =>
    !text
    || /^[.#][\w-]+\s*\{/.test(text)          // a CSS rule
    || /\{[^}]*:[^}]*\}/.test(text)            // a declaration block anywhere
    || /^</.test(text);                        // markup as text

  const push = (rawHref, title, snippet) => {
    const href = unwrap(rawHref);
    if (!href || !/^https?:/i.test(href) || blocked(href) || seen.has(href)) return;
    const text = clean(title, 300);
    if (junk(text)) return;
    seen.add(href);
    out.push({ url: href, title: text, snippet: clean(snippet, 600) });
  };

  // The heading first, then the link's own rendered text. `innerText` rather
  // than `textContent` throughout: textContent includes the contents of
  // <style> and <script>, which is exactly how a stylesheet ends up being
  // reported as the title of a search result.
  const titleOf = (node, a) => {
    const heading = node && node.querySelector('h1, h2, h3, h4, [role=heading]');
    for (const candidate of [
      heading && heading.innerText,
      a.getAttribute('aria-label'),
      a.innerText,
      a.getAttribute('title'),
    ]) {
      const text = clean(candidate, 300);
      if (!junk(text)) return text;
    }
    return '';
  };

  for (const group of cfg.groups) {
    for (const node of document.querySelectorAll(group.result)) {
      const a = node.querySelector(group.link || 'a[href]') || node.querySelector('a[href]');
      if (!a) continue;
      const sn = group.snippet ? node.querySelector(group.snippet) : null;
      push(a.href, titleOf(node, a), sn ? sn.innerText : '');
      if (out.length >= cfg.limit) return out;
    }
    // A group that matched anything is the layout this page is using; trying
    // the next one as well would mix two layouts' ideas of the same result.
    if (out.length) return out;
  }

  // Nothing matched: the markup changed. A results page is still a list of
  // headings that link somewhere else, so find those.
  for (const el of document.querySelectorAll('h2 a[href], h3 a[href], a[href] h2, a[href] h3')) {
    const a = el.tagName === 'A' ? el : el.closest('a[href]');
    if (!a) continue;
    const block = a.closest('li, article, [data-testid], .result, div');
    let snippet = '';
    if (block) {
      const p = block.querySelector(
        'p, [class*="snippet"], [class*="description"], [data-result="snippet"]'
      );
      if (p) snippet = p.innerText;
    }
    push(a.href, titleOf(block, a), snippet);
    if (out.length >= cfg.limit) break;
  }
  return out;
}
"""


class SearchUnavailable(RuntimeError):
    """The headless backend cannot run here, with the reason a person can act on."""


class HeadlessSearcher:
    """One shared Chromium, many searches.

    Shared because five parallel searches are five pages in one browser, not
    five browsers — `research` issues exactly that, and the difference is a
    gigabyte. Reaped because a browser held open for an hour after the last
    question is a resource leak with a long fuse.
    """

    def __init__(
        self,
        engine: str | Engine = 'auto',
        *,
        headless: bool = True,
        patience: float = EXTRACT_PATIENCE,
    ) -> None:
        # `auto` is a list to walk; a name or an `Engine` is one engine and no
        # falling through. An `Engine` object is also how a test points this at
        # a page it controls — worth the branch, because the extraction is the
        # part most likely to be wrong and testing it against the live internet
        # tests the internet.
        if isinstance(engine, Engine):
            self.engines: tuple[Engine, ...] = (engine,)
        elif engine in ('auto', ''):
            self.engines = tuple(ENGINES[name] for name in ENGINE_ORDER)
        elif engine in ENGINES:
            self.engines = (ENGINES[engine],)
        else:
            known = ', '.join(('auto', *sorted(ENGINES)))
            raise SearchUnavailable(f'unknown search engine {engine!r}. Available: {known}.')
        self.headless = headless
        # How long to keep re-reading a page before calling it empty. A caller
        # that knows it is looking at a challenge page — a test, mostly — has no
        # reason to wait out the full patience to be told so.
        self.patience = patience

        self._pw: Any = None
        self._browser: Any = None
        self._lock = asyncio.Lock()
        self._in_flight = 0
        self._reaper: asyncio.Task[None] | None = None

    @property
    def engine(self) -> Engine:
        """The first engine it will try. For logging, and for saying which one answered."""
        return self.engines[0]

    # -- the browser --------------------------------------------------------

    async def _browser_ready(self) -> Any:
        async with self._lock:
            if self._browser is not None and self._browser.is_connected():
                return self._browser
            try:
                from playwright.async_api import async_playwright
            except ImportError as exc:
                raise SearchUnavailable(
                    'the headless search backend needs Playwright. Install it with '
                    "`pip install 'openmirror[browser]'` then `playwright install chromium`, "
                    'or set OPENMIRROR_SEARCH_BACKEND to brave, tavily or searxng.'
                ) from exc

            if self._pw is None:
                self._pw = await async_playwright().start()

            # The same browser the agent drives, because "which browser does
            # this use" having two answers is how one of them quietly stops
            # being maintained.
            from openmirror.agent.browsers import current

            choice = current()
            launch: dict[str, Any] = {'headless': self.headless, **choice.launch_kwargs()}
            if choice.engine == 'chromium':
                launch['args'] = [
                    '--disable-blink-features=AutomationControlled',
                    '--no-sandbox' if _in_container() else '--use-mock-keychain',
                ]
            try:
                self._browser = await getattr(self._pw, choice.engine).launch(**launch)
            except Exception as exc:  # noqa: BLE001 - playwright raises its own
                raise SearchUnavailable(
                    f'could not start {choice.label()} for headless search: {exc}. '
                    f'If the browser is missing, run `playwright install {choice.engine}`, '
                    'or choose another browser in settings.'
                ) from exc
            log.debug('headless search: using %s', choice.label())
            return self._browser

    async def close(self) -> None:
        if self._reaper and not self._reaper.done():
            self._reaper.cancel()
        self._reaper = None
        async with self._lock:
            browser, self._browser = self._browser, None
            pw, self._pw = self._pw, None
        if browser is not None:
            try:
                await browser.close()
            except Exception as exc:  # noqa: BLE001
                log.debug('headless search: closing the browser: %s', exc)
        if pw is not None:
            try:
                await pw.stop()
            except Exception as exc:  # noqa: BLE001
                log.debug('headless search: stopping playwright: %s', exc)

    def _touch(self) -> None:
        """Restart the idle countdown. Called as each search finishes."""
        if self._reaper and not self._reaper.done():
            self._reaper.cancel()
        self._reaper = asyncio.create_task(self._reap())

    async def _reap(self) -> None:
        try:
            await asyncio.sleep(IDLE_TIMEOUT)
        except asyncio.CancelledError:
            return
        if self._in_flight == 0:
            log.debug('headless search: idle for %.0fs, closing the browser', IDLE_TIMEOUT)
            await self.close()

    # -- searching ----------------------------------------------------------

    async def search(self, query: str, count: int = 8) -> list[dict[str, str]]:
        """Results from the first engine that will give any.

        One engine refusing is not an empty web, and reporting it as one is the
        failure this file cares most about: a model told the web has no answer
        stops searching and starts guessing. So the chain is walked, and the
        error — when every engine has refused — names what each one did.
        """
        browser = await self._browser_ready()
        self._in_flight += 1
        try:
            failures: list[str] = []
            for engine in self.engines:
                try:
                    return await self._search_one(browser, engine, query, count)
                except SearchUnavailable as exc:
                    if len(self.engines) == 1:
                        # An engine somebody named explicitly. Falling through
                        # to another would substitute a search engine they did
                        # not choose, which for anyone who picked one on privacy
                        # grounds is the opposite of helping.
                        raise
                    failures.append(f'{engine.name}: {exc}')
                    log.info('headless search: %s gave nothing, trying the next engine', engine.name)
            raise SearchUnavailable(
                'every search engine refused:\n  ' + '\n  '.join(failures)
                + '\nSet OPENMIRROR_SEARCH_ENGINE to one of '
                + f'{", ".join(sorted(ENGINES))} to try a single one, or configure a keyed '
                'backend with OPENMIRROR_SEARCH_BACKEND.'
            )
        finally:
            self._in_flight -= 1
            self._touch()

    async def _search_one(
        self, browser: Any, engine: Engine, query: str, count: int
    ) -> list[dict[str, str]]:
        # A context per search, so two searches never share a cookie jar and
        # neither of them ever sees the profile you log in with.
        context = await browser.new_context(
            user_agent=USER_AGENT,
            viewport={'width': 1280, 'height': 900},
            locale='en-US',
        )
        try:
            # Pictures and fonts are most of a results page's weight and none
            # of its text.
            await context.route('**/*', _text_only)
            page = await context.new_page()
            page.set_default_timeout(PAGE_TIMEOUT)
            url = engine.url.format(q=quote_plus(query))
            # Retried once. A dropped connection (ERR_NETWORK_CHANGED on a
            # laptop that just changed wifi) is not the engine refusing, and
            # treating it as one moves the chain on to a worse engine for a
            # reason that has nothing to do with the engine.
            for attempt in (1, 2):
                try:
                    await page.goto(url, wait_until='domcontentloaded', timeout=PAGE_TIMEOUT)
                    break
                except Exception as exc:  # noqa: BLE001 - playwright raises its own
                    if attempt == 2:
                        raise SearchUnavailable(f'could not load the results page: {exc}') from exc
                    log.debug('headless search: %s: %s — retrying once', engine.name, exc)
                    await page.wait_for_timeout(700)
            await self._decline_consent(page)

            config = {
                'groups': [dict(g) for g in engine.groups],
                'blocked': list(engine.blocked),
                'limit': max(1, min(count, 20)),
            }

            # Wait for a row, then read; and keep reading for a few seconds if
            # there is nothing, because both halves have been wrong before. The
            # wait is an optimisation rather than a precondition — extraction
            # has a selector-free fallback — and the retry is what stops a slow
            # render from being reported as an empty web.
            try:
                await page.wait_for_selector(engine.ready, timeout=10_000, state='attached')
            except Exception:  # noqa: BLE001 - a layout change is not fatal yet
                log.debug('headless search: %s never showed %s', engine.name, engine.ready)

            results: list[dict[str, str]] = []
            deadline = time.monotonic() + self.patience
            while True:
                results = await page.evaluate(EXTRACT_JS, config)
                if results or time.monotonic() >= deadline:
                    break
                await page.wait_for_timeout(int(EXTRACT_INTERVAL * 1000))

            if not results:
                title = (await page.title()) or ''
                where = (await page.evaluate('() => location.href'))[:120]
                raise SearchUnavailable(
                    f'no results on the page (title: {title[:60]!r}, at {where}) — '
                    'it is most likely showing a challenge'
                )
            log.debug('headless search: %s gave %d result(s)', engine.name, len(results))
            return [
                {'title': r.get('title', ''), 'url': r.get('url', ''), 'snippet': r.get('snippet', '')}
                for r in results
            ]
        finally:
            try:
                await context.close()
            except Exception as exc:  # noqa: BLE001
                log.debug('headless search: closing the context: %s', exc)

    async def _decline_consent(self, page: Any) -> None:
        """Get a consent wall out of the way by refusing it.

        Only refusals are clicked. Accepting on somebody's behalf is not a
        decision this code gets to make, and where refusing is not offered the
        wall simply stays — in which case the extraction finds nothing and says
        so, which is the honest outcome.
        """
        for label in DECLINE:
            try:
                button = page.get_by_role('button', name=label, exact=False)
                if await button.count() == 0:
                    continue
                await button.first.click(timeout=2_000)
                log.debug('headless search: declined a consent dialog (%r)', label)
                return
            except Exception:  # noqa: BLE001 - a dialog that will not close is not fatal
                continue


async def _text_only(route: Any) -> None:
    """Drop everything that is not text on the way in.

    A results page is a few kilobytes of words and a megabyte of pictures,
    thumbnails and icon fonts, none of which are extracted. Both calls are
    guarded because a page that navigates or closes mid-flight leaves routes
    that can no longer be answered, and that is a normal race rather than an
    error worth surfacing.
    """
    try:
        if route.request.resource_type in ('image', 'media', 'font'):
            await route.abort()
        else:
            await route.continue_()
    except Exception as exc:  # noqa: BLE001 - the page moved on
        log.debug('headless search: route %s: %s', route.request.resource_type, exc)


def _in_container() -> bool:
    """Whether Chromium's sandbox is going to fail for want of privileges.

    Docker without `--cap-add SYS_ADMIN` is the common case, and `--no-sandbox`
    there is the difference between searching and not. It is not passed
    anywhere else, because dropping a browser's sandbox on a normal desktop to
    save a conditional would be a poor trade.
    """
    from pathlib import Path

    return Path('/.dockerenv').exists() or Path('/run/.containerenv').exists()


# One searcher per engine, for the same reason the browser is shared: the
# alternative is a Chromium per caller.
_searchers: dict[tuple[str, bool], HeadlessSearcher] = {}


def searcher(engine: str = 'auto', *, headless: bool = True) -> HeadlessSearcher:
    key = (engine, headless)
    if key not in _searchers:
        _searchers[key] = HeadlessSearcher(engine, headless=headless)
    return _searchers[key]


async def shutdown() -> None:
    """Close every shared browser. Called when the daemon stops."""
    for live in list(_searchers.values()):
        await live.close()
    _searchers.clear()
