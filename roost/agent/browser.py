"""A real browser the agent can drive.

Three decisions worth stating.

**The profile is persistent.** Shopping, or anything else on a site you have
an account with, is impossible from a fresh incognito context — you would be
logged out every time. So the profile lives on disk and the agent inherits
whatever sessions are in it. That is also the safest arrangement for
credentials: **you** log in once, by hand, and the agent never sees a password
because it never needs one.

**Actions are by element reference, not coordinates.** `read_page` numbers the
interactive elements and hands back a list; the click tool takes one of those
numbers. Coordinates go stale the moment a page reflows, an ad loads or a
banner appears, and a stale coordinate does not fail — it clicks whatever is
there now, which on a checkout page is exactly the wrong failure mode.

**One browser per agent session.** Two sessions sharing a page would fight
over navigation, and a page's state is part of the conversation's state.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# What a page hands back, after being asked what is on it.
INTERACTIVE_JS = """
() => {
  const out = [];
  const seen = new Set();
  const sel = 'a[href], button, input, select, textarea, [role=button], [role=link], [role=tab], [onclick], [contenteditable=true]';
  let n = 0;
  for (const el of document.querySelectorAll(sel)) {
    const r = el.getBoundingClientRect();
    // Off-screen or zero-size elements are not things a person could click,
    // and offering them to the model invites it to try.
    if (r.width < 2 || r.height < 2) continue;
    const style = getComputedStyle(el);
    if (style.visibility === 'hidden' || style.display === 'none' || style.opacity === '0') continue;

    const label = (
      el.getAttribute('aria-label') ||
      el.innerText ||
      el.value ||
      el.getAttribute('placeholder') ||
      el.getAttribute('title') ||
      el.getAttribute('name') ||
      ''
    ).trim().replace(/\\s+/g, ' ').slice(0, 120);

    const key = el.tagName + '|' + label + '|' + Math.round(r.top) + '|' + Math.round(r.left);
    if (seen.has(key)) continue;
    seen.add(key);

    n += 1;
    el.setAttribute('data-roost-ref', String(n));
    out.push({
      ref: n,
      tag: el.tagName.toLowerCase(),
      type: (el.getAttribute('type') || '').toLowerCase(),
      name: el.getAttribute('name') || '',
      id: el.id || '',
      role: el.getAttribute('role') || '',
      label,
      href: el.getAttribute('href') || '',
      value: (el.type === 'password' ? '' : (el.value || '')).slice(0, 60),
      checked: el.checked === true,
      disabled: el.disabled === true,
      inViewport: r.top >= 0 && r.top < innerHeight,
    });
    if (n >= 300) break;
  }
  return out;
}
"""


@dataclass
class BrowserConfig:
    # A real profile directory. Log in by hand once; the agent inherits it and
    # never handles a password.
    profile_dir: Path = field(default_factory=lambda: Path.home() / '.roost' / 'browser')
    headless: bool = True
    # Headful is the honest default for anything transactional: you can watch
    # what it is doing and take the mouse off it.
    viewport: tuple[int, int] = (1280, 900)
    timeout_ms: int = 30_000
    user_agent: str | None = None
    # Environment for the browser process — in practice a DISPLAY, which is
    # what puts a real headful Chromium onto the agent's own X server rather
    # than onto the screen the person is using. A visible browser on a display
    # nobody is looking at is the best of both: the agent can see it with the
    # desktop tools, and it cannot steal focus from anyone.
    env: dict[str, str] = field(default_factory=dict)


class BrowserSession:
    """Owns one browser for one agent session."""

    def __init__(self, config: BrowserConfig | None = None) -> None:
        self.config = config or BrowserConfig()
        self._pw: Any = None
        self._context: Any = None
        self._page: Any = None
        self._lock = asyncio.Lock()
        # What the last read saw, keyed by ref. The click and type tools grade
        # their risk from this without awaiting, which they must: `assess` is
        # synchronous by design, so that an approval decision cannot itself
        # touch the page it is deciding about.
        self.last_elements: dict[int, dict[str, Any]] = {}

    async def page(self) -> Any:
        """The current page, starting the browser on first use.

        Lazy because most sessions never touch a browser, and paying 300 ms and
        200 MB for one that is not used is a poor trade.
        """
        async with self._lock:
            if self._page is not None and not self._page.is_closed():
                return self._page

            if self._context is None:
                from playwright.async_api import async_playwright

                self._pw = await async_playwright().start()
                self.config.profile_dir.mkdir(parents=True, exist_ok=True)
                width, height = self.config.viewport
                # A persistent context rather than launch()+new_context(): it is
                # what keeps cookies, logins and local storage across runs.
                launch: dict[str, Any] = {
                    'user_data_dir': str(self.config.profile_dir),
                    'headless': self.config.headless,
                    'viewport': {'width': width, 'height': height},
                    'user_agent': self.config.user_agent,
                    'args': ['--disable-blink-features=AutomationControlled'],
                }
                if self.config.env:
                    # Inherited and overlaid rather than replaced: a bare
                    # environment loses PATH, HOME and the XDG variables, and
                    # Chromium fails to start in ways that read as a Playwright
                    # bug rather than as a missing variable.
                    launch['env'] = {**os.environ, **self.config.env}
                self._context = await self._pw.chromium.launch_persistent_context(**launch)
                self._context.set_default_timeout(self.config.timeout_ms)

            pages = self._context.pages
            self._page = pages[0] if pages else await self._context.new_page()

            # A new page means the old refs are meaningless. Clearing here
            # rather than trusting callers means a stale ref grades as unknown
            # — which escalates — instead of inheriting the old element's
            # harmless-looking label.
            self._page.on('framenavigated', lambda frame: self.last_elements.clear()
                          if frame == self._page.main_frame else None)
            return self._page

    async def elements(self) -> list[dict[str, Any]]:
        page = await self.page()
        try:
            found = await page.evaluate(INTERACTIVE_JS)
        except Exception:  # noqa: BLE001 - a navigation mid-evaluate is normal
            await page.wait_for_timeout(500)
            found = await page.evaluate(INTERACTIVE_JS)
        self.last_elements = {e['ref']: e for e in found}
        return found

    async def find(self, ref: int) -> Any:
        """The handle for a ref from the last read_page."""
        page = await self.page()
        locator = page.locator(f'[data-roost-ref="{ref}"]')
        if await locator.count() == 0:
            raise LookupError(
                f'no element {ref} on this page — the page has changed since you read it. '
                'Read it again and use a fresh reference.'
            )
        return locator.first

    async def close(self) -> None:
        try:
            if self._context is not None:
                await self._context.close()
            if self._pw is not None:
                await self._pw.stop()
        except Exception:  # noqa: BLE001
            log.debug('browser teardown was not clean', exc_info=True)
        finally:
            self._pw = self._context = self._page = None
