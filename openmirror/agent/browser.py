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

That last point has a consequence that took a second session to notice.
Chromium locks a profile directory, so the *second* session to open a browser
cannot have the signed-in one — and since sessions outlive their client here,
"the second session" includes one somebody left open yesterday. Failing would
be wrong (the answer to "look something up" should not be "close your other
tab"), and silently sharing is impossible. So the second one gets a fresh
profile, and is *told* it is not signed in to anything — because an agent that
does not know it is logged out will read a login wall as the site being
broken.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
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

    // Whether this field holds a secret, decided *here*, before the value
    // crosses out of the page. It duplicates `_SECRET_NAME` in
    // tools/browser.py and that duplication is deliberate: the Python side is
    // the authority on how a call is *graded*, but by the time it sees this
    // list the value has already been read out of the browser, put in a
    // result and handed to a model. The only place a secret can be kept out
    // of all three is on this side of the boundary.
    const secretish = /(pass(word|wd|phrase)?|pwd|otp|mfa|2fa|totp|one[-_ ]?time|card[-_ ]?(number|num)?|ccnum|cc[-_ ]?number|credit[-_ ]?card|cvv|cvc|csc|security[-_ ]?code|verification[-_ ]?code|ssn|social[-_ ]?security|passport|pin|secret|api[-_ ]?key|token|private[-_ ]?key|seed[-_ ]?phrase|mnemonic|recovery[-_ ]?phrase)/i;
    const naming = [el.getAttribute('name'), el.id, el.getAttribute('placeholder'),
                    el.getAttribute('aria-label')].filter(Boolean).join(' ');
    const isSecret = el.type === 'password' || secretish.test(naming);

    const label = (
      el.getAttribute('aria-label') ||
      el.innerText ||
      // Never the value on a secret field. This fallback was how a typed
      // password came back out: with no aria-label and no inner text, the
      // element's *label* became whatever had been typed into it, and the
      // separate redaction of `value` below did nothing to stop it.
      (isSecret ? '' : el.value) ||
      el.getAttribute('placeholder') ||
      el.getAttribute('title') ||
      el.getAttribute('name') ||
      ''
    ).trim().replace(/\\s+/g, ' ').slice(0, 120);

    const key = el.tagName + '|' + label + '|' + Math.round(r.top) + '|' + Math.round(r.left);
    if (seen.has(key)) continue;
    seen.add(key);

    n += 1;
    el.setAttribute('data-openmirror-ref', String(n));
    out.push({
      ref: n,
      tag: el.tagName.toLowerCase(),
      type: (el.getAttribute('type') || '').toLowerCase(),
      name: el.getAttribute('name') || '',
      id: el.id || '',
      role: el.getAttribute('role') || '',
      label,
      href: el.getAttribute('href') || '',
      // Redacted by what the field *is*, not by its input type: a card
      // number is `type=text` and was coming back in full.
      value: (isSecret ? '' : (el.value || '')).slice(0, 60),
      secret: isSecret,
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
    profile_dir: Path = field(default_factory=lambda: Path.home() / '.openmirror' / 'browser')
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
        # False once this session has had to fall back to a throwaway profile
        # because another one holds the signed-in one. Read by the tools, so
        # the model is told rather than left to work it out from a login page.
        self.signed_in_profile = True
        self._temp_profile: Any = None
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
                    'headless': self.config.headless,
                    'viewport': {'width': width, 'height': height},
                    'user_agent': self.config.user_agent,
                    'args': ['--disable-blink-features=AutomationControlled'],
                }
                launch['user_data_dir'] = str(await self._profile_dir())
                if self.config.env:
                    # Inherited and overlaid rather than replaced: a bare
                    # environment loses PATH, HOME and the XDG variables, and
                    # Chromium fails to start in ways that read as a Playwright
                    # bug rather than as a missing variable.
                    #
                    # An empty value means remove. That is how a stage says
                    # "there is no Wayland here" — and it has to be able to,
                    # because Chromium prefers Wayland when it finds it and
                    # would put its window on the screen the person is using
                    # while the agent screenshots an empty X display.
                    merged = {**os.environ, **self.config.env}
                    launch['env'] = {k: v for k, v in merged.items() if v != ''}
                    if self.config.env.get('DISPLAY'):
                        # Told rather than inferred. Chromium's own detection
                        # reads the environment we have just edited, and being
                        # explicit costs nothing and removes a guess.
                        launch['args'] = [*launch['args'], '--ozone-platform=x11']
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

    async def _profile_dir(self) -> Path:
        """The shared profile, or a throwaway when something else holds it.

        Chromium's lock is a file in the directory and it is not advisory —
        launching against a locked profile fails with a message about "an
        existing browser session", which reads like a bug in this program.
        Checking first and stepping aside turns that into a fact the agent can
        work with.
        """
        import tempfile

        shared = self.config.profile_dir
        shared.mkdir(parents=True, exist_ok=True)

        # `SingletonLock` is a symlink Chromium leaves while a profile is in
        # use, and removes on a clean exit. A stale one after a crash points
        # at a pid that is gone, so it is checked rather than trusted.
        lock = shared / 'SingletonLock'
        if _profile_is_busy(lock):
            self.signed_in_profile = False
            self._temp_profile = tempfile.mkdtemp(prefix='openmirror-browser-')
            log.info('the signed-in browser profile is in use; this session gets a fresh one')
            return Path(self._temp_profile)
        return shared

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
        locator = page.locator(f'[data-openmirror-ref="{ref}"]')
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
            if self._temp_profile:
                # A throwaway profile is exactly that: it holds whatever was
                # signed into during this session and nothing anyone asked to
                # keep, and it is a few hundred megabytes.
                shutil.rmtree(self._temp_profile, ignore_errors=True)
                self._temp_profile = None


def _profile_is_busy(lock: Path) -> bool:
    """Whether a Chromium is currently holding this profile.

    The lock is a symlink whose target is `<host>-<pid>`. A crashed Chromium
    leaves one behind pointing at a pid that no longer exists, and treating
    that as busy would mean every session after a crash silently loses its
    logins — so the pid is checked.
    """
    try:
        target = os.readlink(lock)
    except OSError:
        return False

    pid = target.rsplit('-', 1)[-1]
    if not pid.isdigit():
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Somebody else's process, which is still a process holding it.
        return True
    return True
