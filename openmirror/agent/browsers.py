"""Which browser this install drives, and where that choice is kept.

Everything here exists because "a browser" is not one thing. Playwright ships
its own Chromium, Firefox and WebKit builds; it can also drive a Google Chrome
or a Microsoft Edge that is already installed, by channel; and it can be pointed
at an arbitrary binary. Those are different answers for different reasons, and
the reasons are not ours to guess:

  * Playwright's Chromium is the default because it is the one that is
    definitely there and definitely matches the library version.
  * Real Chrome has the codecs Chromium lacks and is what a site fingerprints
    as an ordinary visitor, which matters on the pages people actually want
    automated — a bank, an airline, a shop.
  * Firefox and WebKit are the two engines Chromium cannot imitate. Somebody
    checking how a page behaves outside Blink needs one of them.
  * A named executable is for a Chromium fork that is neither — Brave,
    Chromium-from-the-distribution, an ungoogled build.

**The choice is a file, not an environment variable.** It is the same argument
`providers/connections.py` makes for connections: a setting you can only change
by editing `.env` and restarting is not a setting a person changes at four in
the afternoon when a site starts refusing the default. The environment still
provides the *default*, so an install managed by a file keeps working exactly as
it did; a saved choice overrides it, and the API says which of the two is in
force.

**A choice that cannot launch is refused when it is made**, not when the agent
first tries to browse. Otherwise the error arrives halfway through a task, from
inside a tool, as a Playwright traceback — and the person reading it is trying
to book a flight, not debug a browser installation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# The Playwright engines. A channel or an executable only makes sense with
# chromium, which is the one with forks and installed variants.
ENGINES = ('chromium', 'firefox', 'webkit')

# Channels Playwright knows how to find by name, and the binaries they are,
# so an install can be reported as present without launching it.
CHANNELS: dict[str, tuple[str, ...]] = {
    'chrome': ('google-chrome', 'google-chrome-stable', 'chrome'),
    'chrome-beta': ('google-chrome-beta',),
    'chromium-channel': ('chromium', 'chromium-browser'),
    'msedge': ('microsoft-edge', 'microsoft-edge-stable', 'msedge'),
    'msedge-beta': ('microsoft-edge-beta',),
}

# What the settings UI offers, in the order it offers them. `id` is what gets
# stored; the rest is for a person choosing.
CATALOG: tuple[dict[str, str], ...] = (
    {
        'id': 'chromium',
        'label': "Chromium (Playwright's own)",
        'note': 'Bundled with the browser extra and always version-matched. The default.',
    },
    {
        'id': 'chrome',
        'label': 'Google Chrome',
        'note': 'Your installed Chrome. Has the codecs Chromium lacks, and looks like an '
                'ordinary visitor to sites that check.',
    },
    {
        'id': 'msedge',
        'label': 'Microsoft Edge',
        'note': 'Your installed Edge. Chromium underneath, so the tools behave identically.',
    },
    {
        'id': 'firefox',
        'label': 'Firefox',
        'note': "Playwright's Gecko build. A different engine, not a reskinned Chromium.",
    },
    {
        'id': 'webkit',
        'label': 'WebKit',
        'note': 'What Safari renders with. The closest thing to testing on an iPhone from here.',
    },
    {
        'id': 'custom',
        'label': 'Another Chromium build',
        'note': 'Brave, ungoogled-chromium, a distribution build — give the path to the binary.',
    },
)


class BrowserUnavailable(RuntimeError):
    """The chosen browser is not installed, or not where it was said to be."""


@dataclass(slots=True)
class BrowserChoice:
    """One decision about what to launch.

    Three fields rather than one string because Playwright wants them
    separately, and collapsing them into `'chrome'` would mean parsing that
    back apart at every launch site — which is where the spelling would
    eventually diverge.
    """

    # Which Playwright browser type: chromium, firefox or webkit.
    engine: str = 'chromium'
    # An installed variant Playwright can find by name. chromium only.
    channel: str = ''
    # An explicit binary. Wins over `channel`, and is why `engine` stays
    # chromium for a fork: Brave is Chromium, whatever its icon says.
    executable: str = ''

    @property
    def id(self) -> str:
        """The catalog id this corresponds to, for the UI's benefit."""
        if self.executable:
            return 'custom'
        if self.channel:
            return self.channel
        return self.engine

    def label(self) -> str:
        if self.executable:
            return f'{Path(self.executable).name} ({self.executable})'
        for entry in CATALOG:
            if entry['id'] == self.id:
                return entry['label']
        return self.id

    def launch_kwargs(self) -> dict[str, Any]:
        """What to add to a `launch` or `launch_persistent_context` call.

        Never both: Playwright refuses a channel and an executable together,
        and an executable is the more specific of the two.
        """
        if self.executable:
            return {'executable_path': self.executable}
        if self.channel:
            return {'channel': self.channel}
        return {}

    def validate(self) -> None:
        """Refuse a choice that cannot launch, at the moment it is made."""
        if self.engine not in ENGINES:
            raise BrowserUnavailable(
                f'{self.engine!r} is not a browser engine. One of: {", ".join(ENGINES)}.'
            )
        if self.executable:
            path = Path(self.executable).expanduser()
            if not path.is_file():
                raise BrowserUnavailable(f'{path} is not a file.')
            # Checked because a path that is not executable fails inside
            # Playwright with an error about a missing library.
            import os

            if not os.access(path, os.X_OK):
                raise BrowserUnavailable(f'{path} is not executable.')
            if self.engine != 'chromium':
                raise BrowserUnavailable(
                    'a named executable is only supported for chromium — a Firefox or WebKit '
                    'build has to be the one Playwright installed.'
                )
            return
        if self.channel:
            if self.engine != 'chromium':
                raise BrowserUnavailable(f'{self.engine} has no channels; only chromium does.')
            if self.channel not in CHANNELS:
                raise BrowserUnavailable(
                    f'{self.channel!r} is not a channel Playwright knows. '
                    f'One of: {", ".join(sorted(CHANNELS))}.'
                )
            if not channel_installed(self.channel):
                raise BrowserUnavailable(
                    f'{self.channel} does not appear to be installed on this machine. '
                    'Install it, or choose another browser.'
                )


def from_id(choice_id: str, executable: str = '') -> BrowserChoice:
    """Turn what the UI sends into a choice.

    One place that knows `'chrome'` means chromium-on-channel-chrome, so the
    eight callers that could have each decided that for themselves do not.
    """
    if choice_id == 'custom':
        if not executable.strip():
            raise BrowserUnavailable('choosing another build needs the path to its binary.')
        return BrowserChoice(engine='chromium', executable=str(Path(executable).expanduser()))
    if choice_id in ENGINES:
        return BrowserChoice(engine=choice_id)
    if choice_id in CHANNELS:
        return BrowserChoice(engine='chromium', channel=choice_id)
    raise BrowserUnavailable(f'unknown browser {choice_id!r}.')


def channel_installed(channel: str) -> bool:
    return any(shutil.which(binary) for binary in CHANNELS.get(channel, ()))


async def available() -> list[dict[str, Any]]:
    """The catalog, with whether each entry can actually be launched here.

    Installed state is read rather than probed: Playwright exposes the path of
    each engine it has downloaded, and a channel is a binary on PATH. Launching
    five browsers to populate a settings dialog would take ten seconds and
    several gigabytes of RAM to tell somebody what a `which` call already knows.
    """
    engines: dict[str, bool] = dict.fromkeys(ENGINES, False)
    missing_playwright = False
    try:
        from playwright.async_api import async_playwright

        pw = await async_playwright().start()
        try:
            paths: dict[str, str] = {}
            for name in ENGINES:
                try:
                    paths[name] = getattr(pw, name).executable_path
                except Exception as exc:  # noqa: BLE001 - an engine that was never downloaded
                    log.debug('browsers: %s is not installed: %s', name, exc)
        finally:
            await pw.stop()
        # Off the loop: five stat calls is nothing, but a settings dialog is not
        # a reason to block an event loop that is also carrying a voice session.
        engines.update(
            await asyncio.to_thread(
                lambda: {name: Path(path).exists() for name, path in paths.items()}
            )
        )
    except ImportError:
        missing_playwright = True

    rows: list[dict[str, Any]] = []
    for entry in CATALOG:
        row = dict(entry)
        if missing_playwright:
            row['installed'] = False
            row['why'] = "Playwright is not installed — pip install 'openmirror[browser]'"
        elif entry['id'] == 'custom':
            # Always offerable: whether the binary exists is checked when the
            # path is given, which is the only moment it can be checked.
            row['installed'] = True
        elif entry['id'] in ENGINES:
            row['installed'] = engines[entry['id']]
            if not row['installed']:
                row['why'] = f'not downloaded — run `playwright install {entry["id"]}`'
        else:
            row['installed'] = channel_installed(entry['id'])
            if not row['installed']:
                row['why'] = 'not found on this machine'
        rows.append(row)
    return rows


class BrowserSettings:
    """The saved choice, if there is one.

    A tiny JSON file, loaded on every read rather than cached. Caching it would
    save a few microseconds and introduce the bug where the dialog saves a
    setting that the next session does not use.
    """

    def __init__(self, path: Path, default: BrowserChoice | None = None) -> None:
        self.path = Path(path)
        self.default = default or BrowserChoice()

    def load(self) -> BrowserChoice:
        if not self.path.is_file():
            return self.default
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            # The environment's default rather than a failure: a corrupt
            # preferences file must not stop the agent from browsing.
            log.warning('browser settings: could not read %s: %s', self.path, exc)
            return self.default
        choice = BrowserChoice(
            engine=str(raw.get('engine') or self.default.engine),
            channel=str(raw.get('channel') or ''),
            executable=str(raw.get('executable') or ''),
        )
        try:
            choice.validate()
        except BrowserUnavailable as exc:
            log.warning(
                'browser settings: the saved choice (%s) is not usable — %s. Using %s instead.',
                choice.id, exc, self.default.id,
            )
            return self.default
        return choice

    def save(self, choice: BrowserChoice) -> BrowserChoice:
        choice.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(choice), indent=2) + '\n')
        return choice

    def clear(self) -> BrowserChoice:
        """Back to whatever the environment says. Returns that."""
        self.path.unlink(missing_ok=True)
        return self.default

    @property
    def saved(self) -> bool:
        return self.path.is_file()


def settings() -> BrowserSettings:
    """The store for this install, with the environment as its default."""
    from openmirror.config import config

    return BrowserSettings(
        config.data_dir / 'browser.json',
        BrowserChoice(
            engine=config.browser_engine,
            channel=config.browser_channel,
            executable=config.browser_executable,
        ),
    )


def current() -> BrowserChoice:
    """What to launch right now. The one function the launch sites call."""
    return settings().load()
