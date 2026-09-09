"""Installing software, and finding out what kind of machine this is.

"Install Steam" means five different commands depending on where it is said,
and on one of those machines it means nothing at all because Steam is already
there. So these tools do not take a command — they take a *name*, and the
platform decides what that means. See `roost/system.py` for the table.

Three things this is careful about, each because the alternative is an agent
that hangs or breaks something:

**It never runs a command that will wait for input.** A package manager asked
to install something interactively stops at "Do you want to continue? [Y/n]"
and sits there until the tool times out, in a terminal nobody is reading. Every
manager here is invoked with its own non-interactive flag.

**It refuses rather than hangs when it cannot elevate.** `sudo` on a machine
that wants a password will not fail — it will block on a prompt going nowhere.
That is worse than an error, so the elevation route is worked out first and a
system-wide install is refused, with the per-user alternative named, when
there is no way to become root without a person.

**It will not fight an immutable system.** On SteamOS and the ostree family
the root filesystem is read-only by design, and an agent that meets `apt` failing
there will reach for `steamos-readonly disable` if nothing tells it not to.
Something tells it not to.
"""

from __future__ import annotations

import asyncio
from typing import Any

from roost.agent.tools.base import Assessment, Output, Tool, ToolContext, ToolError, truncate
from roost.protocol.agent import Risk
from roost.system import Manager, Platform, detect, recipe_for, route_for

# Installing a large package is minutes, not seconds.
INSTALL_TIMEOUT = 900
SEARCH_TIMEOUT = 120


class _SystemTool(Tool):
    def __init__(self, host: Platform | None = None) -> None:
        # Detected once per session: none of it changes while a session runs,
        # and `detect()` shells out to work out the elevation route.
        self.host = host or detect()


def _elevate(host: Platform, manager: Manager) -> list[str]:
    """The prefix that makes a command run as root here, if it needs one."""
    if not manager.needs_elevation or host.elevation == 'root':
        return []
    if host.elevation == 'sudo-nopasswd':
        return ['sudo', '-n']
    if host.elevation == 'pkexec':
        return ['pkexec']
    if host.elevation == 'uac':
        # winget asks Windows for elevation itself, through a dialog on the
        # person's own screen. Nothing to prefix.
        return []
    raise ToolError(
        f'{manager.name} installs for the whole machine and there is no way to become root '
        f'here without a person: {host.elevation_note} '
        + (
            'Use flatpak instead — it installs into their home directory and needs no '
            'elevation. '
            if host.manager('flatpak')
            else ''
        )
        + 'Or give them the exact command to run themselves.'
    )


class SystemInfoTool(_SystemTool):
    name = 'system_info'
    description = (
        'What machine this is: operating system, distribution, desktop, whether the root '
        'filesystem is writable, which package managers exist and how it can elevate. '
        'Ask this before installing anything or changing a system setting — the same '
        'instruction means different commands on different machines, and on some of them '
        'the obvious command is the wrong one.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'about': {
                'type': 'string',
                'description': 'Optionally, something you are about to install — '
                               '"steam", "epic games store". Answers how it would be done here.',
            },
        },
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        return Assessment(risk=Risk.READ, summary='ask what kind of machine this is')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        host = self.host
        lines = [
            f'{host.system}'
            + (f' · {host.distro} {host.version}'.rstrip() if host.distro else '')
            + (f' (like {", ".join(host.like)})' if host.like else ''),
        ]
        if host.desktop or host.session:
            lines.append(f'Desktop: {host.desktop or "unknown"} on {host.session or "unknown"}')
        if host.steam_deck:
            lines.append('This is a Steam Deck. Steam is already installed.')
        if host.immutable:
            lines.append(
                'The root filesystem is read-only by design. The system package managers '
                'below exist but will not work, and turning the protection off is not the '
                'answer — install per-user things with flatpak instead.'
            )

        lines.append('')
        lines.append('Package managers:')
        for manager in host.managers:
            scope = 'whole machine, needs root' if manager.needs_elevation else 'this user only'
            lines.append(f'  {manager.name} — {scope}' + (f'. {manager.note}' if manager.note else ''))
        if not host.managers:
            lines.append('  none found')

        lines.append('')
        lines.append(f'Elevation: {host.elevation} — {host.elevation_note}')

        about = (args.get('about') or '').strip()
        if about:
            lines.append('')
            recipe = recipe_for(about)
            if recipe is None:
                lines.append(
                    f'Nothing is known about {about!r} by name. Use package_search to look '
                    'for it in the managers above.'
                )
            else:
                route = route_for(recipe, host)
                if route is None:
                    lines.append(f'{about!r}: nothing here can install it.')
                else:
                    lines.append(f'{about!r}: {route.manager} install {route.package}')
                    if route.note:
                        lines.append(f'  {route.note}')
                if recipe.note:
                    lines.append(f'  {recipe.note}')

        return Output(content='\n'.join(lines), display=host.to_json())


class PackageSearchTool(_SystemTool):
    name = 'package_search'
    description = (
        'Search this machine for an installable package by name. Use it when system_info '
        'does not already know what you are looking for. Searching costs nothing and '
        'changes nothing.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'query': {'type': 'string'},
            'manager': {'type': 'string', 'description': 'Which one. Default: try each in turn.'},
        },
        'required': ['query'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        query = (args.get('query') or '').strip()
        if not query:
            return Assessment(risk=Risk.NETWORK, summary='', invalid='query is required')
        return Assessment(risk=Risk.NETWORK, summary=f'search for {query!r}')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        query = args['query'].strip()
        wanted = args.get('manager')
        managers = [m for m in self.host.managers if not wanted or m.name == wanted]
        if not managers:
            raise ToolError(
                f'no package manager called {wanted!r} here. There is: '
                + (', '.join(m.name for m in self.host.managers) or 'none')
            )

        chunks: list[str] = []
        for manager in managers:
            argv = [*manager.search, query]
            code, out = await _run(argv, SEARCH_TIMEOUT)
            body = out.strip()
            if code == 0 and body:
                chunks.append(f'--- {manager.name}\n{body}')
            elif code != 0:
                chunks.append(f'--- {manager.name}: search failed ({body[:200] or "no output"})')

        if not chunks:
            return Output(content=f'Nothing matching {query!r} in: '
                                  + ', '.join(m.name for m in managers))
        text, cut = truncate('\n\n'.join(chunks), 20_000, keep='head')
        return Output(content=text, truncated=cut)


class PackageInstallTool(_SystemTool):
    name = 'package_install'
    description = (
        'Install something. Give it a plain name — "steam", "discord", "epic games store" — '
        'and it works out what that means on this machine, or give an exact package and the '
        'manager to use. Tell the person what it is about to install and where it comes '
        'from before you call this, especially when the answer is a different program from '
        'the one they named.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'what': {'type': 'string', 'description': 'What to install, by name.'},
            'manager': {'type': 'string', 'description': 'Override the chosen manager.'},
            'package': {'type': 'string', 'description': 'Override the exact package id.'},
        },
        'required': ['what'],
    }

    def _plan(self, args: dict[str, Any]) -> tuple[Manager, str, str]:
        """(manager, package, why) for this request, or a ToolError explaining."""
        what = (args.get('what') or '').strip()
        host = self.host

        name = args.get('manager')
        package = (args.get('package') or '').strip()
        note = ''

        if not package or not name:
            recipe = recipe_for(what)
            route = route_for(recipe, host) if recipe else None
            if route is None:
                raise ToolError(
                    f'nothing here knows how to install {what!r}. Use package_search to find '
                    'the package, then pass `manager` and `package` explicitly.'
                )
            name = name or route.manager
            package = package or route.package
            note = ' '.join(filter(None, (route.note, recipe.note if recipe else '')))

        manager = host.manager(name)
        if manager is None:
            available = ', '.join(m.name for m in host.managers) or 'none'
            raise ToolError(f'{name!r} is not installed here. Available: {available}')

        if host.immutable and manager.needs_elevation:
            alternative = 'flatpak' if host.manager('flatpak') else None
            raise ToolError(
                f'{manager.name} changes the root filesystem, which is read-only on this '
                'system by design. Do not try to disable that. '
                + (f'Use {alternative} instead.' if alternative else 'There is no writable alternative here.')
            )

        return manager, package, note

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        if not (args.get('what') or '').strip():
            return Assessment(risk=Risk.NETWORK, summary='', invalid='say what to install')
        try:
            manager, package, _ = self._plan(args)
        except ToolError as exc:
            return Assessment(risk=Risk.NETWORK, summary='', invalid=str(exc))

        # Graded by what actually runs. A per-user install fetches and unpacks
        # into a home directory, which is the same shape as any other network
        # fetch. A system one runs the package's own install scripts as root,
        # which is arbitrary code execution by another name — so it is EXECUTE,
        # and the summary says "as root" because that is the part a person
        # deciding needs to see.
        if manager.needs_elevation:
            return Assessment(
                risk=Risk.EXECUTE,
                summary=f'install {package} with {manager.name}, as root, for the whole machine',
            )
        return Assessment(
            risk=Risk.NETWORK,
            summary=f'install {package} with {manager.name}, for this user only',
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        manager, package, note = self._plan(args)
        prefix = _elevate(self.host, manager)

        # Flatpak needs a remote named before the id, and flathub is where
        # every id in the recipe table lives.
        tail = ['flathub', package] if manager.name == 'flatpak' else [package]
        argv = [*prefix, *manager.install, *tail, *manager.assume_yes]

        await ctx.emit(f'$ {" ".join(argv)}\n', 'stdout')
        code, out = await _run(argv, INSTALL_TIMEOUT, emit=ctx.emit)

        if code != 0:
            raise ToolError(
                f'{manager.name} failed (exit {code}). '
                + (f'{note} ' if note else '')
                + f'Output:\n{truncate(out, 4000, keep="tail")[0]}'
            )

        return Output(
            content=f'Installed {package} with {manager.name}.' + (f'\n{note}' if note else ''),
            display={'manager': manager.name, 'package': package, 'scope': manager.scope},
        )


class PackageRemoveTool(_SystemTool):
    name = 'package_remove'
    description = 'Uninstall a package. Give the exact package id and the manager that has it.'
    input_schema = {
        'type': 'object',
        'properties': {
            'package': {'type': 'string'},
            'manager': {'type': 'string'},
        },
        'required': ['package', 'manager'],
    }

    REMOVE = {
        'apt': ('apt-get', 'remove'), 'dnf': ('dnf', 'remove'), 'pacman': ('pacman', '-R'),
        'zypper': ('zypper', 'remove'), 'apk': ('apk', 'del'),
        'flatpak': ('flatpak', 'uninstall', '--user'), 'snap': ('snap', 'remove'),
        'winget': ('winget', 'uninstall'), 'choco': ('choco', 'uninstall'),
        'scoop': ('scoop', 'uninstall'), 'brew': ('brew', 'uninstall'),
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        package = (args.get('package') or '').strip()
        name = (args.get('manager') or '').strip()
        if not package or not name:
            return Assessment(risk=Risk.DESTRUCTIVE, summary='', invalid='package and manager are required')
        if name not in self.REMOVE:
            return Assessment(risk=Risk.DESTRUCTIVE, summary='',
                              invalid=f'do not know how to remove with {name!r}')
        manager = self.host.manager(name)
        if manager is None:
            return Assessment(risk=Risk.DESTRUCTIVE, summary='', invalid=f'{name!r} is not installed here')

        # Destructive, always. Removing a package can take its dependents with
        # it, and on a system manager that can mean a desktop environment.
        where = 'from the whole machine' if manager.needs_elevation else 'for this user'
        return Assessment(risk=Risk.DESTRUCTIVE, summary=f'remove {package} with {name} {where}')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        manager = self.host.manager(args['manager'])
        if manager is None:
            raise ToolError(f'{args["manager"]!r} is not installed here')
        prefix = _elevate(self.host, manager)
        argv = [*prefix, *self.REMOVE[manager.name], args['package'], *manager.assume_yes]

        await ctx.emit(f'$ {" ".join(argv)}\n', 'stdout')
        code, out = await _run(argv, INSTALL_TIMEOUT, emit=ctx.emit)
        if code != 0:
            raise ToolError(f'{manager.name} failed (exit {code}):\n{truncate(out, 3000, keep="tail")[0]}')
        return Output(content=f'Removed {args["package"]}.')


async def _run(
    argv: list[str],
    timeout: int,  # noqa: ASYNC109 - a subprocess deadline, which is what wait_for is for
    emit: Any = None,
) -> tuple[int, str]:
    """Run a command, streaming as it goes, and never waiting for input.

    stdin is closed rather than inherited. A manager that decides to ask a
    question then gets EOF and gives up, which is a failure the model can read
    — whereas an inherited stdin means it waits, forever, on a terminal nobody
    is looking at.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**_env()},
        )
    except FileNotFoundError as exc:
        raise ToolError(f'{argv[0]} is not installed here') from exc
    except OSError as exc:
        raise ToolError(f'could not run {argv[0]}: {exc}') from exc

    collected: list[str] = []
    try:
        async def pump() -> None:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                chunk = raw.decode('utf-8', 'replace')
                collected.append(chunk)
                if emit is not None:
                    await emit(chunk, 'stdout')

        await asyncio.wait_for(pump(), timeout=timeout)
        await asyncio.wait_for(proc.wait(), timeout=30)
    except TimeoutError:
        proc.kill()
        raise ToolError(
            f'{argv[0]} did not finish within {timeout}s and was stopped. It may have been '
            'waiting for input, or the download may simply be very large.'
        ) from None

    return proc.returncode or 0, ''.join(collected)


def _env() -> dict[str, str]:
    """The environment a package manager should run in.

    `DEBIAN_FRONTEND=noninteractive` is the one that matters: without it apt
    opens a full-screen configuration dialog on some upgrades and blocks
    until somebody answers it.
    """
    import os

    return {
        **os.environ,
        'DEBIAN_FRONTEND': 'noninteractive',
        'NEEDRESTART_MODE': 'a',
    }





# ---------------------------------------------------------------------------
# Display settings
# ---------------------------------------------------------------------------


class DisplayInfoTool(_SystemTool):
    name = 'display_info'
    description = (
        'The monitors attached to this machine, and how a display setting like HDR is '
        'changed on this desktop — by command, or by opening a settings window. Ask this '
        'before trying to change one: there is no command that works everywhere, and '
        'guessing at one produces a "command not found" and then another guess.'
    )
    input_schema = {'type': 'object', 'properties': {}}

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        return Assessment(risk=Risk.READ, summary='ask about the displays')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        from roost.agent.stage import monitors
        from roost.system import hdr_route

        lines = [f'Desktop: {self.host.desktop or "unknown"} on {self.host.session or "unknown"}']

        found = monitors()
        if found:
            lines.append('')
            lines.append('Displays:')
            for m in found:
                primary = ' (primary)' if m.primary else ''
                lines.append(f'  {m.name}{primary} — {m.rect.width}x{m.rect.height} '
                             f'at {m.rect.x},{m.rect.y}')
        else:
            lines.append('The displays could not be enumerated from here.')

        route = hdr_route(self.host)
        lines.append('')
        if route.command:
            lines.append('HDR: there is a command for it here — use display_hdr.')
        elif route.supported:
            lines.append(
                'HDR: no command on this desktop. It is a switch in a window, and '
                'display_hdr will open the right one for you to click with the desktop '
                'tools — take a screenshot after it opens.'
            )
        else:
            lines.append(f'HDR: {route.where}')
        if route.where and route.supported:
            lines.append(f'  {route.where}')

        return Output(
            content='\n'.join(lines),
            display={'desktop': self.host.desktop,
                     'displays': [{'name': m.name, 'width': m.rect.width,
                                   'height': m.rect.height, 'primary': m.primary}
                                  for m in found],
                     'hdr_command': bool(route.command)},
        )


class DisplayHdrTool(_SystemTool):
    name = 'display_hdr'
    description = (
        'Turn HDR on or off for a display. Where this desktop has a command for it, this '
        'runs it. Where it does not — which is most of them — this opens the settings '
        'window at the right page and tells you what to click; finish it with '
        'desktop_screenshot and desktop_click. Run display_info first to get the display '
        'name and see which of the two will happen.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'output': {'type': 'string', 'description': 'Display name, from display_info.'},
            'enable': {'type': 'boolean', 'description': 'True to turn HDR on. Default true.'},
        },
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        from roost.system import hdr_route

        route = hdr_route(self.host)
        on = args.get('enable', True)
        output = (args.get('output') or '').strip()

        if not route.supported:
            return Assessment(risk=Risk.READ, summary='', invalid=route.where)

        if route.command:
            if not output:
                return Assessment(
                    risk=Risk.WRITE, summary='',
                    invalid='which display? Run display_info for the names.',
                )
            # It changes how the machine drives a monitor, and a wrong mode can
            # leave a screen blank until someone power-cycles it — but it is
            # not destructive and it is trivially reversible.
            return Assessment(
                risk=Risk.WRITE,
                summary=f'turn HDR {"on" if on else "off"} for {output}',
            )

        return Assessment(
            risk=Risk.EXECUTE,
            summary=f'open the display settings so HDR can be switched {"on" if on else "off"}',
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        from roost.system import hdr_route

        route = hdr_route(self.host)
        on = args.get('enable', True)
        output = (args.get('output') or '').strip()

        if route.command:
            argv = [
                part.format(output=output, state='enable' if on else 'disable')
                for part in route.command
            ]
            await ctx.emit(f'$ {" ".join(argv)}\n', 'stdout')
            code, out = await _run(argv, 60, emit=ctx.emit)
            if code != 0:
                raise ToolError(
                    f'that did not work (exit {code}): {out.strip()[:400]}\n'
                    'The display may not support HDR, or the name may be wrong — '
                    'display_info lists them.'
                )
            return Output(content=f'HDR is now {"on" if on else "off"} for {output}.')

        if not route.panel:
            raise ToolError(route.where)

        try:
            await asyncio.create_subprocess_exec(
                *route.panel,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=_env(),
            )
        except (OSError, FileNotFoundError) as exc:
            raise ToolError(
                f'could not open the settings window ({exc}). {route.where}'
            ) from exc

        return Output(
            content=(
                f'Opened the display settings. This desktop has no command for HDR, so the '
                f'rest is on screen: {route.where}\n\n'
                'Take a screenshot to see it, then click the switch. It may take a second '
                'to appear.'
            ),
            display={'panel': list(route.panel), 'where': route.where},
        )


def system_tools() -> list[Tool]:
    host = detect()
    return [
        SystemInfoTool(host),
        PackageSearchTool(host),
        PackageInstallTool(host),
        PackageRemoveTool(host),
        DisplayInfoTool(host),
        DisplayHdrTool(host),
    ]
