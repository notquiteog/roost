"""Installing things, on whatever machine this turns out to be.

"Install Steam" is five different commands and, on a Steam Deck, none. The
detection is what makes that difference visible, so most of what is here feeds
it a synthetic machine and checks the route — the one thing that cannot be
tested by running it, since this box is only ever one of the platforms.

The rest are the two failure modes that are worse than an error: a package
manager that stops to ask a question in a terminal nobody is reading, and an
agent that meets a read-only root filesystem and starts trying to disable it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from roost.agent.tools.base import ToolContext, ToolError
from roost.agent.tools.system import PackageInstallTool, PackageRemoveTool, _elevate
from roost.protocol.agent import Risk
from roost.system import (
    LINUX_MANAGERS,
    WINDOWS_MANAGERS,
    Platform,
    _is_immutable,
    _os_release,
    detect,
    hdr_route,
    recipe_for,
    route_for,
)


def linux(**kw) -> Platform:
    base = {
        'system': 'linux', 'distro': 'ubuntu', 'like': ('debian',),
        'managers': tuple(m for m in LINUX_MANAGERS if m.name in ('apt', 'flatpak')),
        'elevation': 'sudo-nopasswd',
    }
    return Platform(**{**base, **kw})


# -- reading the machine ----------------------------------------------------


def test_os_release_is_parsed_not_guessed(tmp_path):
    path = tmp_path / 'os-release'
    path.write_text(
        '# a comment\n'
        'NAME="Pop!_OS"\n'
        "ID=pop\n"
        'ID_LIKE="ubuntu debian"\n'
        'VERSION_ID="24.04"\n'
        '\n'
    )
    parsed = _os_release(path)
    assert parsed['ID'] == 'pop'
    assert parsed['ID_LIKE'] == 'ubuntu debian'
    assert parsed['NAME'] == 'Pop!_OS'


def test_a_missing_os_release_is_not_an_error(tmp_path):
    """macOS and Windows have none, and neither does a scratch container."""
    assert _os_release(tmp_path / 'nope') == {}


def test_an_ostree_variant_is_immutable():
    assert _is_immutable({'VARIANT_ID': 'silverblue'}) is True
    assert _is_immutable({'VARIANT_ID': 'steamdeck'}) is True


def test_this_machine_detects_as_something_coherent():
    """The one end-to-end check available: whatever this is, it is consistent."""
    host = detect()
    assert host.system in ('linux', 'windows', 'darwin')
    assert host.elevation in (
        'root', 'sudo-nopasswd', 'pkexec', 'sudo-password', 'uac', 'none',
    )
    for manager in host.managers:
        assert manager.scope in ('system', 'user')
        assert manager.needs_elevation == (manager.scope == 'system')


# -- what a name means here -------------------------------------------------


def test_a_derivative_inherits_its_parents_route():
    """Pop!_OS is not in the table and does not need to be: it declares
    `ID_LIKE="ubuntu debian"`, which is the point of reading that at all
    rather than enumerating every derivative anyone ships."""
    pop = linux(distro='pop', like=('ubuntu', 'debian'))
    route = route_for(recipe_for('steam'), pop)
    assert route.manager == 'apt'
    assert route.package == 'steam-installer'


def test_epic_on_linux_is_heroic_and_says_so():
    """There is no Epic client for Linux. The useful answer is a different
    program, and someone who asked for Epic has to be told that."""
    recipe = recipe_for('install the epic games store')
    route = route_for(recipe, linux())
    assert route.manager == 'flatpak'
    assert 'heroic' in route.package.lower()
    assert 'Heroic, not Epic' in route.note
    assert 'Say so before doing it' in recipe.note


def test_the_same_name_means_a_different_thing_on_windows():
    windows = Platform(system='windows', managers=WINDOWS_MANAGERS[:1], elevation='uac')
    route = route_for(recipe_for('epic games store'), windows)
    assert route.manager == 'winget'
    assert route.package == 'EpicGames.EpicGamesLauncher'


def test_a_name_nobody_knows_is_not_invented():
    assert recipe_for('some-program-nobody-has-heard-of') is None


@pytest.mark.parametrize('phrasing', [
    'steam', 'Steam', 'install steam', 'the steam client please',
])
def test_a_name_is_matched_loosely(phrasing):
    assert recipe_for(phrasing) is not None


def test_the_longer_name_wins():
    """'epic games store' must not be matched by the 'epic' entry first."""
    assert 'heroic' in route_for(recipe_for('epic games store'), linux()).package


# -- grading ----------------------------------------------------------------


async def noop(*args):
    return None


async def noask(*args):
    return ''


def ctx() -> ToolContext:
    return ToolContext(root=Path('/tmp'), cwd=Path('/tmp'), emit=noop, ask=noask, session_id='t')


def test_a_system_install_grades_higher_than_a_user_one():
    """A system package runs its own install scripts as root, which is
    arbitrary code execution by another name. A per-user one unpacks into a
    home directory, which is the same shape as any other network fetch."""
    tool = PackageInstallTool(linux())

    system = tool.assess({'what': 'steam'}, ctx())
    assert system.risk is Risk.EXECUTE
    assert 'as root' in system.summary

    user = tool.assess({'what': 'discord'}, ctx())
    assert user.risk is Risk.NETWORK
    assert 'this user only' in user.summary


def test_removing_is_always_destructive():
    """It can take a package's dependents with it, and on a system manager
    that can mean a desktop environment."""
    tool = PackageRemoveTool(linux())
    assessed = tool.assess({'package': 'steam', 'manager': 'apt'}, ctx())
    assert assessed.risk is Risk.DESTRUCTIVE


def test_an_unknown_name_is_refused_before_anything_runs():
    assessed = PackageInstallTool(linux()).assess({'what': 'not-a-real-thing'}, ctx())
    assert assessed.invalid and 'package_search' in assessed.invalid


# -- the two failures worse than an error -----------------------------------


def test_an_immutable_system_is_routed_around_not_fought():
    """An agent that meets apt failing on SteamOS will reach for
    `steamos-readonly disable` unless something tells it not to."""
    deck = linux(distro='steamos', like=(), immutable=True, steam_deck=True)
    with pytest.raises(ToolError) as caught:
        PackageInstallTool(deck)._plan({'what': 'steam', 'manager': 'apt', 'package': 'steam'})

    message = str(caught.value)
    assert 'read-only' in message
    assert 'Do not try to disable that' in message
    assert 'flatpak' in message


def test_no_way_to_elevate_is_refused_rather_than_hung():
    """`sudo` on a machine wanting a password does not fail — it blocks on a
    prompt going nowhere, until the tool times out. Saying so lets the agent
    explain the problem instead of becoming it."""
    stuck = linux(elevation='sudo-password', elevation_note='sudo needs a password.')
    apt = stuck.manager('apt')

    with pytest.raises(ToolError) as caught:
        _elevate(stuck, apt)
    message = str(caught.value)
    assert 'without a person' in message
    assert 'flatpak' in message, 'it should name the per-user alternative that is present'

    # And a user-scope manager needs no prefix at all, so it still works.
    assert _elevate(stuck, stuck.manager('flatpak')) == []


@pytest.mark.parametrize(
    ('elevation', 'expected'),
    [('root', []), ('sudo-nopasswd', ['sudo', '-n']), ('pkexec', ['pkexec']), ('uac', [])],
)
def test_the_elevation_prefix_matches_the_route(elevation, expected):
    host = linux(elevation=elevation)
    assert _elevate(host, host.manager('apt')) == expected


def test_every_manager_installs_without_asking_a_question():
    """A manager that stops at "Do you want to continue? [Y/n]" hangs in a
    terminal nobody is reading. Every one of these has to be told not to."""
    interactive = {'apk', 'scoop', 'brew', 'snap', 'nix-env'}
    for manager in (*LINUX_MANAGERS, *WINDOWS_MANAGERS):
        if manager.name in interactive:
            continue        # these do not prompt
        assert manager.assume_yes, f'{manager.name} would stop and ask'


# -- display ----------------------------------------------------------------


def test_hdr_reports_a_route_or_admits_there_is_none():
    """Some desktops expose this as a command and some only as a switch in a
    window. An agent told there is a universal command will invent one."""
    route = hdr_route(detect())
    assert route.command or route.panel or not route.supported
    if route.supported:
        assert route.where, 'a route with no explanation is not a route'


def test_windows_hdr_is_a_settings_page():
    route = hdr_route(Platform(system='windows'))
    assert route.command == ()
    assert 'ms-settings:display' in ' '.join(route.panel)
    assert 'Use HDR' in route.where


def test_a_steam_deck_is_told_where_hdr_actually_is():
    route = hdr_route(Platform(system='linux', steam_deck=True))
    assert 'Game Mode' in route.where
