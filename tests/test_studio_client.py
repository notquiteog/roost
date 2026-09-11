"""The studio panel and the server it talks to, checked against each other.

There is no DOM here and no browser. What these catch is drift, which is the
failure mode a client made of hand-written selectors actually has: a renamed
id, an endpoint that moved, a parameter kind the server can emit and the form
has no control for. Every one of those is silent — the page loads, and one
thing does nothing — which is exactly the kind of bug that survives a manual
click-through.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / 'openmirror' / 'static'
STUDIO_JS = (STATIC / 'studio.js').read_text()
INDEX = (STATIC / 'index.html').read_text()
STYLE = (STATIC / 'style.css').read_text()


def test_every_element_the_studio_reaches_for_is_on_the_page():
    """A renamed id is a control that silently does nothing: `$` returns null,
    the handler is never attached, and the page looks fine."""
    wanted = set(re.findall(r"\$\('#([a-zA-Z0-9_-]+)'\)", STUDIO_JS))
    present = set(re.findall(r'id="([a-zA-Z0-9_-]+)"', INDEX))
    missing = wanted - present
    assert not missing, f'studio.js reaches for ids that are not in index.html: {sorted(missing)}'


def test_every_endpoint_the_studio_calls_is_one_the_server_serves():
    """The other direction of the same drift: a path that moved leaves a panel
    that loads and then fails at the first click."""
    from openmirror.routers import media as media_router

    served = {route.path for route in media_router.router.routes}

    called = set()
    for raw in re.findall(r"['\"`](/api/media[^'\"`?]*)", STUDIO_JS):
        # `${...}` interpolations stand in for a path parameter.
        path = re.sub(r'\$\{[^}]+\}', '{x}', raw).rstrip('/')
        called.add(path or '/api/media')

    def matches(path: str) -> bool:
        for route in served:
            pattern = re.sub(r'\{[^}]+\}', '[^/]+', route.rstrip('/')) or '/api/media'
            if re.fullmatch(pattern, path):
                return True
        return False

    unknown = {p for p in called if not matches(p)}
    assert not unknown, f'studio.js calls paths the media router does not serve: {sorted(unknown)}'


def test_the_form_has_a_control_for_every_kind_a_provider_can_declare():
    """`Kind` is the vocabulary a provider describes its knobs in. A kind with
    no branch here falls through to a text box — which for a picture is a text
    box you can type an id into and never would."""
    import typing

    from openmirror.media.params import Kind

    declared = set(typing.get_args(Kind))
    # `control()` branches on these; `string` is the fall-through and is
    # deliberately not named.
    handled = set(re.findall(r"param\.kind === '(\w+)'", STUDIO_JS))
    unhandled = declared - handled - {'string'}
    assert not unhandled, f'studio.js has no control for: {sorted(unhandled)}'


def test_the_sections_the_form_lays_out_are_the_ones_providers_use():
    """A group the client does not know is rendered last under "other", which
    is correct and is not where `motion` belongs."""
    from openmirror.media.params import Param

    groups = set(re.findall(r"GROUPS = \[([^\]]+)\]", STUDIO_JS)[0].replace("'", '').split(', '))
    documented = set(re.findall(r"'(\w+)'", Param.__doc__ or '')) if Param.__doc__ else set()
    # Taken from where the vocabulary is actually written down: the comment on
    # `Param.group` in openmirror/media/params.py.
    source = (Path(__file__).resolve().parents[1] / 'openmirror' / 'media' / 'params.py').read_text()
    listed = set(re.findall(r"# For grouping into sections: (.+?)\.\s", source, re.S)[0].replace("'", '')
                 .replace('\n', ' ').replace('#', '').split(', '))
    listed = {g.strip() for g in listed if g.strip()}
    assert listed <= groups, f'params.py documents groups the form does not lay out: {sorted(listed - groups)}'
    del documented


@pytest.mark.parametrize('handle', [
    'studio-jobs', 'studio-gallery', 'studio-more', 'shot-dialog', 'studio-params',
])
def test_the_pieces_added_for_this_panel_are_actually_styled(handle):
    """An element with no rule is an element that inherits whatever the last
    person's flexbox was doing."""
    assert f'id="{handle}"' in INDEX
    # Either the id itself or the class it carries has to appear in the sheet.
    classes = re.findall(rf'id="{handle}"[^>]*class="([^"]+)"', INDEX)
    names = {handle, *(c for group in classes for c in group.split())}
    assert any(f'.{name}' in STYLE or f'#{name}' in STYLE for name in names), \
        f'{handle} has no styling of its own'


def test_the_gallery_reserves_space_before_a_picture_arrives():
    """A grid that reflows as images load is a grid where the thing you were
    about to click moves out from under the pointer."""
    card = STYLE[STYLE.index('.shot-card img'):]
    assert 'aspect-ratio' in card[:600]


def test_a_stop_button_exists_for_work_that_is_in_flight():
    """The panel this replaced could watch one job and could not stop it. A
    four-minute render nobody can cancel is a GPU working for nobody."""
    assert "method: 'DELETE'" in STUDIO_JS
    assert '/api/media/jobs/' in STUDIO_JS


def test_progress_is_only_drawn_when_something_reported_a_number():
    """`progress: null` is a real answer from most of these backends, and a bar
    that creeps to ninety and stops is a lie that costs the true ones their
    credibility."""
    assert 'indeterminate' in STUDIO_JS
    assert 'indeterminate' in STYLE
    assert 'job.progress === null' in STUDIO_JS
