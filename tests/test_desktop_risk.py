"""Desktop control: the compensations for having no DOM to read.

In the browser a click is graded by reading the element. Here there is a
bitmap and a coordinate, so three weaker checks stand in: the model must say
what it is clicking, the screenshot must be recent, and nothing is ever a
plain read.
"""

from __future__ import annotations

import time

import pytest

from openmirror.agent.stage import Rect
from openmirror.agent.tools.desktop import STALE_AFTER, desktop_tools
from openmirror.protocol.agent import Risk


class FakeStage:
    """A stage that reports a size and swallows input.

    Real input is not the thing under test here — the grading is — and a test
    that moved a real pointer would be a test nobody could run while working.
    """

    kind = 'virtual'
    shares_pointer = False

    def __init__(self, width=1280, height=800):
        self._rect = Rect(0, 0, width, height)
        self.actions = []

    def rect(self):
        return self._rect

    def capture(self):
        return b'\x89PNG fake'

    def move(self, x, y):
        self.actions.append(('move', x, y))

    def click(self, button='left', double=False):
        self.actions.append(('click', button, double))

    def type_text(self, text):
        self.actions.append(('type', text))

    def key(self, combo):
        self.actions.append(('key', combo))

    def scroll(self, amount):
        self.actions.append(('scroll', amount))

    def describe(self):
        return {'kind': self.kind, 'shares_pointer': self.shares_pointer,
                'width': self._rect.width, 'height': self._rect.height, 'origin': [0, 0]}


def tools(stage=None):
    found = {t.name: t for t in desktop_tools(stage or FakeStage())}
    return found, found['desktop_click'].state


async def noop(*a):
    pass


async def noask(*a):
    return ''


def ctx(tmp_path):
    from openmirror.agent.tools.base import ToolContext

    return ToolContext(root=tmp_path, cwd=tmp_path, emit=noop, ask=noask, session_id='t')


def test_a_click_needs_a_screenshot_first(tmp_path):
    """Aiming at coordinates from nothing is aiming at nothing."""
    found, _ = tools()
    a = found['desktop_click'].assess({'x': 10, 'y': 10, 'label': 'OK'}, ctx(tmp_path))
    assert a.invalid and 'screenshot first' in a.invalid


def test_a_stale_screenshot_is_refused(tmp_path):
    """A notification sliding in is enough to move what is under a point."""
    found, state = tools()
    state['last_shot'] = time.monotonic() - (STALE_AFTER + 5)
    a = found['desktop_click'].assess({'x': 10, 'y': 10, 'label': 'OK'}, ctx(tmp_path))
    assert a.invalid and 'old' in a.invalid


def test_a_click_must_say_what_it_is_clicking(tmp_path):
    """The label is what the person is shown in the prompt. Without it they
    are approving a coordinate, which tells them nothing."""
    found, state = tools()
    state['last_shot'] = time.monotonic()
    a = found['desktop_click'].assess({'x': 10, 'y': 10, 'label': '  '}, ctx(tmp_path))
    assert a.invalid and 'label is required' in a.invalid


def test_the_declared_label_is_graded_like_a_browser_element(tmp_path):
    found, state = tools()
    state['last_shot'] = time.monotonic()

    buy = found['desktop_click'].assess(
        {'x': 5, 'y': 5, 'label': 'Place your order'}, ctx(tmp_path)
    )
    assert buy.risk is Risk.PURCHASE
    assert 'Place your order' in buy.summary and '(5, 5)' in buy.summary


def test_no_desktop_click_is_ever_a_read(tmp_path):
    """The browser can auto-run a click because it read the element first.
    Nothing here was verified, so the floor is execute."""
    found, state = tools()
    state['last_shot'] = time.monotonic()
    for label in ('Cancel', 'Close', 'Add to cart', 'Next', 'a blank area'):
        a = found['desktop_click'].assess({'x': 1, 'y': 1, 'label': label}, ctx(tmp_path))
        assert a.risk is not Risk.READ, label


def test_secret_fields_are_refused_on_the_desktop_too(tmp_path):
    found, _ = tools()
    for field in ('Password', 'Card number', 'CVV', 'One-time code'):
        a = found['desktop_type'].assess({'text': 'x', 'field': field}, ctx(tmp_path))
        assert a.risk is Risk.CREDENTIAL, field
        # The value must not appear in a prompt that guards it.
        assert 'x' not in a.summary.replace('text', '')


def test_typing_must_say_where(tmp_path):
    found, _ = tools()
    a = found['desktop_type'].assess({'text': 'hello', 'field': ''}, ctx(tmp_path))
    assert a.invalid and 'field is required' in a.invalid


def test_a_screenshot_is_a_read(tmp_path):
    found, _ = tools()
    assert found['desktop_screenshot'].assess({}, ctx(tmp_path)).risk is Risk.READ


@pytest.mark.parametrize('combo,valid', [('ctrl+s', True), ('enter', True), ('', False)])
def test_key_combos(tmp_path, combo, valid):
    found, _ = tools()
    a = found['desktop_key'].assess({'combo': combo}, ctx(tmp_path))
    assert (a.invalid is None) == valid
