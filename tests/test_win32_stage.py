"""The Windows stage layer, tested from wherever the suite happens to run.

Most of this file cannot be exercised off Windows -- SendInput and
CreateDesktop have no stand-in worth writing. What *can* be checked anywhere
is that the module is importable on every platform, that it refuses clearly
rather than raising AttributeError on a missing DLL, and that the vocabulary
it accepts is the same one the X stage accepts. That last one is the point:
an agent should not have to know which operating system it is on to press
ctrl+c, and a key table that drifts between platforms is exactly the kind of
bug that only shows up in front of a user.
"""

from __future__ import annotations

import sys

import pytest

from openmirror.agent import stage as stage_mod
from openmirror.agent import win32


def test_imports_everywhere():
    """The module loads on Linux and macOS without touching a Windows DLL."""
    assert win32.available() is (sys.platform == 'win32')


@pytest.mark.skipif(sys.platform == 'win32', reason='the refusal path is for other platforms')
def test_refuses_clearly_off_windows():
    with pytest.raises(win32.Win32Unavailable) as caught:
        win32.screen_size()
    assert sys.platform in str(caught.value)

    with pytest.raises(win32.Win32Unavailable):
        win32.Desktop()


def test_named_keys_match_the_x_stage():
    """Every key the X stage names is a key Windows can press, and vice versa.

    Not a comparison of the codes -- a keysym and a virtual key code have no
    reason to agree -- but of the *names*, which are what an agent writes.
    """
    assert set(win32.VK) == set(stage_mod.NAMED_KEYS)
    assert set(win32.VK_MODIFIERS) == set(stage_mod.MODIFIER_KEYSYMS)


def test_buttons_match_the_x_stage():
    assert set(win32._BUTTONS) == set(stage_mod.BUTTONS)


def test_navigation_keys_are_marked_extended():
    """Arrows and the navigation cluster need the extended flag.

    Without it they are delivered as their numeric-keypad twins, so Home
    types 7 into an application reading scan codes.
    """
    for name in ('up', 'down', 'left', 'right', 'home', 'end', 'pageup', 'pagedown', 'insert', 'delete'):
        assert win32.VK[name] in win32._EXTENDED, f'{name} is not marked extended'

    # Ordinary character keys must not be.
    assert win32.VK['enter'] not in win32._EXTENDED
    assert win32.VK['space'] not in win32._EXTENDED


def test_surrogate_pairs_are_sent_as_two_units():
    """An emoji is two UTF-16 code units, and sending one inserts a box."""
    assert len(win32._utf16_units('\U0001F600')) == 2
    assert len(win32._utf16_units('a')) == 1
    assert len(win32._utf16_units('é')) == 1


def test_looks_black_finds_a_lit_pixel():
    stride = 997 * 4
    assert win32.looks_black(bytes(stride * 4))

    frame = bytearray(stride * 4)
    frame[stride + 1] = 0x40
    assert not win32.looks_black(bytes(frame))


def test_looks_black_biases_towards_the_slow_path():
    """A missed pixel must read as black, never the other way round.

    A false 'black' costs one unnecessary composite and still returns a real
    picture. A false 'not black' returns the black frame to the model.
    """
    frame = bytearray(997 * 4 * 4)
    frame[5] = 0xFF          # between samples, so it is not seen
    assert win32.looks_black(bytes(frame))
