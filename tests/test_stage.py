"""The stage: what a rectangle means, and how monitors are picked.

The parts that need a real X server live in test_booking_walkthrough.py and
skip where there is none. What is here is the arithmetic and the refusals,
which have to hold on a machine with no display at all — including CI.
"""

from __future__ import annotations

import pytest

from roost.agent.stage import Monitor, Rect, StageUnavailable, _parse_size, pick_monitor


def test_a_point_is_measured_from_the_picture_the_model_was_shown():
    """The model is always shown a region starting at (0, 0), so it never has
    to know its monitor begins 2560 pixels along."""
    second = Rect(2560, 0, 1920, 1080)
    assert second.contains(0, 0)
    assert second.to_display(0, 0) == (2560, 0)
    assert second.to_display(100, 50) == (2660, 50)


def test_the_edge_of_the_region_is_outside_it():
    region = Rect(0, 0, 800, 600)
    assert region.contains(799, 599)
    assert not region.contains(800, 600)
    assert not region.contains(-1, 0)


def test_a_point_on_another_monitor_is_not_in_this_one():
    """Which is the whole of multi-monitor targeting: everything outside the
    rectangle is refused rather than clamped, because a clamped click lands on
    the edge of the allowed screen and does something nobody asked for."""
    assert not Rect(2560, 0, 1920, 1080).contains(3000, 100)


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('1920x1080', (1920, 1080)),
        ('1280X800', (1280, 800)),
        ('nonsense', (1920, 1080)),
        ('', (1920, 1080)),
        (None, (1920, 1080)),
        # Too small to show anything is raised to something usable rather than
        # accepted: a 1x1 display is not a stage, it is a bug that renders.
        ('10x10', (320, 240)),
    ],
)
def test_a_size_that_cannot_be_read_falls_back(text, expected):
    assert _parse_size(text) == expected


def test_naming_a_monitor_that_is_not_there_says_which_are(monkeypatch):
    """An agent confined to "monitor 3" on a two-monitor machine must not
    quietly get monitor 1."""
    monkeypatch.setattr(
        'roost.agent.stage.monitors',
        lambda display='': [
            Monitor('DP-3', Rect(0, 0, 2560, 1440), primary=True),
            Monitor('HDMI-1', Rect(2560, 0, 1920, 1080)),
        ],
    )

    assert pick_monitor('2').name == 'HDMI-1'
    assert pick_monitor('DP-3').rect.width == 2560
    assert pick_monitor('') is None

    with pytest.raises(StageUnavailable, match='no monitor 3'):
        pick_monitor('3')
    with pytest.raises(StageUnavailable, match='DP-3, HDMI-1'):
        pick_monitor('VGA-9')


def test_being_unable_to_enumerate_is_an_error_not_a_free_pass(monkeypatch):
    """Silently using the whole desktop when someone asked for one monitor is
    the failure this refuses to make."""
    monkeypatch.setattr('roost.agent.stage.monitors', lambda display='': [])
    with pytest.raises(StageUnavailable, match='could not be enumerated'):
        pick_monitor('1')
