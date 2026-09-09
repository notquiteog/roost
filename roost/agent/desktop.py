"""Seeing and controlling the desktop itself.

None of this depends on the client being a web page. The daemon is a native
process with the machine's own permissions; the browser is only where the
result is displayed. Screenshotting and clicking are daemon-side, and a
native shell would not make them any more possible than they already are.

What *does* decide the approach is the display server, and on Linux that
split is not a detail:

**X11 tooling silently returns black on Wayland.** `mss`, `pyautogui`, `scrot`
and every other X11 grabber capture XWayland surfaces only, and a native
Wayland desktop has none — so they return a valid, correctly-sized, entirely
black image. Measured on the machine this was written on: 2560x1440, one
unique colour, mean pixel value 0.0. That is far worse than an error, because
a model handed a black screenshot will describe it, reason about it, and act
on the reasoning. So Wayland is detected first and X11 capture is never
attempted there.

**Input is the opposite way round.** `/dev/uinput` works on both, needs no
root where udev grants the user access, and needs no compiled dependency —
the ioctls are three integers and a struct.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# --- uinput ------------------------------------------------------------------
# Constants from linux/uinput.h and linux/input-event-codes.h, inlined rather
# than imported: python-evdev needs kernel headers and a C compiler at install
# time, which is a poor trade for an application meant to be pip-installable.
UI_DEV_CREATE, UI_DEV_DESTROY = 0x5501, 0x5502
UI_SET_EVBIT, UI_SET_KEYBIT, UI_SET_RELBIT, UI_SET_ABSBIT = (
    0x40045564, 0x40045565, 0x40045566, 0x40045567,
)
EV_SYN, EV_KEY, EV_REL, EV_ABS = 0x00, 0x01, 0x02, 0x03
REL_X, REL_Y, REL_WHEEL = 0x00, 0x01, 0x08
ABS_X, ABS_Y = 0x00, 0x01
SYN_REPORT = 0
BTN_LEFT, BTN_RIGHT, BTN_MIDDLE = 0x110, 0x111, 0x112

# US layout scancodes for the characters an agent actually types.
KEYS = {
    'a': 30, 'b': 48, 'c': 46, 'd': 32, 'e': 18, 'f': 33, 'g': 34, 'h': 35, 'i': 23,
    'j': 36, 'k': 37, 'l': 38, 'm': 50, 'n': 49, 'o': 24, 'p': 25, 'q': 16, 'r': 19,
    's': 31, 't': 20, 'u': 22, 'v': 47, 'w': 17, 'x': 45, 'y': 21, 'z': 44,
    '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10, '0': 11,
    '-': 12, '=': 13, '[': 26, ']': 27, '\\': 43, ';': 39, "'": 40, '`': 41,
    ',': 51, '.': 52, '/': 53, ' ': 57,
    'enter': 28, 'esc': 1, 'backspace': 14, 'tab': 15, 'space': 57, 'delete': 111,
    'up': 103, 'down': 108, 'left': 105, 'right': 106,
    'home': 102, 'end': 107, 'pageup': 104, 'pagedown': 109,
    'f1': 59, 'f2': 60, 'f3': 61, 'f4': 62, 'f5': 63, 'f6': 64,
    'f7': 65, 'f8': 66, 'f9': 67, 'f10': 68, 'f11': 87, 'f12': 88,
}
MODIFIERS = {'ctrl': 29, 'shift': 42, 'alt': 56, 'super': 125, 'meta': 125, 'cmd': 125}
SHIFTED = {
    '!': '1', '@': '2', '#': '3', '$': '4', '%': '5', '^': '6', '&': '7', '*': '8',
    '(': '9', ')': '0', '_': '-', '+': '=', '{': '[', '}': ']', '|': '\\',
    ':': ';', '"': "'", '~': '`', '<': ',', '>': '.', '?': '/',
}


class DesktopUnavailable(RuntimeError):
    """No usable capture or input path on this system."""


@dataclass(slots=True)
class Screen:
    width: int
    height: int


def is_wayland() -> bool:
    return bool(os.environ.get('WAYLAND_DISPLAY')) or os.environ.get('XDG_SESSION_TYPE') == 'wayland'


# --- capture -----------------------------------------------------------------


def _capture_wayland(output: str = '') -> bytes:
    """Capture through the compositor's own portal-backed tool.

    Every desktop exposes a different one and none of them share a flag, so
    the sequence is: whichever is installed, asked to write a file
    non-interactively. `grim` covers wlroots, `cosmic-screenshot` COSMIC, and
    the `Screenshot` portal is the fallback that works anywhere but usually
    shows a dialog.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)

        if shutil.which('grim'):
            target = out / 'shot.png'
            # `-o` is the one per-output flag any of these tools offer, which
            # is why multi-monitor targeting elsewhere crops the image instead.
            argv = ['grim', *(['-o', output] if output else []), str(target)]
            subprocess.run(argv, check=True, capture_output=True, timeout=30)
            return target.read_bytes()

        if shutil.which('cosmic-screenshot'):
            subprocess.run(
                ['cosmic-screenshot', '--interactive=false', '--modal=false',
                 '--notify=false', '--save-dir', str(out)],
                check=True, capture_output=True, timeout=30,
            )
            shots = sorted(out.glob('*'), key=lambda p: p.stat().st_mtime)
            if shots:
                return shots[-1].read_bytes()

        if shutil.which('gnome-screenshot'):
            target = out / 'shot.png'
            subprocess.run(['gnome-screenshot', '-f', str(target)], check=True,
                           capture_output=True, timeout=30)
            return target.read_bytes()

        if shutil.which('spectacle'):
            target = out / 'shot.png'
            subprocess.run(['spectacle', '-b', '-n', '-o', str(target)], check=True,
                           capture_output=True, timeout=30)
            return target.read_bytes()

    raise DesktopUnavailable(
        'No Wayland screenshot tool found. Install one of: grim (wlroots), '
        'cosmic-screenshot (COSMIC), gnome-screenshot, spectacle (KDE). '
        'X11 grabbers do not work here — they return a black image.'
    )


def _capture_mss() -> bytes:
    """X11, Windows and macOS, via mss. Never called on Wayland."""
    try:
        import io

        import mss
        from PIL import Image
    except ImportError as exc:
        raise DesktopUnavailable(f'screen capture needs mss and pillow: {exc}') from exc

    with mss.mss() as sct:
        raw = sct.grab(sct.monitors[1])
    image = Image.frombytes('RGB', raw.size, raw.rgb)
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


def capture(output: str = '') -> bytes:
    """A PNG of the whole primary display, or of one named Wayland output."""
    if sys.platform.startswith('linux') and is_wayland():
        return _capture_wayland(output)
    if sys.platform == 'darwin' and shutil.which('screencapture'):
        # The system tool, because it is already permitted once the user has
        # granted Screen Recording — no second permission to explain.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'shot.png'
            subprocess.run(['screencapture', '-x', str(target)], check=True,
                           capture_output=True, timeout=30)
            return target.read_bytes()
    return _capture_mss()


def screen_size() -> Screen:
    if sys.platform.startswith('linux') and is_wayland():
        # No X server to ask, so it comes from the image itself.
        try:
            import io

            from PIL import Image

            with Image.open(io.BytesIO(capture())) as im:
                return Screen(*im.size)
        except Exception as exc:  # noqa: BLE001
            raise DesktopUnavailable(f'could not determine screen size: {exc}') from exc
    try:
        import mss

        with mss.mss() as sct:
            mon = sct.monitors[1]
        return Screen(mon['width'], mon['height'])
    except ImportError as exc:
        raise DesktopUnavailable('screen size needs mss') from exc


# --- input -------------------------------------------------------------------


class UinputPointer:
    """A virtual mouse and keyboard.

    Absolute positioning uses an ABS device rather than relative moves,
    because relative moves compound: a pointer that is already 3 px off ends
    up 6 px off after the next move, and on a desktop that is the difference
    between a button and the thing beside it.
    """

    def __init__(self, screen: Screen) -> None:
        self.screen = screen
        try:
            self.fd = os.open('/dev/uinput', os.O_WRONLY | os.O_NONBLOCK)
        except PermissionError as exc:
            raise DesktopUnavailable(
                '/dev/uinput is not writable. Add yourself to the "input" group, or install a '
                'udev rule granting access, then log out and back in.'
            ) from exc
        except FileNotFoundError as exc:
            raise DesktopUnavailable('/dev/uinput does not exist (is the uinput module loaded?)') from exc

        try:
            for ev in (EV_KEY, EV_REL, EV_ABS, EV_SYN):
                fcntl.ioctl(self.fd, UI_SET_EVBIT, ev)
            for btn in (BTN_LEFT, BTN_RIGHT, BTN_MIDDLE):
                fcntl.ioctl(self.fd, UI_SET_KEYBIT, btn)
            for code in set(KEYS.values()) | set(MODIFIERS.values()):
                fcntl.ioctl(self.fd, UI_SET_KEYBIT, code)
            fcntl.ioctl(self.fd, UI_SET_RELBIT, REL_WHEEL)
            for axis in (ABS_X, ABS_Y):
                fcntl.ioctl(self.fd, UI_SET_ABSBIT, axis)

            # uinput_user_dev: name[80], input_id{4 x u16}, ff_effects_max u32,
            # then absmax/absmin/absfuzz/absflat, 64 s32 each.
            absmax = [0] * 64
            absmax[ABS_X] = screen.width
            absmax[ABS_Y] = screen.height
            dev = struct.pack('80sHHHHi', b'roost-virtual-input', 0x03, 0x1234, 0x5678, 1, 0)
            dev += struct.pack('64i', *absmax)          # absmax
            dev += struct.pack('64i', *([0] * 64))      # absmin
            dev += struct.pack('64i', *([0] * 64))      # absfuzz
            dev += struct.pack('64i', *([0] * 64))      # absflat
            os.write(self.fd, dev)
            fcntl.ioctl(self.fd, UI_DEV_CREATE)
            # The compositor needs a moment to notice a new input device;
            # events sent before it does are dropped silently.
            time.sleep(0.3)
        except OSError:
            os.close(self.fd)
            raise

    def _emit(self, etype: int, code: int, value: int) -> None:
        os.write(self.fd, struct.pack('llHHi', 0, 0, etype, code, value))

    def _sync(self) -> None:
        self._emit(EV_SYN, SYN_REPORT, 0)

    def move(self, x: int, y: int) -> None:
        self._emit(EV_ABS, ABS_X, max(0, min(x, self.screen.width)))
        self._emit(EV_ABS, ABS_Y, max(0, min(y, self.screen.height)))
        self._sync()

    def click(self, button: str = 'left', double: bool = False) -> None:
        code = {'left': BTN_LEFT, 'right': BTN_RIGHT, 'middle': BTN_MIDDLE}[button]
        for _ in range(2 if double else 1):
            self._emit(EV_KEY, code, 1)
            self._sync()
            time.sleep(0.02)
            self._emit(EV_KEY, code, 0)
            self._sync()
            if double:
                time.sleep(0.06)

    def scroll(self, amount: int) -> None:
        self._emit(EV_REL, REL_WHEEL, amount)
        self._sync()

    def _tap(self, code: int, mods: list[int]) -> None:
        for mod in mods:
            self._emit(EV_KEY, mod, 1)
        self._emit(EV_KEY, code, 1)
        self._sync()
        self._emit(EV_KEY, code, 0)
        for mod in reversed(mods):
            self._emit(EV_KEY, mod, 0)
        self._sync()

    def type_text(self, text: str, delay: float = 0.012) -> None:
        for char in text:
            if char == '\n':
                self._tap(KEYS['enter'], [])
            elif char in SHIFTED:
                self._tap(KEYS[SHIFTED[char]], [MODIFIERS['shift']])
            elif char.isupper():
                self._tap(KEYS[char.lower()], [MODIFIERS['shift']])
            elif char in KEYS:
                self._tap(KEYS[char], [])
            else:
                # Anything outside the US layout — accents, emoji, CJK — cannot
                # be reached by scancode. Skipped rather than guessed, because
                # a wrong character typed into a form is worse than a missing one.
                continue
            time.sleep(delay)

    def key(self, combo: str) -> None:
        """A chord such as 'ctrl+shift+t'."""
        parts = [p.strip().lower() for p in combo.split('+') if p.strip()]
        if not parts:
            raise ValueError('empty key combination')
        mods = [MODIFIERS[p] for p in parts[:-1] if p in MODIFIERS]
        last = parts[-1]
        if last not in KEYS:
            raise ValueError(f'unknown key: {last!r}')
        self._tap(KEYS[last], mods)

    def close(self) -> None:
        try:
            fcntl.ioctl(self.fd, UI_DEV_DESTROY)
        except OSError:
            pass
        finally:
            os.close(self.fd)


def make_input(screen: Screen):
    """The input backend for this platform."""
    if sys.platform.startswith('linux'):
        return UinputPointer(screen)
    raise DesktopUnavailable(
        f'desktop input is not implemented for {sys.platform} yet. '
        'Windows needs SendInput and macOS needs CGEventPost with Accessibility permission; '
        'screen capture already works on both.'
    )
