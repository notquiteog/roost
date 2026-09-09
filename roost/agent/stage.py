"""Where the agent's hands are.

The requirement this exists for is blunt: an agent booking a hotel must not
take the mouse away from the person watching it. An agent that owns the
pointer owns the computer, and a computer you cannot use while it works is a
computer you have lent out rather than one you are being helped with.

There is exactly one honest way to get that, and it is not politeness about
where the cursor is put. A pointer is a property of a *display*, so the only
way for the agent to have its own is to give it its own display:

**virtual** — an X server of its own, off-screen, with its own pointer,
keyboard and root window. Applications are launched onto it by environment.
Nothing the agent does can reach your screen, your focus or your clipboard,
and you watch through the frames it captures rather than over its shoulder.
This is the only stage under which "it will not take your mouse" is a
guarantee rather than an intention.

**shared** — the screen you are looking at. There is one pointer and the agent
moves it, so the guarantee is not available; what is available is that it
gives it straight back. It notes where the pointer was, moves, acts, puts it
back, and — if `yield_to_user` is on — refuses to act at all when it finds the
pointer somewhere it did not leave it, because that means a human has a hand
on the mouse. That is cooperation, not isolation, and this file says so rather
than implying more.

A stage also carries a **rectangle**, which is what multi-monitor targeting
is. Everything outside it is neither captured nor clickable: a click is
refused before it happens rather than clamped into the edge of the allowed
screen, because a clamped click lands on something real and does something
nobody asked for. On a shared stage that rectangle is how you keep the agent
on the second monitor while you work on the first.
"""

from __future__ import annotations

import abc
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


class StageUnavailable(RuntimeError):
    """No usable display for this kind of stage on this system."""


class UserHasTheMouse(RuntimeError):
    """A person is using the pointer this stage shares with them.

    Raised rather than waited on. The agent is told plainly, so it stops and
    says so, instead of fighting for a cursor someone else has hold of.
    """


@dataclass(frozen=True, slots=True)
class Rect:
    """The region a stage may see and touch, in that display's coordinates."""

    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    def contains(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def to_display(self, x: int, y: int) -> tuple[int, int]:
        """A point the model gave, in the display's own coordinates.

        The model is always shown a picture of the region starting at (0, 0),
        so it never has to know that its monitor begins 2560 pixels along.
        """
        return self.x + x, self.y + y


@dataclass(frozen=True, slots=True)
class Monitor:
    name: str
    rect: Rect
    primary: bool = False


class Stage(abc.ABC):
    """One place the agent can look at and act on."""

    #: 'virtual' or 'shared' — what a person is being promised.
    kind: str = 'shared'
    #: True when the pointer being moved is the one on the user's desk.
    shares_pointer: bool = True

    @abc.abstractmethod
    def rect(self) -> Rect: ...

    @abc.abstractmethod
    def capture(self) -> bytes:
        """A PNG of this stage's rectangle, cropped, origin at its top left."""

    @abc.abstractmethod
    def move(self, x: int, y: int) -> None: ...

    @abc.abstractmethod
    def click(self, button: str = 'left', double: bool = False) -> None: ...

    @abc.abstractmethod
    def type_text(self, text: str) -> None: ...

    @abc.abstractmethod
    def key(self, combo: str) -> None: ...

    @abc.abstractmethod
    def scroll(self, amount: int) -> None: ...

    def env(self) -> dict[str, str]:
        """Environment for launching an application *onto* this stage.

        The whole point of a virtual stage: a browser started with this in its
        environment appears where the agent can see it and nowhere else.
        """
        return {}

    def describe(self) -> dict[str, object]:
        r = self.rect()
        return {
            'kind': self.kind,
            'shares_pointer': self.shares_pointer,
            'width': r.width,
            'height': r.height,
            'origin': [r.x, r.y],
        }

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# X11, through XTEST
# ---------------------------------------------------------------------------
#
# ctypes rather than python-xlib or xdotool: the first is a dependency that
# adds nothing over eight function signatures, and the second is a binary that
# is missing on most desktop installs — including the one this was written on.


class _X11:
    """The handful of Xlib and XTEST calls a stage needs."""

    def __init__(self, display: str) -> None:
        import ctypes
        import ctypes.util

        self.ctypes = ctypes
        x11_path = ctypes.util.find_library('X11')
        xtst_path = ctypes.util.find_library('Xtst')
        if not x11_path or not xtst_path:
            raise StageUnavailable(
                'libX11 and libXtst are needed to drive an X display. '
                'On Debian and Ubuntu: apt install libx11-6 libxtst6.'
            )

        self.x11 = ctypes.CDLL(x11_path)
        self.xtst = ctypes.CDLL(xtst_path)

        self.x11.XOpenDisplay.restype = ctypes.c_void_p
        self.x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self.dpy = self.x11.XOpenDisplay(display.encode())
        if not self.dpy:
            raise StageUnavailable(f'cannot open the X display {display!r}')
        self.display = display

        self.x11.XDefaultRootWindow.restype = ctypes.c_ulong
        self.x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        self.root = self.x11.XDefaultRootWindow(self.dpy)

        for name in ('XFlush', 'XSync', 'XCloseDisplay'):
            getattr(self.x11, name).argtypes = [ctypes.c_void_p]
        self.x11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]

        self.xtst.XTestFakeMotionEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong
        ]
        self.xtst.XTestFakeButtonEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong
        ]
        self.xtst.XTestFakeKeyEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong
        ]
        self.x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.x11.XKeysymToKeycode.restype = ctypes.c_ubyte

        self._keymap: dict[int, tuple[int, bool]] | None = None

    # -- pointer ------------------------------------------------------------

    def pointer(self) -> tuple[int, int]:
        ct = self.ctypes
        root_ret = ct.c_ulong()
        child = ct.c_ulong()
        root_x, root_y = ct.c_int(), ct.c_int()
        win_x, win_y = ct.c_int(), ct.c_int()
        mask = ct.c_uint()
        self.x11.XQueryPointer(
            ct.c_void_p(self.dpy), ct.c_ulong(self.root),
            ct.byref(root_ret), ct.byref(child),
            ct.byref(root_x), ct.byref(root_y),
            ct.byref(win_x), ct.byref(win_y), ct.byref(mask),
        )
        return root_x.value, root_y.value

    def move(self, x: int, y: int) -> None:
        self.xtst.XTestFakeMotionEvent(self.dpy, -1, int(x), int(y), 0)
        self.x11.XFlush(self.dpy)

    def button(self, button: int, press: bool) -> None:
        self.xtst.XTestFakeButtonEvent(self.dpy, button, 1 if press else 0, 0)
        self.x11.XFlush(self.dpy)

    def key_code(self, code: int, press: bool) -> None:
        self.xtst.XTestFakeKeyEvent(self.dpy, code, 1 if press else 0, 0)
        self.x11.XFlush(self.dpy)

    # -- keyboard -----------------------------------------------------------

    def keymap(self) -> dict[int, tuple[int, bool]]:
        """keysym → (keycode, needs shift), read from the server's own map.

        Read rather than assumed, because a hard-coded US scancode table types
        gibberish on any other layout — and the person whose layout it is has
        no idea why the agent is filling their form with nonsense.
        """
        if self._keymap is not None:
            return self._keymap

        ct = self.ctypes
        low, high = ct.c_int(), ct.c_int()
        self.x11.XDisplayKeycodes(ct.c_void_p(self.dpy), ct.byref(low), ct.byref(high))
        count = high.value - low.value + 1
        per = ct.c_int()
        self.x11.XGetKeyboardMapping.restype = ct.POINTER(ct.c_ulong)
        syms = self.x11.XGetKeyboardMapping(
            ct.c_void_p(self.dpy), ct.c_ubyte(low.value), ct.c_int(count), ct.byref(per)
        )

        out: dict[int, tuple[int, bool]] = {}
        width = per.value
        for i in range(count):
            keycode = low.value + i
            for level in (0, 1):        # unshifted, then shifted
                if level >= width:
                    continue
                keysym = syms[i * width + level]
                if keysym and keysym not in out:
                    out[keysym] = (keycode, level == 1)
        self.x11.XFree(syms)
        self._keymap = out
        return out

    def keysym_for(self, char: str) -> int:
        """The X keysym for a character.

        Latin-1 keysyms are their own code points, and everything above that
        is the code point with the Unicode flag set. Both rules are from the X
        protocol rather than a table, which is why this handles an accented
        character the hard-coded scancode approach could not.
        """
        code = ord(char)
        if 0x20 <= code <= 0xFF:
            return code
        return 0x01000000 | code

    def close(self) -> None:
        try:
            self.x11.XCloseDisplay(self.dpy)
        except Exception:  # noqa: BLE001
            pass


# The keysyms for keys that have names rather than characters.
NAMED_KEYS = {
    'enter': 0xFF0D, 'return': 0xFF0D, 'esc': 0xFF1B, 'escape': 0xFF1B,
    'backspace': 0xFF08, 'tab': 0xFF09, 'space': 0x20, 'delete': 0xFFFF,
    'up': 0xFF52, 'down': 0xFF54, 'left': 0xFF51, 'right': 0xFF53,
    'home': 0xFF50, 'end': 0xFF57, 'pageup': 0xFF55, 'pagedown': 0xFF56,
    'insert': 0xFF63, 'menu': 0xFF67, 'print': 0xFF61,
    **{f'f{n}': 0xFFBE + n - 1 for n in range(1, 13)},
}

MODIFIER_KEYSYMS = {
    'ctrl': 0xFFE3, 'control': 0xFFE3, 'shift': 0xFFE1, 'alt': 0xFFE9,
    'super': 0xFFEB, 'meta': 0xFFEB, 'cmd': 0xFFEB,
}

BUTTONS = {'left': 1, 'middle': 2, 'right': 3}


class XStage(Stage):
    """A stage on an X display — virtual or shared, the mechanics are the same.

    What differs between them is only what is being promised, which is why
    `shares_pointer` is a constructor argument rather than a subclass: the
    code that moves a pointer should not be duplicated for the sake of a
    label, and the label is the thing a person is relying on.
    """

    def __init__(
        self,
        display: str,
        *,
        region: Rect | None = None,
        kind: str = 'shared',
        shares_pointer: bool = True,
        yield_to_user: bool = True,
    ) -> None:
        self.display = display
        self.kind = kind
        self.shares_pointer = shares_pointer
        self.yield_to_user = yield_to_user and shares_pointer
        self.x = _X11(display)
        self._region = region or self._whole_screen()
        # Where we left the pointer. Finding it somewhere else means a person
        # moved it, which is the only reliable signal available that someone
        # has a hand on the mouse.
        self._left_at: tuple[int, int] | None = None

    def _whole_screen(self) -> Rect:
        ct = self.x.ctypes
        self.x.x11.XDisplayWidth.argtypes = [ct.c_void_p, ct.c_int]
        self.x.x11.XDisplayHeight.argtypes = [ct.c_void_p, ct.c_int]
        width = self.x.x11.XDisplayWidth(ct.c_void_p(self.x.dpy), 0)
        height = self.x.x11.XDisplayHeight(ct.c_void_p(self.x.dpy), 0)
        return Rect(0, 0, width, height)

    def rect(self) -> Rect:
        return self._region

    def env(self) -> dict[str, str]:
        return {'DISPLAY': self.display}

    # -- looking ------------------------------------------------------------

    def capture(self) -> bytes:
        try:
            import io

            import mss
            from PIL import Image
        except ImportError as exc:
            raise StageUnavailable(f'screen capture needs mss and pillow: {exc}') from exc

        r = self._region
        with mss.MSS(display=self.display) as sct:
            raw = sct.grab({'left': r.x, 'top': r.y, 'width': r.width, 'height': r.height})
        image = Image.frombytes('RGB', raw.size, raw.rgb)
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        return buf.getvalue()

    # -- acting -------------------------------------------------------------

    def _check_the_user(self) -> None:
        """Refuse to act if someone else has hold of the pointer."""
        if not self.yield_to_user or self._left_at is None:
            return
        where = self.x.pointer()
        # A couple of pixels of slack: a compositor with pointer constraints or
        # a hidpi scale can report a position one off from where it was put,
        # and stopping the agent over that would make the feature unusable.
        if abs(where[0] - self._left_at[0]) > 3 or abs(where[1] - self._left_at[1]) > 3:
            self._left_at = None
            raise UserHasTheMouse(
                'the pointer has moved since the last action, so someone is using this '
                'screen. Stopped rather than fighting for the mouse. Ask them whether to '
                'carry on, or move to a stage of your own.'
            )

    def move(self, x: int, y: int) -> None:
        if not self._region.contains(x, y):
            raise ValueError(
                f'({x}, {y}) is outside this stage — it is {self._region.width}x'
                f'{self._region.height}. Coordinates are relative to the picture you were shown.'
            )
        self._check_the_user()
        dx, dy = self._region.to_display(x, y)
        self.x.move(dx, dy)
        self._left_at = (dx, dy)

    def click(self, button: str = 'left', double: bool = False) -> None:
        code = BUTTONS.get(button, 1)
        self._check_the_user()
        for i in range(2 if double else 1):
            if i:
                time.sleep(0.06)
            self.x.button(code, True)
            time.sleep(0.02)
            self.x.button(code, False)

    def scroll(self, amount: int) -> None:
        # X has no scroll axis: wheel up and down are buttons 4 and 5.
        self._check_the_user()
        code = 4 if amount > 0 else 5
        for _ in range(min(abs(int(amount)), 30)):
            self.x.button(code, True)
            self.x.button(code, False)
            time.sleep(0.01)

    def _tap(self, keysym: int, modifiers: list[int], shifted: bool = False) -> None:
        keymap = self.x.keymap()
        entry = keymap.get(keysym)
        if entry is None:
            code = self.x.x11.XKeysymToKeycode(self.x.ctypes.c_void_p(self.x.dpy), keysym)
            if not code:
                raise ValueError(f'this keyboard layout cannot produce keysym {keysym:#x}')
            entry = (code, False)
        code, needs_shift = entry

        held = list(modifiers)
        if needs_shift or shifted:
            shift = keymap.get(MODIFIER_KEYSYMS['shift'])
            if shift:
                held.append(shift[0])

        for mod in held:
            self.x.key_code(mod, True)
        self.x.key_code(code, True)
        self.x.key_code(code, False)
        for mod in reversed(held):
            self.x.key_code(mod, False)

    def type_text(self, text: str) -> None:
        self._check_the_user()
        keymap = self.x.keymap()
        for char in text:
            if char == '\n':
                self._tap(NAMED_KEYS['enter'], [])
            elif char == '\t':
                self._tap(NAMED_KEYS['tab'], [])
            else:
                keysym = self.x.keysym_for(char)
                if keysym not in keymap:
                    # A character this layout cannot reach. Skipped rather
                    # than approximated: a wrong character typed into a form
                    # is worse than a missing one, and silently substituting
                    # 'e' for 'é' produces a booking in someone else's name.
                    log.info('skipping a character this layout cannot type: %r', char)
                    continue
                self._tap(keysym, [])
            time.sleep(0.012)

    def key(self, combo: str) -> None:
        self._check_the_user()
        parts = [p.strip().lower() for p in combo.split('+') if p.strip()]
        if not parts:
            raise ValueError('empty key combination')

        keymap = self.x.keymap()
        modifiers: list[int] = []
        for part in parts[:-1]:
            keysym = MODIFIER_KEYSYMS.get(part)
            if keysym is None:
                raise ValueError(f'unknown modifier: {part!r}')
            entry = keymap.get(keysym)
            if entry:
                modifiers.append(entry[0])

        last = parts[-1]
        keysym = NAMED_KEYS.get(last)
        if keysym is None:
            if len(last) != 1:
                raise ValueError(f'unknown key: {last!r}')
            keysym = self.x.keysym_for(last)
        self._tap(keysym, modifiers)

    def close(self) -> None:
        self.x.close()


# ---------------------------------------------------------------------------
# A display of its own
# ---------------------------------------------------------------------------


class VirtualStage(XStage):
    """An Xvfb server started for this session, and torn down with it.

    This is the stage that makes the promise keepable. The pointer it moves
    does not exist on your desk; the window it clicks is not on your screen;
    the keystrokes it sends go to an application that has focus on a display
    nobody is looking at. You can keep typing.
    """

    kind = 'virtual'
    shares_pointer = False

    def __init__(self, width: int = 1920, height: int = 1080, *, depth: int = 24) -> None:
        if not shutil.which('Xvfb'):
            raise StageUnavailable(
                'Xvfb is not installed, so the agent cannot be given a display of its own. '
                'Install it (apt install xvfb) — or set ROOST_DESKTOP_STAGE=shared to let it '
                'drive the screen you are looking at, which means sharing your mouse with it.'
            )

        self.display_number = _free_display()
        display = f':{self.display_number}'
        self._xauth = Path(tempfile.mkdtemp(prefix='roost-stage-')) / 'Xauthority'
        self._proc = subprocess.Popen(
            [
                'Xvfb', display,
                '-screen', '0', f'{width}x{height}x{depth}',
                # No TCP: this display is for this machine, and an X server on
                # a network port is an X server anyone on the network can
                # watch and type into.
                '-nolisten', 'tcp',
                '-noreset',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Wait for the socket rather than sleeping a fixed amount: on a loaded
        # machine a fixed sleep is either too short to work or too long to
        # tolerate, and the socket appearing is the actual event.
        socket = Path(f'/tmp/.X11-unix/X{self.display_number}')
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if socket.exists():
                break
            if self._proc.poll() is not None:
                err = (self._proc.stderr.read() or b'').decode('utf-8', 'replace')[:400]
                raise StageUnavailable(f'Xvfb exited immediately: {err}')
            time.sleep(0.05)
        else:
            self._proc.kill()
            raise StageUnavailable(f'Xvfb did not come up on {display} within 10s')

        super().__init__(display, region=Rect(0, 0, width, height), kind='virtual',
                         shares_pointer=False, yield_to_user=False)
        log.info('virtual stage on %s at %dx%d (your pointer is untouched)', display, width, height)

    def close(self) -> None:
        super().close()
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        shutil.rmtree(self._xauth.parent, ignore_errors=True)


def _free_display() -> int:
    """A display number nothing is using.

    Starts at 90 rather than 1 to stay clear of real sessions: `:0` and `:1`
    are somebody's desktop, and an Xvfb that lost a race for one of those
    would be a very confusing failure.
    """
    used = set()
    sockets = Path('/tmp/.X11-unix')
    if sockets.is_dir():
        for entry in sockets.iterdir():
            if entry.name.startswith('X') and entry.name[1:].isdigit():
                used.add(int(entry.name[1:]))
    for candidate in range(90, 200):
        if candidate not in used and not Path(f'/tmp/.X{candidate}-lock').exists():
            return candidate
    raise StageUnavailable('no free X display number between :90 and :200')


# ---------------------------------------------------------------------------
# Wayland
# ---------------------------------------------------------------------------


class WaylandStage(Stage):
    """The Wayland desktop you are looking at.

    Wayland has no XTEST and, by design, no way for a client to synthesise
    input to another client — which is a security property rather than an
    oversight. So capture goes through the compositor's own screenshot tool
    and input through `/dev/uinput`, a virtual device the kernel presents to
    the compositor as though it were a second mouse and keyboard plugged in.

    That second device shares your pointer, and there is no Wayland equivalent
    of "give the agent its own screen" short of running a nested compositor.
    Where that matters — which is most of the time — the answer is a virtual
    X stage, which works perfectly well on a Wayland desktop because the
    applications being driven are the ones that must run there, not the
    desktop itself.
    """

    kind = 'shared'
    shares_pointer = True

    def __init__(self, region: Rect | None = None, *, output: str = '') -> None:
        from roost.agent import desktop as desktop_mod

        self._desktop = desktop_mod
        self.output = output
        self._region = region
        self._pointer = None

    def rect(self) -> Rect:
        if self._region is None:
            size = self._desktop.screen_size()
            self._region = Rect(0, 0, size.width, size.height)
        return self._region

    def capture(self) -> bytes:
        png = self._desktop.capture(output=self.output)
        region = self._region
        if region is None or (region.x == 0 and region.y == 0):
            return png
        # Cropped here rather than by the screenshot tool, because only `grim`
        # takes a geometry and it is not the tool on most desktops.
        import io

        from PIL import Image

        with Image.open(io.BytesIO(png)) as im:
            cropped = im.crop((region.x, region.y, region.x + region.width, region.y + region.height))
            buf = io.BytesIO()
            cropped.save(buf, format='PNG')
            return buf.getvalue()

    def _input(self):
        if self._pointer is None:
            r = self.rect()
            self._pointer = self._desktop.make_input(self._desktop.Screen(r.width, r.height))
        return self._pointer

    def move(self, x: int, y: int) -> None:
        r = self.rect()
        if not r.contains(x, y):
            raise ValueError(f'({x}, {y}) is outside this stage — it is {r.width}x{r.height}')
        self._input().move(*r.to_display(x, y))

    def click(self, button: str = 'left', double: bool = False) -> None:
        self._input().click(button, double)

    def type_text(self, text: str) -> None:
        self._input().type_text(text)

    def key(self, combo: str) -> None:
        self._input().key(combo)

    def scroll(self, amount: int) -> None:
        self._input().scroll(amount)

    def close(self) -> None:
        if self._pointer is not None:
            self._pointer.close()
            self._pointer = None


# ---------------------------------------------------------------------------
# Monitors
# ---------------------------------------------------------------------------


_XRANDR_LINE = re.compile(r'^(\S+) connected (primary )?(\d+)x(\d+)\+(\d+)\+(\d+)', re.M)


def monitors(display: str = '') -> list[Monitor]:
    """Every monitor on a display, as X arranges them.

    xrandr first because it gives names — "HDMI-1" is a thing a person can
    point at and "monitor 2" is not — and mss second because it works where
    xrandr is missing. On Wayland both are usually unavailable and the answer
    is one screen, which is the honest answer: there is no portable way to
    enumerate Wayland outputs without talking to each compositor's own
    protocol.
    """
    env = {**os.environ, 'DISPLAY': display} if display else os.environ
    if shutil.which('xrandr'):
        try:
            out = subprocess.run(
                ['xrandr', '--query'], capture_output=True, text=True, timeout=10, env=env
            ).stdout
            found = [
                Monitor(
                    name=name,
                    rect=Rect(int(x), int(y), int(w), int(h)),
                    primary=bool(primary),
                )
                for name, primary, w, h, x, y in _XRANDR_LINE.findall(out)
            ]
            if found:
                return found
        except (OSError, subprocess.SubprocessError, ValueError):
            log.debug('xrandr did not answer', exc_info=True)

    try:
        import mss

        with mss.MSS(display=display) if display else mss.MSS() as sct:
            # monitors[0] is the union of all of them, which is not a monitor.
            return [
                Monitor(name=f'{i}', rect=Rect(m['left'], m['top'], m['width'], m['height']))
                for i, m in enumerate(sct.monitors[1:], start=1)
            ]
    except Exception:  # noqa: BLE001
        log.debug('mss could not enumerate monitors', exc_info=True)
    return []


def pick_monitor(wanted: str, display: str = '') -> Monitor | None:
    """The monitor someone named, by index or by name. None means all of them."""
    if not wanted:
        return None
    found = monitors(display)
    if not found:
        raise StageUnavailable(
            f'cannot confine the agent to monitor {wanted!r}: the monitors on this display '
            'could not be enumerated. Install xrandr, or clear ROOST_DESKTOP_MONITOR.'
        )
    if wanted.isdigit():
        index = int(wanted)
        if not 1 <= index <= len(found):
            names = ', '.join(f'{i}={m.name}' for i, m in enumerate(found, start=1))
            raise StageUnavailable(f'there is no monitor {index} — this display has: {names}')
        return found[index - 1]
    for monitor in found:
        if monitor.name == wanted:
            return monitor
    names = ', '.join(m.name for m in found)
    raise StageUnavailable(f'no monitor called {wanted!r}. This display has: {names}')


# ---------------------------------------------------------------------------
# Choosing one
# ---------------------------------------------------------------------------


def build(cfg) -> Stage:
    """The stage this configuration asks for.

    `auto` prefers a display of its own, because that is the setting under
    which the agent cannot interfere with the person watching it — and if
    Xvfb is not installed it falls back to the shared screen and says so in
    the log, since an agent that cannot see anything is not a safer agent, it
    is a useless one.
    """
    requested = (cfg.desktop_stage or 'auto').strip().lower()
    width, height = _parse_size(cfg.desktop_virtual_size)

    if requested == 'virtual':
        return VirtualStage(width, height)

    if requested == 'auto':
        try:
            return VirtualStage(width, height)
        except StageUnavailable as exc:
            log.warning('%s — falling back to the screen you are looking at', exc)

    return shared_stage(cfg)


def shared_stage(cfg) -> Stage:
    """The screen in front of the person, confined to one monitor if asked."""
    from roost.agent import desktop as desktop_mod

    display = cfg.desktop_display or os.environ.get('DISPLAY', '')

    if sys.platform.startswith('linux') and desktop_mod.is_wayland() and not cfg.desktop_display:
        monitor = None
        if cfg.desktop_monitor:
            monitor = pick_monitor(cfg.desktop_monitor, display)
        return WaylandStage(monitor.rect if monitor else None,
                            output=monitor.name if monitor else '')

    if not display:
        raise StageUnavailable(
            'there is no X display to drive. Set DISPLAY, or ROOST_DESKTOP_STAGE=virtual '
            'to give the agent one of its own.'
        )

    monitor = pick_monitor(cfg.desktop_monitor, display) if cfg.desktop_monitor else None
    return XStage(
        display,
        region=monitor.rect if monitor else None,
        kind='shared',
        shares_pointer=True,
        yield_to_user=cfg.desktop_yield_to_user,
    )


def _parse_size(text: str, fallback: tuple[int, int] = (1920, 1080)) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in str(text).lower().split('x', 1))
    except (ValueError, AttributeError):
        return fallback
    return max(320, width), max(240, height)
