"""Driving Windows, and giving the agent a desktop of its own.

Windows is the one platform with a first-class answer to the question this
whole stage layer exists for. A *desktop* here is not a metaphor: `HDESK` is
a kernel object with its own input queue, its own cursor position, and its
own set of top-level windows. A process launched onto one cannot draw on
another, cannot read another's input, and cannot steal focus from it. That is
the same guarantee Xvfb gives on Linux, obtained from the operating system
rather than from a second X server.

So the promise holds here for the same reason it holds there, and for once
the platform is doing the work.

Three things about this file are not obvious.

**Everything runs on one thread.** `SetThreadDesktop` binds the *calling
thread* to a desktop, and it silently refuses if that thread already owns a
window or a hook. Input sent from an unbound thread lands on whichever
desktop that thread happens to be on -- which is the user's, which is the
failure this file exists to prevent. So a desktop owns a single worker
thread, bound once when it starts, and every call is marshalled onto it.
That also makes the binding a property of the object rather than of whoever
happened to call it, which matters because the capture path is invoked from
`asyncio.to_thread` and therefore from a different thread each time.

**Launching is not an environment variable.** On X, putting an application
on a display is `DISPLAY=:90` and the toolkit does the rest. Windows has no
equivalent: the desktop is a field in `STARTUPINFO`, read by `CreateProcess`
and nowhere else. `Stage.env()` cannot express that, which is why this module
exposes `spawn()` instead and the Windows stage overrides how it launches.

**Capture is the hard part, not input.** A desktop nobody is looking at is
not composited by the desktop window manager, so the classic trick of
blitting the desktop device context comes back black for anything drawn with
hardware acceleration -- which today is most things, a browser included. The
fallback is to ask each window to paint itself with `PrintWindow` and
`PW_RENDERFULLCONTENT`, compositing the results by their own rectangles.
That is slower and imperfect for video and WebGL, and this file says so
rather than pretending the picture is always faithful.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger(__name__)


class Win32Unavailable(RuntimeError):
    """Windows cannot provide what was asked for on this machine.

    Its own error rather than `StageUnavailable` so this module stays
    importable without the stage layer -- which also keeps the import
    one-directional, since the stage layer is what imports this.
    """


# ---------------------------------------------------------------------------
# Bindings
# ---------------------------------------------------------------------------
#
# ctypes rather than pywin32, for the same reason the X layer is ctypes rather
# than python-xlib: this needs about twenty function signatures, and a
# compiled dependency on a platform that is already awkward to install on
# would cost more than it saves.

if sys.platform == 'win32':  # pragma: no cover - imported for its symbols only elsewhere
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL('user32', use_last_error=True)
    gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

    ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32
else:
    ctypes = None  # type: ignore[assignment]
    wintypes = None  # type: ignore[assignment]
    user32 = gdi32 = kernel32 = None  # type: ignore[assignment]
    ULONG_PTR = None  # type: ignore[assignment]


# Desktop access rights. GENERIC_ALL is what a desktop you created and will
# destroy needs; the granular ones exist so a future read-only attach can ask
# for less than everything.
GENERIC_ALL = 0x10000000

# SendInput.
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

WHEEL_DELTA = 120

SM_CXSCREEN = 0
SM_CYSCREEN = 1

SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0
BI_RGB = 0

PW_RENDERFULLCONTENT = 0x00000002

STARTF_USESHOWWINDOW = 0x00000001
SW_SHOW = 5
CREATE_UNICODE_ENVIRONMENT = 0x00000400

# The buttons, as down/up flag pairs. Mirrors BUTTONS in the stage layer.
_BUTTONS = {
    'left': (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    'middle': (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
    'right': (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
}

# Virtual key codes for the keys that have names rather than characters. The
# names are the same ones the X stage accepts, so a combination written for
# one platform means the same thing on the other -- an agent should not have
# to know which operating system it is on to press Enter.
VK = {
    'enter': 0x0D, 'return': 0x0D, 'esc': 0x1B, 'escape': 0x1B,
    'backspace': 0x08, 'tab': 0x09, 'space': 0x20, 'delete': 0x2E,
    'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27,
    'home': 0x24, 'end': 0x23, 'pageup': 0x21, 'pagedown': 0x22,
    'insert': 0x2D, 'menu': 0x5D, 'print': 0x2C,
    **{f'f{n}': 0x70 + n - 1 for n in range(1, 13)},
}

VK_MODIFIERS = {
    'ctrl': 0x11, 'control': 0x11, 'shift': 0x10, 'alt': 0x12,
    'super': 0x5B, 'meta': 0x5B, 'cmd': 0x5B,
}

# Keys that live on the extended part of the keyboard. Without the extended
# flag the arrows and navigation cluster are delivered as their numeric-keypad
# twins, so Home types 7 in an application that is reading scan codes.
_EXTENDED = {0x26, 0x28, 0x25, 0x27, 0x24, 0x23, 0x21, 0x22, 0x2D, 0x2E, 0x5D}


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

if sys.platform == 'win32':  # pragma: no cover - Windows only

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ('dx', wintypes.LONG), ('dy', wintypes.LONG),
            ('mouseData', wintypes.DWORD), ('dwFlags', wintypes.DWORD),
            ('time', wintypes.DWORD), ('dwExtraInfo', ULONG_PTR),
        ]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ('wVk', wintypes.WORD), ('wScan', wintypes.WORD),
            ('dwFlags', wintypes.DWORD), ('time', wintypes.DWORD),
            ('dwExtraInfo', ULONG_PTR),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [('uMsg', wintypes.DWORD), ('wParamL', wintypes.WORD), ('wParamH', wintypes.WORD)]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [('mi', MOUSEINPUT), ('ki', KEYBDINPUT), ('hi', HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ('u',)
        _fields_ = [('type', wintypes.DWORD), ('u', _INPUTUNION)]

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ('biSize', wintypes.DWORD), ('biWidth', wintypes.LONG), ('biHeight', wintypes.LONG),
            ('biPlanes', wintypes.WORD), ('biBitCount', wintypes.WORD),
            ('biCompression', wintypes.DWORD), ('biSizeImage', wintypes.DWORD),
            ('biXPelsPerMeter', wintypes.LONG), ('biYPelsPerMeter', wintypes.LONG),
            ('biClrUsed', wintypes.DWORD), ('biClrImportant', wintypes.DWORD),
        ]

    class BITMAPINFO(ctypes.Structure):
        _fields_ = [('bmiHeader', BITMAPINFOHEADER), ('bmiColors', wintypes.DWORD * 3)]

    class STARTUPINFOW(ctypes.Structure):
        _fields_ = [
            ('cb', wintypes.DWORD), ('lpReserved', wintypes.LPWSTR),
            ('lpDesktop', wintypes.LPWSTR), ('lpTitle', wintypes.LPWSTR),
            ('dwX', wintypes.DWORD), ('dwY', wintypes.DWORD),
            ('dwXSize', wintypes.DWORD), ('dwYSize', wintypes.DWORD),
            ('dwXCountChars', wintypes.DWORD), ('dwYCountChars', wintypes.DWORD),
            ('dwFillAttribute', wintypes.DWORD), ('dwFlags', wintypes.DWORD),
            ('wShowWindow', wintypes.WORD), ('cbReserved2', wintypes.WORD),
            ('lpReserved2', ctypes.POINTER(ctypes.c_byte)),
            ('hStdInput', wintypes.HANDLE), ('hStdOutput', wintypes.HANDLE),
            ('hStdError', wintypes.HANDLE),
        ]

    class PROCESS_INFORMATION(ctypes.Structure):
        _fields_ = [
            ('hProcess', wintypes.HANDLE), ('hThread', wintypes.HANDLE),
            ('dwProcessId', wintypes.DWORD), ('dwThreadId', wintypes.DWORD),
        ]

    def _bind_signatures() -> None:
        """Argument and return types, so ctypes does not guess at a boundary.

        Guessing is fine until a handle is truncated to 32 bits on a 64-bit
        build, at which point the failure is a desktop that cannot be found
        rather than an obvious crash.
        """
        user32.CreateDesktopW.restype = wintypes.HANDLE
        user32.CreateDesktopW.argtypes = [
            wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.c_void_p,
            wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        ]
        user32.OpenDesktopW.restype = wintypes.HANDLE
        user32.OpenDesktopW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        user32.SetThreadDesktop.restype = wintypes.BOOL
        user32.SetThreadDesktop.argtypes = [wintypes.HANDLE]
        user32.CloseDesktop.restype = wintypes.BOOL
        user32.CloseDesktop.argtypes = [wintypes.HANDLE]
        user32.GetThreadDesktop.restype = wintypes.HANDLE
        user32.GetThreadDesktop.argtypes = [wintypes.DWORD]

        user32.SendInput.restype = wintypes.UINT
        user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
        user32.GetSystemMetrics.restype = ctypes.c_int
        user32.GetSystemMetrics.argtypes = [ctypes.c_int]
        user32.GetCursorPos.restype = wintypes.BOOL
        user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]

        user32.GetDC.restype = wintypes.HDC
        user32.GetDC.argtypes = [wintypes.HWND]
        user32.ReleaseDC.restype = ctypes.c_int
        user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        user32.PrintWindow.restype = wintypes.BOOL
        user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
        user32.GetWindowRect.restype = wintypes.BOOL
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.IsWindowVisible.restype = wintypes.BOOL
        user32.IsWindowVisible.argtypes = [wintypes.HWND]

        gdi32.CreateCompatibleDC.restype = wintypes.HDC
        gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
        gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
        gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
        gdi32.SelectObject.restype = wintypes.HGDIOBJ
        gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        gdi32.BitBlt.restype = wintypes.BOOL
        gdi32.BitBlt.argtypes = [
            wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD,
        ]
        gdi32.GetDIBits.restype = ctypes.c_int
        gdi32.GetDIBits.argtypes = [
            wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
            ctypes.c_void_p, ctypes.POINTER(BITMAPINFO), wintypes.UINT,
        ]
        gdi32.DeleteObject.restype = wintypes.BOOL
        gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        gdi32.DeleteDC.restype = wintypes.BOOL
        gdi32.DeleteDC.argtypes = [wintypes.HDC]

        kernel32.CreateProcessW.restype = wintypes.BOOL
        kernel32.CreateProcessW.argtypes = [
            wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
            wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR,
            ctypes.POINTER(STARTUPINFOW), ctypes.POINTER(PROCESS_INFORMATION),
        ]

    _bind_signatures()


if sys.platform == 'win32':  # pragma: no cover - Windows only
    user32.VkKeyScanW.restype = ctypes.c_short
    user32.VkKeyScanW.argtypes = [wintypes.WCHAR]
    _WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumDesktopWindows.restype = wintypes.BOOL
    user32.EnumDesktopWindows.argtypes = [wintypes.HANDLE, _WNDENUMPROC, wintypes.LPARAM]


def available() -> bool:
    """Whether this module can do anything at all here."""
    return sys.platform == 'win32'


def _require() -> None:
    if not available():
        raise Win32Unavailable(f'this is {sys.platform}, not Windows')


def screen_size() -> tuple[int, int]:
    """The size of the display this session is attached to.

    Note for the virtual stage: a created desktop inherits the session's
    display metrics and cannot be given a size of its own, so the
    `desktop_virtual_size` setting -- which Xvfb honours -- has no effect on
    Windows. Documented rather than silently ignored.
    """
    _require()
    return user32.GetSystemMetrics(SM_CXSCREEN), user32.GetSystemMetrics(SM_CYSCREEN)


class Desktop:
    """A Windows desktop the agent can look at and act on.

    `own=True` creates one. `own=False` binds to the desktop the process is
    already on, which is the user's -- the shared stage, where the cursor
    being moved is the one on their desk.
    """

    def __init__(self, name: str | None = None, *, own: bool = True) -> None:
        _require()
        self.own = own
        self.name = name or f'openmirror-{kernel32.GetCurrentProcessId()}-{int(time.time() * 1000) % 100000}'
        self._closed = False

        if own:
            handle = user32.CreateDesktopW(self.name, None, None, 0, GENERIC_ALL, None)
            if not handle:
                err = ctypes.get_last_error()
                raise Win32Unavailable(
                    f'could not create a desktop of its own (error {err}). This needs a session '
                    'that owns an interactive window station; a service running as LocalSystem '
                    'without one cannot do it.'
                )
            self.hdesk = handle
        else:
            self.hdesk = user32.GetThreadDesktop(kernel32.GetCurrentThreadId())
            if not self.hdesk:
                raise Win32Unavailable('no desktop is attached to this thread')

        # One thread, bound once. See the module docstring: the binding is a
        # property of the thread, so it cannot be left to whichever caller
        # happens to arrive.
        self._pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f'stage-{self.name}',
            initializer=self._bind,
        )
        self._lock = threading.Lock()

    def _bind(self) -> None:
        if not user32.SetThreadDesktop(self.hdesk):
            err = ctypes.get_last_error()
            # Error 170 is ERROR_BUSY, and here it means precisely one thing:
            # this thread already owns a window or a hook. A freshly created
            # pool thread does not, so if it happens the pool was reused.
            raise Win32Unavailable(f'could not bind this thread to the desktop (error {err})')

    def _call(self, fn, *args):
        """Run something on the thread that belongs to this desktop."""
        if self._closed:
            raise Win32Unavailable('this desktop has been closed')
        return self._pool.submit(fn, *args).result()

    # -- input --------------------------------------------------------------

    @staticmethod
    def _send(*inputs) -> None:
        array = (INPUT * len(inputs))(*inputs)
        sent = user32.SendInput(len(inputs), array, ctypes.sizeof(INPUT))
        if sent != len(inputs):
            raise Win32Unavailable(f'input was refused (error {ctypes.get_last_error()})')

    @staticmethod
    def _mouse(flags: int, dx: int = 0, dy: int = 0, data: int = 0):
        inp = INPUT(type=INPUT_MOUSE)
        inp.mi = MOUSEINPUT(dx, dy, data & 0xFFFFFFFF, flags, 0, 0)
        return inp

    @staticmethod
    def _keyb(vk: int, flags: int = 0, scan: int = 0):
        inp = INPUT(type=INPUT_KEYBOARD)
        inp.ki = KEYBDINPUT(vk, scan, flags, 0, 0)
        return inp

    def _move(self, x: int, y: int) -> None:
        width, height = screen_size()
        # Absolute motion is in a 0..65535 space over the display rather than
        # in pixels, and the scale is over width-1 so that the last pixel is
        # reachable at all -- dividing by width leaves the right-hand column
        # permanently unclickable.
        nx = int(x * 65535 / max(width - 1, 1))
        ny = int(y * 65535 / max(height - 1, 1))
        self._send(self._mouse(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, nx, ny))

    def move(self, x: int, y: int) -> None:
        self._call(self._move, int(x), int(y))

    def _click(self, button: str, double: bool) -> None:
        down, up = _BUTTONS.get(button, _BUTTONS['left'])
        for i in range(2 if double else 1):
            if i:
                time.sleep(0.06)
            self._send(self._mouse(down))
            time.sleep(0.02)
            self._send(self._mouse(up))

    def click(self, button: str = 'left', double: bool = False) -> None:
        self._call(self._click, button, double)

    def _scroll(self, amount: int) -> None:
        # One notch per unit, matching what the X stage does with buttons 4
        # and 5, so a scroll of 3 means the same thing on both platforms.
        notches = max(-30, min(30, int(amount)))
        if not notches:
            return
        self._send(self._mouse(MOUSEEVENTF_WHEEL, data=notches * WHEEL_DELTA))

    def scroll(self, amount: int) -> None:
        self._call(self._scroll, amount)

    def _type_text(self, text: str) -> None:
        for char in text:
            if char == '\n':
                self._tap(VK['enter'], [])
            elif char == '\t':
                self._tap(VK['tab'], [])
            else:
                # Injected as a Unicode code unit rather than as a key on some
                # layout. This is the Windows equivalent of reading the X
                # keymap: it types the character that was asked for whatever
                # the keyboard is set to, and it needs no table.
                for unit in _utf16_units(char):
                    self._send(self._keyb(0, KEYEVENTF_UNICODE, unit))
                    self._send(self._keyb(0, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, unit))
            time.sleep(0.012)

    def type_text(self, text: str) -> None:
        self._call(self._type_text, text)

    def _tap(self, vk: int, modifiers: list[int]) -> None:
        flags = KEYEVENTF_EXTENDEDKEY if vk in _EXTENDED else 0
        for mod in modifiers:
            self._send(self._keyb(mod))
        self._send(self._keyb(vk, flags))
        self._send(self._keyb(vk, flags | KEYEVENTF_KEYUP))
        for mod in reversed(modifiers):
            self._send(self._keyb(mod, KEYEVENTF_KEYUP))

    def _key(self, combo: str) -> None:
        parts = [p.strip().lower() for p in combo.split('+') if p.strip()]
        if not parts:
            raise ValueError('empty key combination')

        modifiers = []
        for part in parts[:-1]:
            vk = VK_MODIFIERS.get(part)
            if vk is None:
                raise ValueError(f'unknown modifier: {part!r}')
            modifiers.append(vk)

        last = parts[-1]
        vk = VK.get(last)
        if vk is None:
            if len(last) != 1:
                raise ValueError(f'unknown key: {last!r}')
            # A shortcut is a *key*, not a character, so this asks the layout
            # which key produces it. Unicode injection would type the letter
            # instead of pressing the accelerator, and ctrl+c would insert a
            # 'c' rather than copying.
            scan = user32.VkKeyScanW(last)
            if scan == -1:
                raise ValueError(f'this keyboard layout cannot produce {last!r}')
            vk = scan & 0xFF
            if scan & 0x100 and VK_MODIFIERS['shift'] not in modifiers:
                modifiers.append(VK_MODIFIERS['shift'])
        self._tap(vk, modifiers)

    def key(self, combo: str) -> None:
        self._call(self._key, combo)

    def pointer(self) -> tuple[int, int]:
        def read():
            point = wintypes.POINT()
            user32.GetCursorPos(ctypes.byref(point))
            return point.x, point.y
        return self._call(read)

    # -- looking ------------------------------------------------------------

    def capture(self) -> bytes:
        """A PNG of this desktop, taken on the thread that belongs to it.

        The binding matters as much here as it does for input: `GetDC(NULL)`
        returns a context for the *calling thread's* desktop, so a capture
        taken from an unbound thread is a picture of the user's screen
        labelled as the agent's.
        """
        return self._call(_capture_png, self.hdesk)

    def spawn(self, argv: list[str], *, cwd: str | None = None, env: dict | None = None) -> int:
        """Start an application on this desktop. Returns its process id."""
        desktop = f'WinSta0\\{self.name}' if self.own else None
        return self._call(_spawn_on, desktop, argv, cwd, env)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._pool.shutdown(wait=True)
        if self.own and self.hdesk:
            user32.CloseDesktop(self.hdesk)
        self.hdesk = None


def _utf16_units(char: str) -> list[int]:
    """A character as the one or two code units Windows wants.

    Anything outside the basic plane -- an emoji in a form field, say -- is a
    surrogate pair, and sending only the first half inserts a replacement
    character.
    """
    raw = char.encode('utf-16-le')
    return [int.from_bytes(raw[i:i + 2], 'little') for i in range(0, len(raw), 2)]


# ---------------------------------------------------------------------------
# Looking at a desktop nobody is looking at
# ---------------------------------------------------------------------------

BLACKNESS = 0x00000042
OBJ_BITMAP = 7


def _read_back(hdc, bitmap, width: int, height: int) -> bytes:
    """Read a device context back as top-down BGRX bytes."""
    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = width
    # Negative, so the rows arrive in the order every image library expects.
    # A positive height gives a bottom-up bitmap and a picture of the screen
    # upside down, which the model will then describe with full confidence.
    bmi.bmiHeader.biHeight = -height
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = BI_RGB

    buf = (ctypes.c_char * (width * height * 4))()
    if not gdi32.GetDIBits(hdc, bitmap, 0, height, buf, ctypes.byref(bmi), DIB_RGB_COLORS):
        raise Win32Unavailable(f'GetDIBits failed (error {ctypes.get_last_error()})')
    return bytes(buf)


def looks_black(raw: bytes) -> bool:
    """Whether a capture came back empty.

    Sampled rather than scanned: a 1920x1080 frame is eight megabytes and this
    runs several times a second, so walking every byte in Python to decide
    whether to walk it again is the wrong trade. A prime stride spreads the
    samples across rows rather than landing in the same column of each.

    The sampling can miss a small bright region, and that is deliberate: a
    false "black" costs one unnecessary trip through the window-compositing
    path, which returns a correct picture anyway. A false "not black" would
    return the black frame itself. The cheap error is the recoverable one, so
    the bias is towards paying for the slow path.

    Public because it is the one piece of the capture path with logic worth
    testing off Windows.
    """
    step = 997 * 4
    for i in range(0, max(len(raw) - 3, 0), step):
        if raw[i] or raw[i + 1] or raw[i + 2]:
            return False
    return True


def _composite_windows(hdesk, hdc_screen, width: int, height: int):
    """Ask every window on a desktop to paint itself, and stack the results.

    This exists because a desktop that is not being displayed is not being
    composited either, so blitting it gives back black for anything drawn with
    hardware acceleration -- a browser most of all, which is the thing the
    agent is usually driving.

    `PW_RENDERFULLCONTENT` is what makes it work for those windows at all. It
    is still not a faithful picture: video and WebGL surfaces can come back
    blank, and a window caught mid-resize paints torn. It is good enough to
    read a page and find a button, which is what it is for, and this says so
    rather than implying the frame is always true.
    """
    hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
    composite = gdi32.CreateCompatibleBitmap(hdc_screen, width, height)
    old = gdi32.SelectObject(hdc_mem, composite)
    gdi32.PatBlt(hdc_mem, 0, 0, width, height, BLACKNESS)

    windows: list[int] = []

    def collect(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd):
            windows.append(hwnd)
        return True

    user32.EnumDesktopWindows(hdesk, _WNDENUMPROC(collect), 0)

    # Reversed, because EnumDesktopWindows walks front to back and painting in
    # that order would leave the bottom window on top.
    for hwnd in reversed(windows):
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            continue
        w, h = rect.right - rect.left, rect.bottom - rect.top
        if w <= 0 or h <= 0 or w > width * 2 or h > height * 2:
            continue

        hdc_win = gdi32.CreateCompatibleDC(hdc_screen)
        bmp_win = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
        old_win = gdi32.SelectObject(hdc_win, bmp_win)
        try:
            if user32.PrintWindow(hwnd, hdc_win, PW_RENDERFULLCONTENT):
                gdi32.BitBlt(hdc_mem, rect.left, rect.top, w, h, hdc_win, 0, 0, SRCCOPY)
        finally:
            gdi32.SelectObject(hdc_win, old_win)
            gdi32.DeleteObject(bmp_win)
            gdi32.DeleteDC(hdc_win)

    return hdc_mem, composite, old


def _capture_png(hdesk) -> bytes:
    """One frame of a desktop, as PNG.

    Tries the cheap path first and falls back only when it comes back empty,
    so the shared desktop -- which always composites -- never pays for the
    expensive one.
    """
    width, height = screen_size()
    hdc_screen = user32.GetDC(None)
    if not hdc_screen:
        raise Win32Unavailable('could not get a device context for this desktop')

    try:
        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        bitmap = gdi32.CreateCompatibleBitmap(hdc_screen, width, height)
        old = gdi32.SelectObject(hdc_mem, bitmap)
        try:
            gdi32.BitBlt(hdc_mem, 0, 0, width, height, hdc_screen, 0, 0, SRCCOPY)
            raw = _read_back(hdc_mem, bitmap, width, height)
        finally:
            gdi32.SelectObject(hdc_mem, old)
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(hdc_mem)

        if not looks_black(raw):
            return to_png(raw, width, height)

        log.debug('the desktop blit came back black; asking each window to paint itself')
        hdc_mem, composite, old = _composite_windows(hdesk, hdc_screen, width, height)
        try:
            raw = _read_back(hdc_mem, composite, width, height)
        finally:
            gdi32.SelectObject(hdc_mem, old)
            gdi32.DeleteObject(composite)
            gdi32.DeleteDC(hdc_mem)
        return to_png(raw, width, height)
    finally:
        user32.ReleaseDC(None, hdc_screen)


def to_png(raw: bytes, width: int, height: int) -> bytes:
    """Top-down BGRX bytes as a PNG."""
    try:
        import io

        from PIL import Image
    except ImportError as exc:
        raise Win32Unavailable(f'screen capture needs pillow: {exc}') from exc

    image = Image.frombuffer('RGB', (width, height), raw, 'raw', 'BGRX', 0, 1)
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


def _spawn_on(desktop: str | None, argv: list[str], cwd: str | None, env: dict | None) -> int:
    """Start an application on a named desktop.

    The reason this module exposes a launcher at all. On X, putting an
    application on a display is an environment variable and every toolkit
    obeys it; here the desktop is a field in STARTUPINFO that only
    CreateProcess reads, so neither `subprocess` nor `Stage.env()` can say it.
    """
    startup = STARTUPINFOW()
    startup.cb = ctypes.sizeof(STARTUPINFOW)
    # The window station stays the interactive one; only the desktop differs.
    startup.lpDesktop = desktop
    startup.dwFlags = STARTF_USESHOWWINDOW
    startup.wShowWindow = SW_SHOW

    info = PROCESS_INFORMATION()
    command = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))

    block = None
    flags = 0
    if env is not None:
        flags |= CREATE_UNICODE_ENVIRONMENT
        block = ctypes.create_unicode_buffer('\0'.join(f'{k}={v}' for k, v in env.items()) + '\0\0')

    ok = kernel32.CreateProcessW(
        None, command, None, None, False, flags,
        ctypes.cast(block, ctypes.c_void_p) if block else None,
        cwd, ctypes.byref(startup), ctypes.byref(info),
    )
    if not ok:
        raise Win32Unavailable(
            f'could not start {argv[0]!r} on the desktop (error {ctypes.get_last_error()})'
        )
    kernel32.CloseHandle(info.hThread)
    kernel32.CloseHandle(info.hProcess)
    return info.dwProcessId
