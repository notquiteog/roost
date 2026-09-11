# The desktop shell

A Tauri window around the interface the daemon already serves. It contains no
application logic — everything openmirror does lives in the Python daemon — and
exists for the three things a browser tab cannot do: live in the tray, survive
being closed, and put a notification in front of you when the agent is waiting
on an answer.

## What it does at startup

It checks whether a daemon is already listening. If one is, it **attaches** and
leaves it alone on exit — killing a daemon somebody else started would take
their running sessions with it. If nothing is there, it starts one (`openmirror`, or
`python3 -m openmirror.main`) and stops that one when you quit.

Closing the window hides it. Quit from the tray is the way out, so an agent
four minutes into a build is not stopped by someone tidying their desktop.

## Building

```bash
cd desktop/src-tauri
cargo build --release
```

Linux needs the webview development headers first:

```bash
sudo apt install libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev libxdo-dev patchelf
```

Installers for all three platforms are built by
[`.github/workflows/desktop.yml`](../.github/workflows/desktop.yml) — one
runner each, because a Tauri app cannot practically be cross-compiled: every
target needs its own system webview and its own bundler.

## Transparency

`OPENMIRROR_TRANSPARENT=1` makes the window transparent so the interface's own glass
blurs the real desktop behind it, rather than a background it painted itself.

It is **off by default because it is compositor-dependent**. On COSMIC/Wayland
a transparent window did not map at all — no error, no window, which is
indistinguishable from the app failing to start. Off, the window always
appears; on, you get the nicer effect if your compositor supports it.

The page is *told* when the window is transparent, rather than sniffing the
user agent: whether transparency actually worked is a property of the
compositor, which the page cannot observe, and dropping its background without
it would leave the interface floating on nothing.
