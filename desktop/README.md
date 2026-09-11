# The desktop app

A native window and a tray icon around the interface the daemon serves — with
the daemon inside. Installing the app is installing openmirror: a release
carries its own daemon, frozen with PyInstaller, so there is no Python to set
up and no terminal to keep open.

It contains no application logic. Everything openmirror does lives in the
daemon, and the app exists for what a browser tab cannot do: live in the
tray, survive being closed, tell you when the agent is waiting on you, and
keep the daemon running for exactly as long as the app is.

## Getting it

From the [releases](https://github.com/notquiteog/openmirror/releases): an
`.exe` or `.msi` for Windows, a `.dmg` for Apple Silicon or Intel Macs, and a
`.deb`, `.rpm` or `.AppImage` for Linux. Neither the Windows nor the macOS
build is signed yet, so each warns once before it will open; the release notes
say what to click.

## What it does at start-up

The window opens at once, on a page of its own that says what is happening,
and moves to the interface as soon as there is one to show.

- **A daemon that is already running is used, not replaced.** The app looks
  on `OPENMIRROR_PORT` — from the environment, then `~/.openmirror/.env`, then
  8477, the order the daemon itself reads them in. If openmirror answers there,
  the app attaches, and leaves that daemon running when it quits: killing one
  somebody else started would take their sessions with it.
- **If something else holds the port,** the page says so, rather than showing
  someone else's server in the window.
- **If nothing is there, it starts one:** the daemon it shipped with, or — in a
  build without one — `openmirror` from your PATH, then `python3 -m
  openmirror.main` (`OPENMIRROR_PYTHON` names the interpreter).

A daemon the app starts:

- runs in your home directory, as it would from a new terminal, so that is its
  default workspace too — set `OPENMIRROR_WORKSPACE` to narrow it;
- keeps its data in `~/.openmirror/data`, reads `~/.openmirror/.env`, and logs
  to `~/.openmirror/logs/daemon.log`, with the previous run's kept beside it;
- on macOS, gets the PATH your login shell would give it. An app opened from
  the Dock gets only the system directories, and every command the agent ran
  would be missing Homebrew, `~/.local/bin` and the rest;
- **stops with the app, however the app stops.** It is started with a pipe on
  stdin that the app holds and never writes to (`OPENMIRROR_EXIT_WITH_STDIN`),
  and it shuts itself down when that closes — which the operating system does
  on a quit, a crash and a kill alike. There is no orphaned daemon left holding
  the port for the next launch to mistake for somebody else's.

If the daemon fails to start, or stops later, the window goes back to its own
page with the end of the log and a **Try again** button. The app's own notes
are in `~/.openmirror/logs/desktop.log`, and the tray has **Open the logs
folder**.

## The window

- **Closing it hides it.** An agent four minutes into a build is not stopped
  by someone tidying their desktop. **Quit** is in the tray; on macOS, clicking
  the Dock icon brings the window back.
- **There is only one.** Opening the app again brings the running one forward.
- **Links to anywhere else open in your browser**, which has an address bar,
  your passwords and a back button. This window has none of them, and a page
  that navigated it away would leave no way back to the interface.
- **Downloads** — Studio's, for one — go to your Downloads folder, under a name
  that overwrites nothing, with a notification saying where.
- **A notification when the agent is waiting on you** — an approval, or a
  question — and you are looking at something else. The daemon's page asks for
  it (`openmirror/static/native.js`), and it is the only thing that page is
  allowed to ask the app for.
- **The microphone,** for Talk and Live. macOS asks you once and Windows'
  WebView2 asks with its own prompt. On Linux WebKitGTK refuses any request
  nobody answers, which left voice hearing nothing, so the app answers it — for
  the daemon's page only.

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

## Building it

Linux needs the webview headers first:

```bash
sudo apt install libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev libxdo-dev libssl-dev patchelf
```

To work on the app itself, `cargo run` in `src-tauri` is enough: with no
frozen daemon beside it, it starts the `openmirror` you already have.

A build with the daemon inside is two steps — freeze the daemon with the
extras a release carries, then bundle:

```bash
pip install '.[desktop,tor,vec]' pyinstaller
python desktop/sidecar/build.py
cd desktop && npx @tauri-apps/cli@2 build --config src-tauri/tauri.release.conf.json
```

A frozen daemon is only right for the kind of machine that froze it, so every
platform builds its own.

## Releasing

Set the new version in `pyproject.toml` and `desktop/src-tauri/Cargo.toml` —
the Tauri config has none of its own, so it takes Cargo's — let a build update
`Cargo.lock`, commit, and push a tag:

```bash
git tag v0.2.0
git push origin v0.2.0
```

[`.github/workflows/release.yml`](../.github/workflows/release.yml) then:

1. checks that the tag, `pyproject.toml` and `Cargo.toml` agree, and runs the
   tests;
2. freezes the daemon and smoke-tests it, then builds the app, on Linux,
   Windows, and Apple Silicon and Intel macOS — each on its own runner, because
   neither PyInstaller nor Tauri cross-compiles;
3. installs every installer on a fresh machine the way a person would — the
   `.deb` with `apt` on Ubuntu 22.04 and 24.04, the AppImage, the `.exe`
   silently and the `.msi` on Windows, the `.dmg` on both Macs — opens it,
   checks the window loaded the interface from the daemon, opens a second
   copy, then kills or quits it and checks the daemon went with it;
4. only then publishes the release, with checksums. A tag with a hyphen in it,
   like `v0.2.0-rc.1`, is published as a pre-release.

To do all of that without publishing anything, run the workflow by hand from
the Actions tab, or open a pull request that touches `desktop/`.

## macOS, without a Mac

Nobody on the project owns one, so CI is the Mac. On both architectures, for
every release, it checks that the disk image mounts, the app copies into
Applications, its signature verifies, it opens through LaunchServices, the
window loads the interface, and both Quit and a crash stop the daemon. It also
fails the build if anything frozen into the daemon needs a newer macOS than
the app declares (`minimumSystemVersion`, 11.0), since pip otherwise takes
packages built for the runner's own, newer macOS.

What it cannot check: Gatekeeper's refusal of a downloaded copy, the
microphone and screen-recording prompts, notifications, the menu-bar icon, and
anything about real hardware.

Releases are signed ad hoc: enough for Apple Silicon to run them, not enough
for Gatekeeper to trust them. Given an Apple Developer ID, adding
`APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD`, `APPLE_SIGNING_IDENTITY`,
`APPLE_ID`, `APPLE_PASSWORD` and `APPLE_TEAM_ID` as repository secrets makes the
same workflow sign with it and notarise. That path has never run. It turns on
the hardened runtime, which is what `Entitlements.plist` is for, and the frozen
daemon is where it would fail first.

Windows builds are not signed either; SmartScreen warns once.
