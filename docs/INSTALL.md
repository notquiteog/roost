# Installing

Roost is a daemon plus a client. The daemon runs on the machine you want the
agent to control; the client is a page it serves. That shape is why there is
no domain, no certificate and no account anywhere in this document.

## No domain is needed. No certificate either.

`http://127.0.0.1` **is a secure context** as far as every browser is
concerned — localhost is specifically exempted from the HTTPS requirement. So
the microphone, `AudioWorklet` and everything else the voice pipeline needs
work on plain HTTP, with no certificate at all.

Measured on this machine:

| origin | secure context | `navigator.mediaDevices` |
|---|---|---|
| `http://127.0.0.1:8477` | yes | available |
| `http://192.168.1.x:8477` | **no** | **undefined** |

So the only case that needs anything extra is reaching Roost from *another*
machine — and the answer there is not a domain either.

## Reaching it from your phone or laptop

Forward the port over SSH:

```bash
ssh -N -L 8477:127.0.0.1:8477 you@your-desktop
```

The far end is now `http://127.0.0.1:8477` on the machine you are sitting at —
which is a secure context, so voice works, with no certificate, no domain and
nothing published to the internet. This is the same trick Perch uses to reach
a GPU box, in the same direction.

For a phone, any Tailscale-style overlay does the same job. `mkcert` is a
third option, and the only one that needs a local CA in your trust store —
which is a lot to ask of an installer, so it is not the default.

## Installing the daemon

Python 3.11+, on Linux, macOS or Windows:

```bash
uv tool install roost
```

or `pipx install roost`. Then:

```bash
roost
```

and open <http://127.0.0.1:8477>.

For the browser tools, once:

```bash
roost-browser-setup    # wraps: playwright install chromium
```

## Platform notes

The shell tool is the only part that has to care which system it is on, and
it is written for all three:

* Commands run in their own process group — `start_new_session` on POSIX,
  `CREATE_NEW_PROCESS_GROUP` on Windows — so a timeout or an interrupt reaches
  the command's *children* and not just the shell. Killing only the shell
  leaves an orphaned dev server holding its port.
* Killing uses `killpg` on POSIX and `taskkill /F /T` on Windows, because
  Windows has no process groups in the POSIX sense and no `SIGTERM`.
* The risk classifier knows `cmd.exe` and PowerShell as well as POSIX shells —
  `Remove-Item -Recurse -Force`, `del /f /s /q`, `format C:`, `vssadmin delete
  shadows` are all graded destructive. Both families are matched on every
  platform, because Git Bash and WSL put `bash` on Windows and PowerShell runs
  on Linux and macOS.
* PowerShell cmdlet names are matched case-insensitively; POSIX command names
  are not, because lower-casing `LS` into `ls` would grade an unknown binary
  as a known-safe read.

## Running it as a service

Nothing here survives a reboot yet. An autonomous agent probably should, so
this is the next thing worth adding: a systemd user unit on Linux, a
`launchd` plist on macOS, and a scheduled task or service on Windows.

## Do you need a desktop app?

Not for this to work. A native shell would add a tray icon, a dock entry and
OS notifications, and it would wrap the *same* page — so it is additive rather
than a fork in the road.

One thing it would not add on every platform is a global push-to-talk hotkey.
On Wayland that needs the `org.freedesktop.portal.GlobalShortcuts` portal, and
on the COSMIC session this was developed on that portal is not present — so no
framework, native or otherwise, can offer it there today.

If you do build one, Tauri wraps the existing HTML for about 10 MB. Note that
on Linux it renders in WebKitGTK rather than Chromium, which is the least
exercised of the three engines for microphone capture — worth testing the
voice path early rather than late.
