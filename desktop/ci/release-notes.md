Each installer is the whole of openmirror — the app, and the daemon it runs, with its own Python inside. Nothing else needs installing. It keeps its data, its logs and an optional `.env` in `~/.openmirror`. If a daemon is already running on port 8477, one you started yourself, the app uses that one and leaves it running when it quits.

## Windows

`…_x64-setup.exe` installs for you alone, with no administrator prompt; the `.msi` installs for everyone on the machine.

The installers are not code-signed yet, so SmartScreen will say it protected your PC. Choose **More info**, then **Run anyway**.

## macOS

`…_aarch64.dmg` is for Apple Silicon, `…_x64.dmg` for Intel. Open it and drag openmirror into Applications.

openmirror is not notarised yet — the project has no Apple Developer ID — so the first time you open it, macOS will refuse. Either go to **System Settings → Privacy & Security**, find the line about openmirror near the bottom and choose **Open Anyway**, or run this once:

```
xattr -dr com.apple.quarantine /Applications/openmirror.app
```

Nobody on the project owns a Mac. These builds are installed and started on GitHub's macOS machines before they are published here, which is the whole of their testing: reports from real Macs are especially welcome.

## Linux

- Debian, Ubuntu: `sudo apt install ./openmirror_…_amd64.deb`
- Fedora, openSUSE: `sudo dnf install ./openmirror-….x86_64.rpm`
- Anything else: the `.AppImage` — `chmod +x` it and run it. If it will not start, install `libfuse2`.

`SHA256SUMS.txt` has the checksum of every file here.
