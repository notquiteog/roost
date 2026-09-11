//! The daemon: finding one already running, starting one when there is not,
//! and stopping only the one this app started.
//!
//! Nothing here knows about windows. `main.rs` decides what to show; this
//! decides what runs.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// The daemon's own default, from `openmirror/config.py`.
pub const DEFAULT_PORT: u16 = 8477;

/// `~/.openmirror`: where the daemon already keeps its browser profile, and
/// where a daemon this app starts keeps everything else. Not the directory
/// the app was launched from, which is where the daemon's own defaults point
/// — and from a Dock or a Start menu that is `/` or the install directory.
pub struct Home {
    /// The user's home directory. The daemon runs there, as it would when
    /// started from a new terminal, so its workspace defaults to it too.
    pub user: PathBuf,
    pub root: PathBuf,
}

impl Home {
    pub fn new(user: PathBuf) -> Self {
        let root = user.join(".openmirror");
        Self { user, root }
    }

    pub fn logs(&self) -> PathBuf {
        self.root.join("logs")
    }
}

/// The settings a daemon started here will see, read the way the daemon reads
/// them: the environment first, then the `.env` file. The app needs one of
/// them itself — the port, since an app looking on 8477 for a daemon that a
/// `.env` moved to 8500 would find nothing and start a second one — and must
/// not paper over the rest with defaults of its own.
pub struct Settings {
    pub env_file: PathBuf,
    file: HashMap<String, String>,
}

impl Settings {
    pub fn load(home: &Home) -> Self {
        // Relative to where the daemon runs, which is the home directory.
        let env_file = match std::env::var_os("OPENMIRROR_ENV") {
            Some(path) if !path.is_empty() => home.user.join(path),
            _ => home.root.join(".env"),
        };
        let file = fs::read(&env_file)
            .map(|bytes| parse_env(&String::from_utf8_lossy(&bytes)))
            .unwrap_or_default();
        Self { env_file, file }
    }

    /// Anything set in the environment wins, even when it is set to nothing.
    pub fn get(&self, key: &str) -> Option<String> {
        std::env::var(key).ok().or_else(|| self.file.get(key).cloned())
    }

    pub fn port(&self) -> u16 {
        self.get("OPENMIRROR_PORT")
            .and_then(|port| port.trim().parse().ok())
            .unwrap_or(DEFAULT_PORT)
    }
}

/// `KEY=value` lines, as `openmirror/config.py` reads them: `#` comments, an
/// optional `export `, optional matching quotes, and the first of a repeated
/// key winning — the daemon only ever sets a key that is not set yet.
pub fn parse_env(text: &str) -> HashMap<String, String> {
    let mut found = HashMap::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let key = key.strip_prefix("export ").unwrap_or(key).trim();
        let mut value = value.trim();
        let bytes = value.as_bytes();
        if bytes.len() >= 2 && bytes[0] == bytes[bytes.len() - 1] && matches!(bytes[0], b'"' | b'\'') {
            value = &value[1..value.len() - 1];
        }
        found.entry(key.to_string()).or_insert_with(|| value.to_string());
    }
    found
}

/// What answers on a port.
#[derive(Debug, PartialEq, Eq)]
pub enum Probe {
    Nothing,
    /// An openmirror daemon that has finished starting: it only listens once
    /// start-up is done, so answering at all means ready.
    Openmirror,
    /// Something else, or something that would not say what it is.
    Other(String),
}

pub fn listening(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(500)).is_ok()
}

pub fn probe(port: u16) -> Probe {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(500)) else {
        return Probe::Nothing;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(3)));
    // HTTP/1.0 by hand. One request, and the server closes once it has
    // answered, so reading to the end is reading the whole response.
    if stream
        .write_all(b"GET /healthz HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
        .is_err()
    {
        return Probe::Other("it took the connection and then refused the request".into());
    }
    let mut response = Vec::new();
    let _ = stream.take(256 * 1024).read_to_end(&mut response);
    classify(&String::from_utf8_lossy(&response))
}

/// Whether a response to `GET /healthz` came from openmirror.
pub fn classify(response: &str) -> Probe {
    let status = response.lines().next().unwrap_or("").trim();
    if !status.starts_with("HTTP/") {
        return Probe::Other(if response.is_empty() {
            "it took the connection and said nothing".into()
        } else {
            "it does not speak HTTP".into()
        });
    }
    let ok = status.split_whitespace().nth(1) == Some("200");
    let ours = response.contains("\"service\":\"openmirror\"") || response.contains("\"service\": \"openmirror\"");
    if ok && ours {
        Probe::Openmirror
    } else {
        Probe::Other(format!("it answered /healthz with \"{status}\""))
    }
}

/// A daemon this app started, and the pipe it watches.
pub struct Daemon {
    child: Child,
    /// Never written to. The daemon runs with OPENMIRROR_EXIT_WITH_STDIN and
    /// shuts itself down when this closes — which happens however this app
    /// ends, crash and kill included, because the operating system closes it.
    lifeline: Option<ChildStdin>,
    pub program: String,
    /// What happened to its PATH, on macOS: the line to look for when a
    /// report says the agent could not find `brew`.
    pub path_note: Option<String>,
}

impl Daemon {
    /// Start one, with its output going to `log`.
    pub fn spawn(home: &Home, settings: &Settings, log: &Path) -> Result<Self, String> {
        let logs = home.logs();
        fs::create_dir_all(&logs).map_err(|e| format!("Could not create {}: {e}", logs.display()))?;
        let out = open_log(log).map_err(|e| format!("Could not open {}: {e}", log.display()))?;
        let clone = || out.try_clone().map_err(|e| format!("Could not share {}: {e}", log.display()));
        let (path, path_note) = login_path();

        let mut tried = Vec::new();
        for (program, args) in candidates() {
            let mut cmd = Command::new(&program);
            cmd.args(&args)
                .current_dir(&home.user)
                .env("OPENMIRROR_EXIT_WITH_STDIN", "1")
                // Always the file the port was read from. The daemon's own
                // default is a .env in its working directory — here the home
                // directory, where a .env belongs to whatever else put it there.
                .env("OPENMIRROR_ENV", &settings.env_file)
                .stdin(Stdio::piped())
                .stdout(clone()?)
                .stderr(clone()?);
            if settings.get("OPENMIRROR_DATA_DIR").is_none() {
                cmd.env("OPENMIRROR_DATA_DIR", home.root.join("data"));
            }
            if let Some(path) = &path {
                cmd.env("PATH", path);
            }
            outside_appimage(&mut cmd);
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                // It is a console program, so that it has a stdout to log;
                // started from a windowed app it would otherwise open a
                // console window of its own.
                const CREATE_NO_WINDOW: u32 = 0x0800_0000;
                cmd.creation_flags(CREATE_NO_WINDOW);
            }

            match cmd.spawn() {
                Ok(mut child) => {
                    let lifeline = child.stdin.take();
                    let program = program.display().to_string();
                    return Ok(Self { child, lifeline, program, path_note });
                }
                Err(e) => tried.push(format!("{}: {e}", program.display())),
            }
        }
        Err(format!("There was nothing to start it with. Tried {}.", tried.join("; ")))
    }

    pub fn exited(&mut self) -> Option<ExitStatus> {
        self.child.try_wait().ok().flatten()
    }

    /// Close the pipe, give the daemon `grace` to shut down the way it would
    /// for Ctrl-C — closing its browser, its MCP servers — and kill it only
    /// if it has not. Says which happened, for the log.
    pub fn stop(mut self, grace: Duration) -> String {
        drop(self.lifeline.take());
        let deadline = Instant::now() + grace;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => return format!("stopped ({status})"),
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(100)),
                _ => break,
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        format!("had not stopped after {}s, and was killed", grace.as_secs())
    }
}

/// How to start a daemon, best first.
fn candidates() -> Vec<(PathBuf, Vec<String>)> {
    let mut found = Vec::new();
    // The one this release ships with. Tauri puts a sidecar beside the app's
    // own executable on every platform, the AppImage included, and it is the
    // daemon that matches this app's version.
    let sidecar = if cfg!(windows) { "openmirror-daemon.exe" } else { "openmirror-daemon" };
    if let Some(dir) = std::env::current_exe().ok().and_then(|exe| exe.parent().map(Path::to_path_buf)) {
        if dir.join(sidecar).is_file() {
            found.push((dir.join(sidecar), vec![]));
        }
    }
    // Then an installed `openmirror`, or a source checkout: what a build with
    // no sidecar, like `cargo run`, has to go on.
    found.push((PathBuf::from("openmirror"), vec![]));
    let python = std::env::var("OPENMIRROR_PYTHON").unwrap_or_else(|_| "python3".into());
    found.push((PathBuf::from(python), vec!["-m".into(), "openmirror.main".into()]));
    found
}

/// This run's log, with the last run's kept beside it as `.1`: enough to see
/// what happened just before a crash without the file growing forever.
pub fn open_log(path: &Path) -> io::Result<File> {
    if path.exists() {
        let mut previous = path.as_os_str().to_owned();
        previous.push(".1");
        let _ = fs::rename(path, previous);
    }
    OpenOptions::new().create(true).append(true).open(path)
}

/// The last `lines` of a log: where the reason for a failure usually is.
pub fn tail(path: &Path, lines: usize) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let text = String::from_utf8_lossy(&bytes);
    let all: Vec<&str> = text.lines().collect();
    let tail = all[all.len().saturating_sub(lines)..].join("\n");
    (!tail.trim().is_empty()).then_some(tail)
}

/// An AppImage runs with variables pointing into its own mount — GTK modules,
/// data directories — for its own sake, and the daemon would pass them on to
/// every program the agent runs. Those are the user's programs, which want
/// the user's environment; and the mount is gone the moment the app quits.
fn outside_appimage(cmd: &mut Command) {
    let Some(appdir) = std::env::var("APPDIR").ok().filter(|dir| !dir.is_empty()) else {
        return;
    };
    if std::env::var_os("APPIMAGE").is_none() {
        return;
    }
    for key in ["APPDIR", "APPIMAGE", "ARGV0", "OWD"] {
        cmd.env_remove(key);
    }
    for (key, value) in std::env::vars_os() {
        let Some(value) = value.to_str() else { continue };
        if !value.contains(&appdir) {
            continue;
        }
        match without(value, &appdir) {
            Some(kept) => cmd.env(&key, kept),
            None => cmd.env_remove(&key),
        };
    }
}

/// A path list with the entries inside `dir` taken out, or nothing if that
/// was all of it.
fn without(value: &str, dir: &str) -> Option<String> {
    let kept: Vec<&str> = value.split(':').filter(|part| !part.is_empty() && !part.starts_with(dir)).collect();
    (!kept.is_empty()).then(|| kept.join(":"))
}

/// On macOS, the PATH a terminal would have. An app opened from the Finder or
/// the Dock is started by launchd with only the system directories on its
/// PATH — no Homebrew, no ~/.local/bin, no ~/.cargo/bin — and the daemon hands
/// that to every command the agent runs, so `brew`, `node` and `uv` would be
/// "not found" in a release and there from a terminal. So ask the login
/// shell, as a terminal does, and put what it says first.
///
/// Returns the PATH to give the daemon, if it differs from this app's, and a
/// line for the log saying which happened.
#[cfg(target_os = "macos")]
fn login_path() -> (Option<String>, Option<String>) {
    let shell = std::env::var("SHELL").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "/bin/zsh".into());
    let current = std::env::var("PATH").unwrap_or_default();
    match ask_shell(&shell) {
        Some(found) => {
            let merged = merge_paths(&found, &current);
            let note = format!("gave the daemon the PATH {shell} sets up: {merged}");
            (Some(merged), Some(note))
        }
        None => {
            let note = format!("could not read the PATH {shell} sets up, so the daemon has the app's own: {current}");
            (None, Some(note))
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn login_path() -> (Option<String>, Option<String>) {
    (None, None)
}

#[cfg(any(target_os = "macos", test))]
const MARK: &str = "__OPENMIRROR_PATH__";

/// The PATH an interactive login shell ends up with. Marked, because an
/// interactive shell may print a banner of its own around it.
#[cfg(any(target_os = "macos", test))]
fn ask_shell(shell: &str) -> Option<String> {
    let script = format!("printf %s {MARK}; /usr/bin/printenv PATH; printf %s {MARK}");
    let mut child = Command::new(shell)
        .args(["-l", "-i", "-c", &script])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    // Bounded: a profile that waits on the network, or asks a question, must
    // not hold the daemon's start hostage. Without it, the PATH is launchd's.
    let deadline = Instant::now() + Duration::from_secs(5);
    while child.try_wait().ok()?.is_none() {
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
        thread::sleep(Duration::from_millis(50));
    }
    let mut out = String::new();
    child.stdout.take()?.read_to_string(&mut out).ok()?;
    between_marks(&out).map(str::to_string)
}

#[cfg(any(target_os = "macos", test))]
fn between_marks(out: &str) -> Option<&str> {
    let start = out.find(MARK)? + MARK.len();
    let len = out[start..].find(MARK)?;
    let path = out[start..start + len].trim();
    (!path.is_empty()).then_some(path)
}

/// `first`'s entries, then any of `then`'s it did not have, in order.
#[cfg(any(target_os = "macos", test))]
fn merge_paths(first: &str, then: &str) -> String {
    let mut seen: Vec<&str> = Vec::new();
    for dir in first.split(':').chain(then.split(':')) {
        if !dir.is_empty() && !seen.contains(&dir) {
            seen.push(dir);
        }
    }
    seen.join(":")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn env_files_are_read_the_way_the_daemon_reads_them() {
        let found = parse_env(
            "# a comment\n\
             OPENMIRROR_PORT=8500\n\
             export OPENMIRROR_DATA_DIR = \"/srv/openmirror\"\n\
             QUOTED='single'\n\
             not a setting\n\
             OPENMIRROR_PORT=9999\n\
             EMPTY=\n",
        );
        assert_eq!(found["OPENMIRROR_PORT"], "8500", "the first of a repeated key wins");
        assert_eq!(found["OPENMIRROR_DATA_DIR"], "/srv/openmirror");
        assert_eq!(found["QUOTED"], "single");
        assert_eq!(found["EMPTY"], "");
        assert_eq!(found.len(), 4);
    }

    #[test]
    fn healthz_from_openmirror_is_told_apart_from_anything_else() {
        let ours = "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{\"ok\":true,\"service\":\"openmirror\",\"providers\":[]}";
        assert_eq!(classify(ours), Probe::Openmirror);

        let theirs = "HTTP/1.1 200 OK\r\n\r\n{\"ok\":true,\"service\":\"something-else\"}";
        assert!(matches!(classify(theirs), Probe::Other(_)));
        let missing = "HTTP/1.1 404 Not Found\r\n\r\n{\"service\":\"openmirror\"}";
        assert!(matches!(classify(missing), Probe::Other(_)));
        assert!(matches!(classify(""), Probe::Other(_)));
        assert!(matches!(classify("SSH-2.0-OpenSSH_9.6\r\n"), Probe::Other(_)));
    }

    #[test]
    fn a_port_with_nothing_on_it_is_nothing_and_one_with_a_stranger_is_other() {
        let stranger = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = stranger.local_addr().unwrap().port();
        let answer = thread::spawn(move || {
            let (mut conn, _) = stranger.accept().unwrap();
            let mut buf = [0u8; 512];
            let _ = conn.read(&mut buf);
            let _ = conn.write_all(b"HTTP/1.0 200 OK\r\n\r\nhello");
        });
        assert!(matches!(probe(port), Probe::Other(_)));
        answer.join().unwrap();
        // The listener is gone with the thread, so the port empties — once
        // every process holding a copy has let go. A test spawning a process
        // at the same moment briefly holds one, until its child execs: in
        // parallel runs that failed three times in six before this waited.
        let deadline = Instant::now() + Duration::from_secs(3);
        while probe(port) != Probe::Nothing && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(50));
        }
        assert_eq!(probe(port), Probe::Nothing);
    }

    #[test]
    fn a_log_keeps_the_previous_run() {
        let dir = std::env::temp_dir().join(format!("openmirror-log-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("daemon.log");
        writeln!(open_log(&path).unwrap(), "first run").unwrap();
        writeln!(open_log(&path).unwrap(), "second run").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "second run\n");
        assert_eq!(fs::read_to_string(dir.join("daemon.log.1")).unwrap(), "first run\n");
        assert_eq!(tail(&path, 5).as_deref(), Some("second run"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn appimage_paths_are_taken_out_and_the_rest_kept() {
        let dir = "/tmp/.mount_openmiAbC";
        assert_eq!(
            without("/tmp/.mount_openmiAbC/usr/share:/usr/local/share:/usr/share", dir).as_deref(),
            Some("/usr/local/share:/usr/share")
        );
        assert_eq!(without("/tmp/.mount_openmiAbC/usr/lib/gtk-3.0", dir), None);
    }

    #[test]
    fn the_login_path_comes_first_and_nothing_is_listed_twice() {
        assert_eq!(
            merge_paths("/opt/homebrew/bin:/usr/bin:/bin", "/usr/bin:/bin:/usr/sbin"),
            "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin"
        );
        assert_eq!(between_marks(&format!("Welcome!\n{MARK}/a:/b\n{MARK}")), Some("/a:/b"));
        assert_eq!(between_marks("no marks here"), None);
    }

    #[cfg(unix)]
    #[test]
    fn a_real_login_shell_reports_its_path() {
        let path = ask_shell("/bin/sh").expect("sh -l -i -c should answer");
        assert!(path.split(':').any(|dir| dir == "/usr/bin" || dir == "/bin"), "{path}");
    }
}
