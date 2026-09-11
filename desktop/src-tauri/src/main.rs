// openmirror's desktop shell.
//
// This deliberately contains no application logic. Everything openmirror does — the
// agent, the providers, voice, the browser, desktop control — lives in the
// Python daemon, and this is a window onto it plus the three things a browser
// tab genuinely cannot do: live in the tray, survive being closed, and put a
// notification in front of you when the agent is waiting on an answer.
//
// It also manages the daemon's lifetime, but only when it started it. If one
// is already listening — run by systemd, or by hand in a terminal — this
// attaches to that and leaves it alone on exit. Killing a daemon somebody else
// started would take their running sessions with it.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

/// The daemon, if this process started it. `None` means one was already there.
struct Daemon(Mutex<Option<Child>>);

fn base_url() -> String {
    let port = std::env::var("OPENMIRROR_PORT").unwrap_or_else(|_| "8477".into());
    format!("http://127.0.0.1:{port}")
}

/// Is something already answering on the port?
fn daemon_is_up(url: &str) -> bool {
    // A plain TCP connect rather than an HTTP request: it needs no dependency,
    // and "is the port open" is the only question being asked.
    let addr = url.trim_start_matches("http://");
    std::net::TcpStream::connect_timeout(
        &addr.parse().unwrap_or_else(|_| "127.0.0.1:8477".parse().unwrap()),
        Duration::from_millis(400),
    )
    .is_ok()
}

/// Start the daemon, preferring an installed `openmirror` over a source checkout.
fn spawn_daemon() -> Option<Child> {
    let candidates: Vec<(String, Vec<String>)> = vec![
        ("openmirror".into(), vec![]),
        (
            std::env::var("OPENMIRROR_PYTHON").unwrap_or_else(|_| "python3".into()),
            vec!["-m".into(), "openmirror.main".into()],
        ),
    ];

    for (program, args) in candidates {
        match Command::new(&program)
            .args(&args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => {
                eprintln!("openmirror: started daemon via {program}");
                return Some(child);
            }
            Err(err) => eprintln!("openmirror: could not start {program}: {err}"),
        }
    }
    None
}

fn wait_until_up(url: &str, limit: Duration) -> bool {
    let deadline = Instant::now() + limit;
    while Instant::now() < deadline {
        if daemon_is_up(url) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    false
}

fn main() {
    let url = base_url();

    // Attach if one is already running; only start one if not. This is what
    // lets the app coexist with a systemd-managed daemon.
    let owned = if daemon_is_up(&url) {
        eprintln!("openmirror: attaching to the daemon already on {url}");
        None
    } else {
        spawn_daemon()
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_opener::init())
        .manage(Daemon(Mutex::new(owned)))
        .setup(move |app| {
            let transparent = std::env::var("OPENMIRROR_TRANSPARENT")
                .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
                .unwrap_or(false);

            if !wait_until_up(&url, Duration::from_secs(30)) {
                eprintln!("openmirror: the daemon did not come up on {url}");
            }

            let window = WebviewWindowBuilder::new(
                app,
                "main",
                WebviewUrl::External(url.parse().expect("bad daemon url")),
            )
            .title("openmirror")
            .inner_size(1200.0, 820.0)
            .min_inner_size(720.0, 520.0)
            // Transparency lets the page's own glass blur the real desktop
            // rather than a background it painted itself. It is opt-in because
            // it depends entirely on the compositor: on COSMIC/Wayland a
            // transparent window did not map at all, which is indistinguishable
            // from the app failing to start. Off by default, so the window
            // always appears; on for anyone whose compositor handles it.
            .transparent(transparent)
            .decorations(true)
            .build()
            .inspect_err(|e| eprintln!("openmirror: window build failed: {e}"))?;
            let _ = window.show();
            if transparent {
                // Told, not sniffed: the page cannot see the window's own
                // transparency, and dropping its background without it would
                // leave the interface floating on nothing.
                let _ = window.eval("document.body.classList.add('native')");
            }

            // Closing the window hides it rather than quitting. An agent
            // mid-build should not be stopped by someone tidying their desktop;
            // Quit in the tray is the way out.
            let hidden = window.clone();
            window.on_window_event(move |event| {
                if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    let _ = hidden.hide();
                }
            });

            let show = MenuItem::with_id(app, "show", "Show openmirror", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show, &quit])?;

            TrayIconBuilder::new()
                .icon(app.default_window_icon().unwrap().clone())
                .tooltip("openmirror")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "show" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    // Left click raises the window, which is what every tray
                    // application does and what people expect without being told.
                    if let tauri::tray::TrayIconEvent::Click {
                        button: tauri::tray::MouseButton::Left,
                        button_state: tauri::tray::MouseButtonState::Up,
                        ..
                    } = event
                    {
                        if let Some(w) = tray.app_handle().get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                })
                .build(app)
                .inspect_err(|e| eprintln!("openmirror: tray build failed: {e}"))?;

            Ok(())
        })
        .on_window_event(|_window, _event| {})
        .build(tauri::generate_context!())
        .expect("failed to build the openmirror shell")
        .run(|app, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                // Only ours. A daemon this process did not start keeps running,
                // along with whatever sessions it is holding.
                if let Some(state) = app.try_state::<Daemon>() {
                    if let Ok(mut guard) = state.0.lock() {
                        if let Some(child) = guard.as_mut() {
                            let _ = child.kill();
                            let _ = child.wait();
                        }
                    }
                }
            }
        });
}
