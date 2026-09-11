"""Configuration, from the environment.

Everything is optional and nothing is a secret in code. A provider with no
key is simply not registered, which is why an install with only Perch
configured is a complete install rather than a broken one.

A `.env` beside the project is read first, if there is one. That is what the
README has always told people to do — `cp .env.example .env`, edit it, run —
and until this was here it did nothing at all: the file was written, the
daemon started with none of it, and the symptom was "no providers configured"
next to a file that plainly configures several. Real environment variables
still win, so a shell that exports something overrides the file rather than
being overridden by it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """Read a .env from the working directory, or wherever OPENMIRROR_ENV says.

    Deliberately not a dependency. python-dotenv arrives with uvicorn on most
    installs and is absent on some, and the format that matters here is
    KEY=value with optional quotes and # comments — twenty lines rather than a
    package that might not be there.
    """
    path = Path(os.getenv('OPENMIRROR_ENV') or '.env')
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as exc:
        log.warning('could not read %s: %s', path, exc)
        return

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip().removeprefix('export ').strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '\'"':
            value = value[1:-1]
        # The environment wins. Someone who exported a variable to override
        # the file for one run should get the override, not the file.
        os.environ.setdefault(key, value)


_load_dotenv()


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ('1', 'true', 'yes', 'on')


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, '') or default)
    except ValueError:
        return default


@dataclass(slots=True)
class Config:
    host: str = field(default_factory=lambda: os.getenv('OPENMIRROR_HOST', '127.0.0.1'))
    port: int = field(default_factory=lambda: _int('OPENMIRROR_PORT', 8477))

    # Exit when stdin closes. The desktop app sets this: it starts the daemon
    # with a pipe on stdin and holds the other end, so the daemon cannot
    # outlive the app that owns it. Not something to set by hand — a daemon
    # run from a terminal would stop at the first Ctrl-D.
    exit_with_stdin: bool = field(default_factory=lambda: _bool('OPENMIRROR_EXIT_WITH_STDIN'))

    # The directory an agent session may touch. Sessions are confined to it,
    # and the confinement is the only thing between a model and the rest of
    # the disk — so it defaults to the working directory rather than to $HOME.
    workspace: Path = field(default_factory=lambda: Path(os.getenv('OPENMIRROR_WORKSPACE', os.getcwd())).resolve())
    approval_mode: str = field(default_factory=lambda: os.getenv('OPENMIRROR_APPROVAL_MODE', 'ask'))

    # Give the agent the whole filesystem instead of confining it to the
    # workspace. A declared mode, not a default: the file tools honour the
    # root, and the shell cannot be confined at all — so with this off, the
    # shell at least escalates any command reaching outside it.
    unconfined: bool = field(default_factory=lambda: _bool('OPENMIRROR_UNCONFINED'))

    # Money and secrets are their own axis, deliberately separate from
    # `approval_mode`. Both are on: this is meant to be able to finish a task
    # that ends at a checkout. Neither is ever automatic — a purchase is
    # confirmed in every mode including `unrestricted`, and in a run nobody is
    # watching. Setting either to false makes it a refusal rather than a
    # prompt, for an install that should not be able to do it at all.
    allow_purchases: bool = field(default_factory=lambda: _bool('OPENMIRROR_ALLOW_PURCHASES', True))
    allow_credentials: bool = field(default_factory=lambda: _bool('OPENMIRROR_ALLOW_CREDENTIALS', True))

    # --- the web ----------------------------------------------------------
    web_enabled: bool = field(default_factory=lambda: _bool('OPENMIRROR_WEB', True))
    # Fetching loopback and link-local addresses would let a page it is reading
    # reach services never exposed to the internet — a cloud metadata endpoint,
    # or Perch's own console.
    web_allow_private: bool = field(default_factory=lambda: _bool('OPENMIRROR_WEB_ALLOW_PRIVATE'))
    # `headless` by default: it is the only backend that needs no key, and the
    # keyless HTTP one it replaced stopped working when the engines began
    # answering automated requests with a challenge page. It costs a Chromium
    # (the `browser` extra), which is why the keyed backends are still here and
    # still better where a key exists.
    search_backend: str = field(default_factory=lambda: os.getenv('OPENMIRROR_SEARCH_BACKEND', 'headless'))
    search_key: str = field(default_factory=lambda: os.getenv('OPENMIRROR_SEARCH_KEY', ''))
    search_url: str = field(default_factory=lambda: os.getenv('OPENMIRROR_SEARCH_URL', ''))
    # Which engine the headless backend asks. `auto` walks them in order —
    # least-tracking first — and stops at the first that answers, because an
    # engine being blocked is the ordinary case. Naming one (startpage,
    # duckduckgo, bing, google) uses only that one and fails loudly: somebody
    # who chose an engine for what it does not log has not agreed to fall
    # through to one that does. Ignored by every other backend.
    search_engine: str = field(default_factory=lambda: os.getenv('OPENMIRROR_SEARCH_ENGINE', 'auto'))

    # Undo for the agent's own file edits. On by default: it is cheap, and the
    # moment you want it is always after the fact.
    checkpoints_enabled: bool = field(default_factory=lambda: _bool('OPENMIRROR_CHECKPOINTS', True))

    # --- agents, skills and the code ---------------------------------------
    # Subagents: the `agent` tool, in the foreground and the background. On,
    # because every call a subagent makes is graded and asked about exactly
    # like the session's own; what one costs is tokens, not permission.
    agents_enabled: bool = field(default_factory=lambda: _bool('OPENMIRROR_AGENTS', True))
    # Skills: the bundled ones, this person's (~/.openmirror/skills and
    # ~/.claude/skills) and the project's own.
    skills_enabled: bool = field(default_factory=lambda: _bool('OPENMIRROR_SKILLS', True))
    # Language servers, for the `lsp` tool and for errors after an edit. Only
    # ever does anything where a server is installed; the tool is not offered
    # at all where none is.
    lsp_enabled: bool = field(default_factory=lambda: _bool('OPENMIRROR_LSP', True))
    lsp_config: Path = field(
        default_factory=lambda: Path(os.getenv('OPENMIRROR_LSP_CONFIG', '')) if os.getenv('OPENMIRROR_LSP_CONFIG')
        else Path.cwd() / '.lsp.json'
    )
    # When the conversation is summarised to make room, in estimated tokens.
    # Checked at the start of each request, and it only ever summarises what
    # came *before* the current turn, so the work in hand is never cut in half.
    # 0 turns it off; `/compact` in the composer still works either way.
    #
    # The default suits a hosted model with a large window. A local model's
    # real limit is its context length — Ollama's `num_ctx` — and this should
    # sit well under that, because Ollama does not refuse an overlong prompt:
    # it drops the beginning of it, silently, and the first thing to go is the
    # system prompt.
    compact_at: int = field(default_factory=lambda: _int('OPENMIRROR_COMPACT_AT', 100_000))

    # --- MCP --------------------------------------------------------------
    # Servers are read from a `.mcp.json` in the shape the rest of the
    # ecosystem uses, so a file written for another client works unchanged.
    mcp_config: Path = field(
        default_factory=lambda: Path(os.getenv('OPENMIRROR_MCP_CONFIG', '')) if os.getenv('OPENMIRROR_MCP_CONFIG')
        else Path.cwd() / '.mcp.json'
    )
    mcp_enabled: bool = field(default_factory=lambda: _bool('OPENMIRROR_MCP', True))

    # --- openmirror AS an MCP server ------------------------------------------
    # The other direction: another client — an editor, a desktop assistant,
    # another agent — using this twin's memory, routing and search backend.
    #
    # Off, and token-gated when on. Memory is the most personal thing in the
    # project, and the HTTP endpoint refuses to mount on a non-loopback bind
    # with no token rather than warning about it. The stdio transport
    # (`openmirror-mcp`) needs no token: a parent process that can spawn it
    # already has everything it could hand over.
    mcp_serve: bool = field(default_factory=lambda: _bool('OPENMIRROR_MCP_SERVE'))
    mcp_serve_token: str = field(default_factory=lambda: os.getenv('OPENMIRROR_MCP_SERVE_TOKEN', ''))
    # How much of the twin to lend out. `read` recalls, searches, fetches and
    # researches; `write` adds remembering; `all` adds media generation, which
    # spends money. Nothing on any scope runs a command, touches a file or
    # drives the screen — those are the tools openmirror guards with a human, and
    # an MCP call has no human in front of it.
    mcp_serve_scope: str = field(default_factory=lambda: os.getenv('OPENMIRROR_MCP_SERVE_SCOPE', 'read'))

    # --- the desktop ------------------------------------------------------
    # Screenshotting the screen and driving the mouse and keyboard. Off by
    # default: it is the widest capability here and the one with the weakest
    # safety net, since a click on a bitmap cannot be graded the way a click
    # on a labelled DOM element can.
    desktop_enabled: bool = field(default_factory=lambda: _bool('OPENMIRROR_DESKTOP'))

    # Where desktop control happens. `virtual` gives the agent a display of
    # its own, which is the only setting under which your mouse and keyboard
    # are genuinely untouched while it works — see openmirror/agent/stage.py.
    # `shared` drives the screen you are looking at. `auto` prefers virtual
    # and falls back to shared when no X server can be started.
    desktop_stage: str = field(default_factory=lambda: os.getenv('OPENMIRROR_DESKTOP_STAGE', 'auto'))
    # Which monitor it may touch, on a shared stage with more than one. An
    # index as X arranges them (1 is the first), or a name from xrandr.
    # Everything outside that rectangle is neither captured nor clickable.
    desktop_monitor: str = field(default_factory=lambda: os.getenv('OPENMIRROR_DESKTOP_MONITOR', ''))
    # The X display for a shared stage, when it is not $DISPLAY.
    desktop_display: str = field(default_factory=lambda: os.getenv('OPENMIRROR_DESKTOP_DISPLAY', ''))
    desktop_virtual_size: str = field(
        default_factory=lambda: os.getenv('OPENMIRROR_DESKTOP_VIRTUAL_SIZE', '1920x1080')
    )
    # On a shared stage, give the pointer back the moment a person moves it.
    # The agent's work is interrupted rather than fought over: a cursor being
    # dragged out from under someone is worse than a task that stopped.
    desktop_yield_to_user: bool = field(default_factory=lambda: _bool('OPENMIRROR_DESKTOP_YIELD', True))

    # Installing software and changing system settings. On, because a harness
    # that can drive a browser but cannot install the program someone asked
    # for is oddly shaped — and every one of these calls is graded and
    # confirmed like any other. A system-wide install grades `execute`,
    # because a package's install scripts run as root; a per-user one grades
    # `network`, because that is what it is.
    system_tools_enabled: bool = field(default_factory=lambda: _bool('OPENMIRROR_SYSTEM', True))

    # --- generated media --------------------------------------------------
    media_dir: Path = field(
        default_factory=lambda: Path(os.getenv('OPENMIRROR_MEDIA_DIR', ''))
        if os.getenv('OPENMIRROR_MEDIA_DIR')
        else Path(os.getenv('OPENMIRROR_DATA_DIR', './data')) / 'media'
    )
    # What a generation may cost before it is refused, in seconds. Video is
    # minutes rather than seconds, and a default HTTP timeout ends jobs that
    # were going to succeed.
    image_timeout: int = field(default_factory=lambda: _int('OPENMIRROR_IMAGE_TIMEOUT', 600))
    video_timeout: int = field(default_factory=lambda: _int('OPENMIRROR_VIDEO_TIMEOUT', 1800))

    # --- the realtime voice API -------------------------------------------
    # OpenAI's speech-to-speech endpoint, which is its own protocol rather
    # than a modality the registry can route: one socket carries audio, text,
    # interruption and tool calls together.
    realtime_model: str = field(
        default_factory=lambda: os.getenv('OPENMIRROR_REALTIME_MODEL', 'gpt-realtime')
    )
    realtime_voice: str = field(default_factory=lambda: os.getenv('OPENMIRROR_REALTIME_VOICE', 'marin'))
    realtime_url: str = field(
        default_factory=lambda: os.getenv('OPENMIRROR_REALTIME_URL', 'wss://api.openai.com/v1/realtime')
    )

    # --- the browser ------------------------------------------------------
    browser_enabled: bool = field(default_factory=lambda: _bool('OPENMIRROR_BROWSER'))
    # A real profile, logged into by hand once. The agent inherits the sessions
    # and never needs a password.
    browser_profile: Path = field(
        default_factory=lambda: Path(os.getenv('OPENMIRROR_BROWSER_PROFILE', str(Path.home() / '.openmirror' / 'browser')))
    )
    # Headful is worth it for anything transactional: watching it, and being
    # able to take the mouse off it, beats the memory it costs.
    browser_headless: bool = field(default_factory=lambda: _bool('OPENMIRROR_BROWSER_HEADLESS', True))

    # WHICH browser. These three are the *default*; a choice saved from the
    # settings dialog lives in `browser.json` under the data directory and
    # overrides them, so an install managed by a file behaves as it always did
    # and a person who wants Chrome this afternoon does not have to edit one.
    # See openmirror/agent/browsers.py.
    #
    # `browser_engine` is a Playwright engine: chromium, firefox or webkit.
    # `browser_channel` names an installed variant (chrome, msedge, …) and is
    # chromium-only. `browser_executable` points at a binary — a Brave or an
    # ungoogled-chromium — and wins over a channel.
    browser_engine: str = field(default_factory=lambda: os.getenv('OPENMIRROR_BROWSER_ENGINE', 'chromium'))
    browser_channel: str = field(default_factory=lambda: os.getenv('OPENMIRROR_BROWSER_CHANNEL', ''))
    browser_executable: str = field(default_factory=lambda: os.getenv('OPENMIRROR_BROWSER_BINARY', ''))

    # Bound to loopback by default. An agent that runs commands must not be on
    # a network interface by accident.
    auth_token: str = field(default_factory=lambda: os.getenv('OPENMIRROR_TOKEN', ''))

    anthropic_key: str = field(default_factory=lambda: os.getenv('ANTHROPIC_API_KEY', ''))
    anthropic_url: str = field(default_factory=lambda: os.getenv('ANTHROPIC_BASE_URL', 'https://api.anthropic.com/v1'))

    openai_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    openai_url: str = field(default_factory=lambda: os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'))

    ollama_url: str = field(default_factory=lambda: os.getenv('OLLAMA_BASE_URL', ''))

    # One host, one token: the five services are derived from it.
    perch_host: str = field(default_factory=lambda: os.getenv('PERCH_HOST', ''))
    perch_token: str = field(default_factory=lambda: os.getenv('PERCH_TOKEN', ''))
    perch_scheme: str = field(default_factory=lambda: os.getenv('PERCH_SCHEME', 'http'))

    # Open WebUI, which is itself a provider: it speaks OpenAI's and
    # Anthropic's shapes, so pointing at it gives openmirror every connection
    # configured over there without configuring any of them here.
    openwebui_url: str = field(default_factory=lambda: os.getenv('OPENWEBUI_BASE_URL', ''))
    openwebui_key: str = field(default_factory=lambda: os.getenv('OPENWEBUI_API_KEY', ''))

    # Which connection answers, per modality. Chat and embedding are named
    # separately and deliberately: the model you think with and the model you
    # remember with are different choices, they have different privacy
    # consequences, and tying them together is how someone ends up sending
    # their whole memory to a vendor they only wanted for chat.
    chat_provider: str = field(default_factory=lambda: os.getenv('OPENMIRROR_CHAT_PROVIDER', ''))
    embed_provider: str = field(default_factory=lambda: os.getenv('OPENMIRROR_EMBED_PROVIDER', ''))
    # Ask a truncatable embedding model for shorter vectors. Cannot be changed
    # under an existing store — vectors of different lengths are not comparable.
    embed_dimensions: int = field(default_factory=lambda: _int('OPENMIRROR_EMBED_DIMENSIONS', 0))

    default_chat_model: str = field(default_factory=lambda: os.getenv('OPENMIRROR_CHAT_MODEL', ''))
    default_stt_model: str = field(default_factory=lambda: os.getenv('OPENMIRROR_STT_MODEL', 'whisper-1'))
    default_tts_model: str = field(default_factory=lambda: os.getenv('OPENMIRROR_TTS_MODEL', 'tts-1'))
    default_tts_voice: str = field(default_factory=lambda: os.getenv('OPENMIRROR_TTS_VOICE', 'alloy'))

    # Refuse any provider that is not local hardware. Off by default because
    # it makes an install with no GPU do nothing at all; on, it is a
    # guarantee rather than a preference.
    local_only: bool = field(default_factory=lambda: _bool('OPENMIRROR_LOCAL_ONLY'))

    # Per-user vector memory. Two switches, and both must be on: this one
    # decides whether the feature exists on this install at all, and each
    # person then decides for themselves. Off here means the store is never
    # even opened.
    memory_enabled: bool = field(default_factory=lambda: _bool('OPENMIRROR_MEMORY'))
    # Where connections, checkpoints, memory and generated media live.
    data_dir: Path = field(default_factory=lambda: Path(os.getenv('OPENMIRROR_DATA_DIR', './data')))
    connections_db: Path = field(
        default_factory=lambda: Path(os.getenv('OPENMIRROR_CONNECTIONS', ''))
        if os.getenv('OPENMIRROR_CONNECTIONS')
        else Path(os.getenv('OPENMIRROR_DATA_DIR', './data')) / 'connections.json'
    )
    memory_db: Path = field(
        default_factory=lambda: Path(os.getenv('OPENMIRROR_MEMORY_DB', '')) if os.getenv('OPENMIRROR_MEMORY_DB')
        else Path(os.getenv('OPENMIRROR_DATA_DIR', './data')) / 'memory.db'
    )
    embed_model: str = field(default_factory=lambda: os.getenv('OPENMIRROR_EMBED_MODEL', ''))

    # Until openmirror has accounts of its own, everything belongs to one person.
    # Named rather than blank so that adding accounts later is a migration
    # rather than a redesign: the store has always been per-user.
    default_user: str = field(default_factory=lambda: os.getenv('OPENMIRROR_USER', 'local'))

    log_level: str = field(default_factory=lambda: os.getenv('OPENMIRROR_LOG_LEVEL', 'INFO'))


config = Config()
