"""Configuration, from the environment.

Everything is optional and nothing is a secret in code. A provider with no
key is simply not registered, which is why an install with only Perch
configured is a complete install rather than a broken one.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ('1', 'true', 'yes', 'on')


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, '') or default)
    except ValueError:
        return default


@dataclass(slots=True)
class Config:
    host: str = field(default_factory=lambda: os.getenv('ROOST_HOST', '127.0.0.1'))
    port: int = field(default_factory=lambda: _int('ROOST_PORT', 8477))

    # The directory an agent session may touch. Sessions are confined to it,
    # and the confinement is the only thing between a model and the rest of
    # the disk — so it defaults to the working directory rather than to $HOME.
    workspace: Path = field(default_factory=lambda: Path(os.getenv('ROOST_WORKSPACE', os.getcwd())).resolve())
    approval_mode: str = field(default_factory=lambda: os.getenv('ROOST_APPROVAL_MODE', 'ask'))

    # Give the agent the whole filesystem instead of confining it to the
    # workspace. A declared mode, not a default: the file tools honour the
    # root, and the shell cannot be confined at all — so with this off, the
    # shell at least escalates any command reaching outside it.
    unconfined: bool = field(default_factory=lambda: _bool('ROOST_UNCONFINED'))

    # Money and secrets are their own axis, deliberately separate from
    # `approval_mode`. `unrestricted` means "stop asking me about this
    # machine"; it does not and should not also mean "spend my money".
    allow_purchases: bool = field(default_factory=lambda: _bool('ROOST_ALLOW_PURCHASES'))
    # Worth leaving off: with it off the browser hands password and payment
    # fields to the person, so the agent never has the value in its context at
    # all. Approving it to type a password is strictly worse than that.
    allow_credentials: bool = field(default_factory=lambda: _bool('ROOST_ALLOW_CREDENTIALS'))

    # --- the web ----------------------------------------------------------
    web_enabled: bool = field(default_factory=lambda: _bool('ROOST_WEB', True))
    # Fetching loopback and link-local addresses would let a page it is reading
    # reach services never exposed to the internet — a cloud metadata endpoint,
    # or Perch's own console.
    web_allow_private: bool = field(default_factory=lambda: _bool('ROOST_WEB_ALLOW_PRIVATE'))
    search_backend: str = field(default_factory=lambda: os.getenv('ROOST_SEARCH_BACKEND', 'duckduckgo'))
    search_key: str = field(default_factory=lambda: os.getenv('ROOST_SEARCH_KEY', ''))
    search_url: str = field(default_factory=lambda: os.getenv('ROOST_SEARCH_URL', ''))

    # Undo for the agent's own file edits. On by default: it is cheap, and the
    # moment you want it is always after the fact.
    checkpoints_enabled: bool = field(default_factory=lambda: _bool('ROOST_CHECKPOINTS', True))

    # --- MCP --------------------------------------------------------------
    # Servers are read from a `.mcp.json` in the shape the rest of the
    # ecosystem uses, so a file written for another client works unchanged.
    mcp_config: Path = field(
        default_factory=lambda: Path(os.getenv('ROOST_MCP_CONFIG', '')) if os.getenv('ROOST_MCP_CONFIG')
        else Path.cwd() / '.mcp.json'
    )
    mcp_enabled: bool = field(default_factory=lambda: _bool('ROOST_MCP', True))

    # --- the desktop ------------------------------------------------------
    # Screenshotting the screen and driving the mouse and keyboard. Off by
    # default: it is the widest capability here and the one with the weakest
    # safety net, since a click on a bitmap cannot be graded the way a click
    # on a labelled DOM element can.
    desktop_enabled: bool = field(default_factory=lambda: _bool('ROOST_DESKTOP'))

    # Where desktop control happens. `virtual` gives the agent a display of
    # its own, which is the only setting under which your mouse and keyboard
    # are genuinely untouched while it works — see roost/agent/stage.py.
    # `shared` drives the screen you are looking at. `auto` prefers virtual
    # and falls back to shared when no X server can be started.
    desktop_stage: str = field(default_factory=lambda: os.getenv('ROOST_DESKTOP_STAGE', 'auto'))
    # Which monitor it may touch, on a shared stage with more than one. An
    # index as X arranges them (1 is the first), or a name from xrandr.
    # Everything outside that rectangle is neither captured nor clickable.
    desktop_monitor: str = field(default_factory=lambda: os.getenv('ROOST_DESKTOP_MONITOR', ''))
    # The X display for a shared stage, when it is not $DISPLAY.
    desktop_display: str = field(default_factory=lambda: os.getenv('ROOST_DESKTOP_DISPLAY', ''))
    desktop_virtual_size: str = field(
        default_factory=lambda: os.getenv('ROOST_DESKTOP_VIRTUAL_SIZE', '1920x1080')
    )
    # On a shared stage, give the pointer back the moment a person moves it.
    # The agent's work is interrupted rather than fought over: a cursor being
    # dragged out from under someone is worse than a task that stopped.
    desktop_yield_to_user: bool = field(default_factory=lambda: _bool('ROOST_DESKTOP_YIELD', True))

    # --- generated media --------------------------------------------------
    media_dir: Path = field(
        default_factory=lambda: Path(os.getenv('ROOST_MEDIA_DIR', ''))
        if os.getenv('ROOST_MEDIA_DIR')
        else Path(os.getenv('ROOST_DATA_DIR', './data')) / 'media'
    )
    # What a generation may cost before it is refused, in seconds. Video is
    # minutes rather than seconds, and a default HTTP timeout ends jobs that
    # were going to succeed.
    image_timeout: int = field(default_factory=lambda: _int('ROOST_IMAGE_TIMEOUT', 600))
    video_timeout: int = field(default_factory=lambda: _int('ROOST_VIDEO_TIMEOUT', 1800))

    # --- the realtime voice API -------------------------------------------
    # OpenAI's speech-to-speech endpoint, which is its own protocol rather
    # than a modality the registry can route: one socket carries audio, text,
    # interruption and tool calls together.
    realtime_model: str = field(
        default_factory=lambda: os.getenv('ROOST_REALTIME_MODEL', 'gpt-realtime')
    )
    realtime_voice: str = field(default_factory=lambda: os.getenv('ROOST_REALTIME_VOICE', 'marin'))
    realtime_url: str = field(
        default_factory=lambda: os.getenv('ROOST_REALTIME_URL', 'wss://api.openai.com/v1/realtime')
    )

    # --- the browser ------------------------------------------------------
    browser_enabled: bool = field(default_factory=lambda: _bool('ROOST_BROWSER'))
    # A real profile, logged into by hand once. The agent inherits the sessions
    # and never needs a password.
    browser_profile: Path = field(
        default_factory=lambda: Path(os.getenv('ROOST_BROWSER_PROFILE', str(Path.home() / '.roost' / 'browser')))
    )
    # Headful is worth it for anything transactional: watching it, and being
    # able to take the mouse off it, beats the memory it costs.
    browser_headless: bool = field(default_factory=lambda: _bool('ROOST_BROWSER_HEADLESS', True))

    # Bound to loopback by default. An agent that runs commands must not be on
    # a network interface by accident.
    auth_token: str = field(default_factory=lambda: os.getenv('ROOST_TOKEN', ''))

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
    # Anthropic's shapes, so pointing at it gives Roost every connection
    # configured over there without configuring any of them here.
    openwebui_url: str = field(default_factory=lambda: os.getenv('OPENWEBUI_BASE_URL', ''))
    openwebui_key: str = field(default_factory=lambda: os.getenv('OPENWEBUI_API_KEY', ''))

    # Which connection answers, per modality. Chat and embedding are named
    # separately and deliberately: the model you think with and the model you
    # remember with are different choices, they have different privacy
    # consequences, and tying them together is how someone ends up sending
    # their whole memory to a vendor they only wanted for chat.
    chat_provider: str = field(default_factory=lambda: os.getenv('ROOST_CHAT_PROVIDER', ''))
    embed_provider: str = field(default_factory=lambda: os.getenv('ROOST_EMBED_PROVIDER', ''))
    # Ask a truncatable embedding model for shorter vectors. Cannot be changed
    # under an existing store — vectors of different lengths are not comparable.
    embed_dimensions: int = field(default_factory=lambda: _int('ROOST_EMBED_DIMENSIONS', 0))

    default_chat_model: str = field(default_factory=lambda: os.getenv('ROOST_CHAT_MODEL', ''))
    default_stt_model: str = field(default_factory=lambda: os.getenv('ROOST_STT_MODEL', 'whisper-1'))
    default_tts_model: str = field(default_factory=lambda: os.getenv('ROOST_TTS_MODEL', 'tts-1'))
    default_tts_voice: str = field(default_factory=lambda: os.getenv('ROOST_TTS_VOICE', 'alloy'))

    # Refuse any provider that is not local hardware. Off by default because
    # it makes an install with no GPU do nothing at all; on, it is a
    # guarantee rather than a preference.
    local_only: bool = field(default_factory=lambda: _bool('ROOST_LOCAL_ONLY'))

    # Per-user vector memory. Two switches, and both must be on: this one
    # decides whether the feature exists on this install at all, and each
    # person then decides for themselves. Off here means the store is never
    # even opened.
    memory_enabled: bool = field(default_factory=lambda: _bool('ROOST_MEMORY'))
    # Where connections, checkpoints, memory and generated media live.
    data_dir: Path = field(default_factory=lambda: Path(os.getenv('ROOST_DATA_DIR', './data')))
    connections_db: Path = field(
        default_factory=lambda: Path(os.getenv('ROOST_CONNECTIONS', ''))
        if os.getenv('ROOST_CONNECTIONS')
        else Path(os.getenv('ROOST_DATA_DIR', './data')) / 'connections.json'
    )
    memory_db: Path = field(
        default_factory=lambda: Path(os.getenv('ROOST_MEMORY_DB', '')) if os.getenv('ROOST_MEMORY_DB')
        else Path(os.getenv('ROOST_DATA_DIR', './data')) / 'memory.db'
    )
    embed_model: str = field(default_factory=lambda: os.getenv('ROOST_EMBED_MODEL', ''))

    # Until Roost has accounts of its own, everything belongs to one person.
    # Named rather than blank so that adding accounts later is a migration
    # rather than a redesign: the store has always been per-user.
    default_user: str = field(default_factory=lambda: os.getenv('ROOST_USER', 'local'))

    log_level: str = field(default_factory=lambda: os.getenv('ROOST_LOG_LEVEL', 'INFO'))


config = Config()
