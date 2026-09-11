"""Running a set of MCP servers and offering their tools to the agent.

The interesting decision here is whether to believe a server about its own
tools. MCP lets a tool declare `readOnlyHint: true`, and taking that at face
value would let any server exempt itself from approval — including one
installed by pasting a command off a web page, which is how most of them get
installed. So hints are ignored unless an operator marks that server trusted,
and until then every MCP tool is at least `execute`.

Names and descriptions still get read, because a tool called `create_order`
is worth grading as a purchase whatever its hints say. That check runs on the
server's own text, which is exactly the kind of thing an attacker controls —
but it can only ever *raise* the risk, so a hostile description makes a tool
harder to run, not easier.

A server is a local command or a remote URL, and the config says which by
which key it carries. Both are the shapes the rest of the ecosystem already
writes, so an `.mcp.json` from another client works here unchanged — including
the hosted servers in it, which before this could only be described and not
reached.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openmirror.mcp.client import (
    HTTPServer,
    MCPError,
    MCPPrompt,
    MCPResource,
    MCPServer,
    MCPTool,
    SSEServer,
    StdioServer,
)

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ServerConfig:
    name: str
    # A local server: the command to run. Empty for a remote one.
    command: str = ''
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    # A remote server: where it lives. `transport` is how to talk to it —
    # `http` for the streamable transport, `sse` for the older pair. Inferred
    # from the URL when the config does not say, because most files do not.
    url: str = ''
    transport: str = ''
    # Sent with every request. Where a hosted server's token goes, which is
    # why it is per-server and not a global setting: two servers needing two
    # different tokens is the normal case.
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    # Believe this server's own readOnly/destructive hints. Off by default:
    # a server that can declare itself harmless is a server that can opt out
    # of the approval it most needs.
    trust_hints: bool = False

    def resolved_transport(self) -> str:
        """How to reach this server, with the guesswork done once, here.

        An explicit `type` wins. Otherwise a URL ending in `/sse` is the older
        transport — that path is the convention every SSE server uses — and
        anything else with a URL is streamable HTTP, which is what a new
        server will be.
        """
        if self.transport:
            return self.transport
        if not self.url:
            return 'stdio'
        return 'sse' if self.url.rstrip('/').endswith('/sse') else 'http'


def build_server(cfg: ServerConfig, timeout: float | None = None) -> MCPServer:
    """The client for one config entry.

    Kept out of `MCPManager.start` so a caller with one server and no manager
    — a test, or `openmirror mcp probe` — builds it the same way the daemon does.
    """
    kind = cfg.resolved_transport()
    extra = {'timeout': timeout} if timeout else {}
    if kind == 'stdio':
        if not cfg.command:
            raise MCPError(f'{cfg.name}: neither a command nor a url')
        return StdioServer(cfg.name, cfg.command, cfg.args, cfg.env, cfg.cwd, **extra)
    if not cfg.url:
        raise MCPError(f'{cfg.name}: transport {kind!r} needs a url')
    if kind == 'sse':
        return SSEServer(cfg.name, cfg.url, cfg.headers, **extra)
    if kind in ('http', 'streamable-http', 'streamableHttp'):
        return HTTPServer(cfg.name, cfg.url, cfg.headers, **extra)
    raise MCPError(f'{cfg.name}: unknown transport {kind!r} (stdio, http, sse)')


def load_config(path: Path) -> list[ServerConfig]:
    """Read a `.mcp.json`, in the shape the ecosystem already uses.

        {"mcpServers": {
          "files":  {"command": "npx", "args": ["-y", "@mcp/files", "/srv"]},
          "hosted": {"type": "http", "url": "https://example.com/mcp",
                     "headers": {"Authorization": "Bearer ..."}},
          "legacy": {"url": "https://example.com/sse"}
        }}

    Following the existing convention rather than inventing one means an
    `.mcp.json` written for another client works here unchanged. An entry
    needs a `command` or a `url`; one with neither is skipped and said so,
    because the alternative is a server that silently is not there.
    """
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning('mcp: could not read %s: %s', path, exc)
        return []

    servers: list[ServerConfig] = []
    for name, entry in (raw.get('mcpServers') or {}).items():
        if not isinstance(entry, dict):
            log.warning('mcp: %s is not an object, skipping', name)
            continue
        if not entry.get('command') and not entry.get('url'):
            log.warning('mcp: %s has neither a command nor a url, skipping', name)
            continue
        servers.append(
            ServerConfig(
                name=name,
                command=entry.get('command', ''),
                args=list(entry.get('args') or []),
                env=dict(entry.get('env') or {}),
                cwd=entry.get('cwd'),
                url=entry.get('url', ''),
                # `type` is what the ecosystem's files say; `transport` is
                # what some of them say instead.
                transport=str(entry.get('type') or entry.get('transport') or ''),
                headers=dict(entry.get('headers') or {}),
                enabled=entry.get('enabled', True),
                trust_hints=bool(entry.get('trustHints', False)),
            )
        )
    return servers


class MCPManager:
    def __init__(self) -> None:
        self.servers: dict[str, MCPServer] = {}
        self.configs: dict[str, ServerConfig] = {}

    async def start(self, configs: list[ServerConfig]) -> dict[str, str]:
        """Start each server. Returns {name: error} for the ones that failed.

        One server failing must not stop the rest: they are independent, and an
        agent with four working tool servers and one broken one is far more
        useful than an agent that refused to start.
        """
        failures: dict[str, str] = {}
        for cfg in configs:
            if not cfg.enabled:
                continue
            try:
                server = build_server(cfg)
            except MCPError as exc:
                failures[cfg.name] = str(exc)
                log.warning('mcp %s: %s', cfg.name, exc)
                continue
            try:
                await server.start()
            except (MCPError, OSError) as exc:
                failures[cfg.name] = str(exc)
                log.warning('mcp %s: %s', cfg.name, exc)
                await server.stop()
                continue
            self.servers[cfg.name] = server
            self.configs[cfg.name] = cfg
        return failures

    async def stop(self) -> None:
        for server in list(self.servers.values()):
            await server.stop()
        self.servers.clear()
        self.configs.clear()

    def tools(self) -> list[tuple[MCPTool, ServerConfig]]:
        out = []
        for name, server in self.servers.items():
            cfg = self.configs[name]
            out.extend((tool, cfg) for tool in server.tools)
        return out

    def resources(self, server: str = '') -> list[MCPResource]:
        """Everything readable across the running servers, one flat list.

        Flat and carrying the server's name on each row, rather than grouped:
        a model deciding what to read wants one list it can scan, and the
        grouping is recoverable from the rows.
        """
        return [
            resource
            for name, live in self.servers.items()
            if not server or name == server
            for resource in live.resources
        ]

    def prompts(self, server: str = '') -> list[MCPPrompt]:
        return [
            prompt
            for name, live in self.servers.items()
            if not server or name == server
            for prompt in live.prompts
        ]

    async def read_resource(self, server: str, uri: str):
        target = self.servers.get(server)
        if target is None:
            raise MCPError(f'no MCP server named {server!r}')
        return await target.read_resource(uri)

    async def get_prompt(self, server: str, name: str, arguments: dict[str, Any] | None = None) -> str:
        target = self.servers.get(server)
        if target is None:
            raise MCPError(f'no MCP server named {server!r}')
        return await target.get_prompt(name, arguments or {})

    def find_resource(self, uri: str, server: str = '') -> tuple[str, MCPResource] | None:
        """Which server owns a URI.

        Needed because the agent-facing tool takes a URI and the server name
        is optional: two servers exposing the same URI is possible but rare,
        and making a model name the server every time costs more than it saves.
        """
        for name, live in self.servers.items():
            if server and name != server:
                continue
            for resource in live.resources:
                if resource.uri == uri:
                    return name, resource
        return None

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                'name': name,
                'running': server.running,
                'server': server.server_info.get('name', ''),
                'version': server.server_info.get('version', ''),
                'transport': server.transport,
                'tools': [t.name for t in server.tools],
                'resources': [r.uri for r in server.resources],
                'prompts': [p.name for p in server.prompts],
                'trust_hints': self.configs[name].trust_hints,
            }
            for name, server in self.servers.items()
        ]

    async def call(self, server: str, tool: str, arguments: dict[str, Any]):
        target = self.servers.get(server)
        if target is None:
            raise MCPError(f'no MCP server named {server!r}')
        return await target.call(tool, arguments)


manager = MCPManager()
