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
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openmirror.mcp.client import MCPError, MCPTool, StdioServer

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    enabled: bool = True
    # Believe this server's own readOnly/destructive hints. Off by default:
    # a server that can declare itself harmless is a server that can opt out
    # of the approval it most needs.
    trust_hints: bool = False


def load_config(path: Path) -> list[ServerConfig]:
    """Read a `.mcp.json`, in the shape the ecosystem already uses.

        {"mcpServers": {"files": {"command": "npx", "args": ["-y", "@mcp/files", "/srv"]}}}

    Following the existing convention rather than inventing one means an
    `.mcp.json` written for another client works here unchanged.
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
        if not isinstance(entry, dict) or not entry.get('command'):
            log.warning('mcp: %s has no command, skipping', name)
            continue
        servers.append(
            ServerConfig(
                name=name,
                command=entry['command'],
                args=list(entry.get('args') or []),
                env=dict(entry.get('env') or {}),
                cwd=entry.get('cwd'),
                enabled=entry.get('enabled', True),
                trust_hints=bool(entry.get('trustHints', False)),
            )
        )
    return servers


class MCPManager:
    def __init__(self) -> None:
        self.servers: dict[str, StdioServer] = {}
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
            server = StdioServer(cfg.name, cfg.command, cfg.args, cfg.env, cfg.cwd)
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

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                'name': name,
                'running': server.running,
                'server': server.server_info.get('name', ''),
                'version': server.server_info.get('version', ''),
                'tools': [t.name for t in server.tools],
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
