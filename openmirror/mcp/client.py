"""A Model Context Protocol client.

MCP is how an agent gets tools it did not ship with: a server exposes some
capability — a database, an issue tracker, a company's internal API — over
JSON-RPC, and any client can use it. Supporting it is the difference between
openmirror having the tools I wrote and openmirror having every tool anyone writes.

Implemented directly rather than through the official SDK. The protocol is
JSON-RPC 2.0 over newline-delimited stdio, which is perhaps two hundred lines
of real work, and this way there is no dependency to keep in step and — more
importantly — the subprocess lifetime is ours. An MCP server is a child
process that outlives its usefulness by default; the reaping below is the part
a wrapper would hide.

**Everything a server sends is untrusted.** Tool descriptions are written by
whoever wrote the server and go straight into the model's prompt, which makes
them an injection vector with a very short path. Their risk annotations are
self-declared, so a server can call itself read-only; that is why the manager
does not believe them unless you say so.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

PROTOCOL_VERSION = '2024-11-05'
DEFAULT_TIMEOUT = 60.0
# A server that has not answered `initialize` by now is not going to.
STARTUP_TIMEOUT = 30.0


class MCPError(RuntimeError):
    pass


@dataclass(slots=True)
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    # The server's own hints: readOnlyHint, destructiveHint, idempotentHint,
    # openWorldHint. Self-declared, so believed only where an operator has said
    # that server may be believed.
    annotations: dict[str, Any] = field(default_factory=dict)
    server: str = ''


@dataclass(slots=True)
class MCPResult:
    text: str
    is_error: bool = False
    # Images a tool returned, as (base64, media_type), for models that can see.
    images: list[tuple[str, str]] = field(default_factory=list)


class StdioServer:
    """One MCP server, running as a child process."""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.cwd = cwd
        self.timeout = timeout

        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader: asyncio.Task[None] | None = None
        self._stderr: asyncio.Task[None] | None = None
        self.tools: list[MCPTool] = []
        self.server_info: dict[str, Any] = {}

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        binary = shutil.which(self.command)
        if binary is None:
            raise MCPError(f'{self.name}: {self.command!r} is not on PATH')

        self._proc = await asyncio.create_subprocess_exec(
            binary,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **self.env},
            cwd=self.cwd,
        )
        self._reader = asyncio.create_task(self._read_loop())
        # Servers use stderr for logging and some are chatty. Drained rather
        # than ignored, because a full pipe blocks the child for ever.
        self._stderr = asyncio.create_task(self._drain_stderr())

        try:
            result = await asyncio.wait_for(
                self._request('initialize', {
                    'protocolVersion': PROTOCOL_VERSION,
                    'capabilities': {},
                    'clientInfo': {'name': 'openmirror', 'version': '0.1.0'},
                }),
                timeout=STARTUP_TIMEOUT,
            )
        except TimeoutError as exc:
            await self.stop()
            raise MCPError(f'{self.name}: did not answer initialize within {STARTUP_TIMEOUT:.0f}s') from exc

        self.server_info = result.get('serverInfo') or {}
        await self._notify('notifications/initialized', {})
        await self.refresh_tools()
        log.info(
            'mcp %s: %s, %d tool(s)',
            self.name, self.server_info.get('name', 'unnamed'), len(self.tools),
        )

    async def stop(self) -> None:
        for task in (self._reader, self._stderr):
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        proc = self._proc
        self._proc = None
        if proc is None or proc.returncode is not None:
            return

        # Closing stdin is the protocol's own way of saying goodbye; most
        # servers exit on EOF. The kill is for the ones that do not.
        with contextlib.suppress(Exception):
            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    # -- transport ----------------------------------------------------------

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                break
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                # Servers occasionally print a banner to stdout before the
                # protocol starts. Skipped rather than fatal.
                log.debug('mcp %s: non-JSON on stdout: %r', self.name, line[:120])
                continue

            msg_id = message.get('id')
            if msg_id is None:
                continue  # a notification from the server; nothing wants it yet
            future = self._pending.pop(msg_id, None)
            if future is None or future.done():
                continue
            if 'error' in message:
                future.set_exception(
                    MCPError(f'{self.name}: {message["error"].get("message", "unknown error")}')
                )
            else:
                future.set_result(message.get('result') or {})

        # The process died. Anything still waiting must be told, or it waits
        # for ever on a pipe that will never answer.
        for future in self._pending.values():
            if not future.done():
                future.set_exception(MCPError(f'{self.name}: server exited'))
        self._pending.clear()

    async def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        while True:
            line = await self._proc.stderr.readline()
            if not line:
                break
            log.debug('mcp %s: %s', self.name, line.decode('utf-8', 'replace').rstrip())

    async def _send(self, payload: dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            raise MCPError(f'{self.name}: not running')
        self._proc.stdin.write(json.dumps(payload).encode() + b'\n')
        await self._proc.stdin.drain()

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        self._next_id += 1
        msg_id = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = future
        await self._send({'jsonrpc': '2.0', 'id': msg_id, 'method': method, 'params': params})
        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except TimeoutError as exc:
            self._pending.pop(msg_id, None)
            raise MCPError(f'{self.name}: {method} timed out after {self.timeout:.0f}s') from exc

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({'jsonrpc': '2.0', 'method': method, 'params': params})

    # -- protocol -----------------------------------------------------------

    async def refresh_tools(self) -> list[MCPTool]:
        result = await self._request('tools/list', {})
        self.tools = [
            MCPTool(
                name=t['name'],
                description=t.get('description', ''),
                input_schema=t.get('inputSchema') or {'type': 'object', 'properties': {}},
                annotations=t.get('annotations') or {},
                server=self.name,
            )
            for t in result.get('tools', [])
            if t.get('name')
        ]
        return self.tools

    async def call(self, tool: str, arguments: dict[str, Any]) -> MCPResult:
        result = await self._request('tools/call', {'name': tool, 'arguments': arguments})

        chunks: list[str] = []
        images: list[tuple[str, str]] = []
        for part in result.get('content') or []:
            kind = part.get('type')
            if kind == 'text':
                chunks.append(part.get('text', ''))
            elif kind == 'image' and part.get('data'):
                images.append((part['data'], part.get('mimeType', 'image/png')))
            elif kind == 'resource':
                resource = part.get('resource') or {}
                chunks.append(resource.get('text') or f'[resource: {resource.get("uri", "?")}]')

        return MCPResult(
            text='\n'.join(c for c in chunks if c) or '(the tool returned nothing)',
            is_error=bool(result.get('isError')),
            images=images,
        )
