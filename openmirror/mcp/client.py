"""A Model Context Protocol client.

MCP is how an agent gets tools it did not ship with: a server exposes some
capability — a database, an issue tracker, a company's internal API — over
JSON-RPC, and any client can use it. Supporting it is the difference between
openmirror having the tools I wrote and openmirror having every tool anyone writes.

Implemented directly rather than through the official SDK. The protocol is
JSON-RPC 2.0, which is perhaps two hundred lines of real work per transport,
and this way there is no dependency to keep in step and — more importantly —
the subprocess lifetime is ours. An MCP server is a child process that
outlives its usefulness by default; the reaping below is the part a wrapper
would hide.

**Three transports, because servers come in three shapes.** A local server is
a child process speaking newline-delimited JSON over stdio. A remote one
speaks HTTP: either the current streamable transport, where each POST's reply
carries the answer, or the older SSE pair, where a long-lived stream carries
replies for requests POSTed somewhere else. Only supporting stdio would have
meant only supporting servers you can install, which leaves out every hosted
one — and a hosted server is exactly the case where *not* running somebody's
code on your machine is the point.

**Tools are not the whole protocol.** A server also exposes *resources* (a
file, a record, a page, addressed by URI) and *prompts* (a canned request its
author thinks you will want). Resources are how a server offers something to
read without guessing when you want it read, which is a better fit for a large
corpus than a tool per document. Both are fetched here and both are optional:
a server that declares neither is never asked, because a client that calls
`resources/list` on a tools-only server gets an error back and logs noise for
ever.

**Everything a server sends is untrusted.** Tool descriptions are written by
whoever wrote the server and go straight into the model's prompt, which makes
them an injection vector with a very short path. Their risk annotations are
self-declared, so a server can call itself read-only; that is why the manager
does not believe them unless you say so. Resource *contents* are worse, since
they are usually fetched from somewhere else again — the agent-facing wrapper
labels them, every time.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

log = logging.getLogger(__name__)

# What we ask for. A server answers with its own, and the difference is not
# enforced: every server in the wild implements the parts used here the same
# way, and refusing to talk to one over a version string helps nobody.
PROTOCOL_VERSION = '2025-06-18'
DEFAULT_TIMEOUT = 60.0
# A server that has not answered `initialize` by now is not going to.
STARTUP_TIMEOUT = 30.0

# JSON-RPC's "I have no such method". Worth its own name because it is the
# answer a tools-only server gives to `resources/list`, and that is a normal
# thing to hear rather than a failure.
METHOD_NOT_FOUND = -32601


class MCPError(RuntimeError):
    pass


class MCPUnsupported(MCPError):
    """The server does not implement that method.

    Separate so an optional part of the protocol can be probed without the
    absence of it reading as a broken server.
    """


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
class MCPResource:
    """Something the server will hand over if asked, addressed by URI.

    `name` and `description` are the server's own words and reach the model,
    so they are treated the same way a tool description is.
    """

    uri: str
    name: str = ''
    description: str = ''
    mime_type: str = ''
    server: str = ''
    # True for a resource *template* — a URI with `{placeholders}` in it,
    # which cannot be read until they are filled in. Kept in the same list
    # because the thing a model wants to see is "what can I ask for", and
    # split apart it would have to learn two lists mean one thing.
    template: bool = False


@dataclass(slots=True)
class MCPPrompt:
    name: str
    description: str = ''
    arguments: list[dict[str, Any]] = field(default_factory=list)
    server: str = ''


@dataclass(slots=True)
class MCPResult:
    text: str
    is_error: bool = False
    # Images a tool returned, as (base64, media_type), for models that can see.
    images: list[tuple[str, str]] = field(default_factory=list)


def _content_to_result(parts: list[dict[str, Any]], *, is_error: bool = False) -> MCPResult:
    """Flatten MCP content blocks into text and images.

    Shared by tool results and resource reads because the two carry the same
    block types, and a resource that returns a PNG should reach a model that
    can see as a picture rather than as the words "[resource]".
    """
    chunks: list[str] = []
    images: list[tuple[str, str]] = []
    for part in parts or []:
        kind = part.get('type')
        if kind == 'text':
            chunks.append(part.get('text', ''))
        elif kind == 'image' and part.get('data'):
            images.append((part['data'], part.get('mimeType', 'image/png')))
        elif kind in ('resource', 'resource_link'):
            resource = part.get('resource') or part
            text = resource.get('text')
            if text:
                chunks.append(text)
            elif resource.get('blob') and str(resource.get('mimeType', '')).startswith('image/'):
                images.append((resource['blob'], resource['mimeType']))
            else:
                chunks.append(f'[resource: {resource.get("uri", "?")}]')
    return MCPResult(
        text='\n'.join(c for c in chunks if c) or '(the tool returned nothing)',
        is_error=is_error,
        images=images,
    )


class MCPServer(abc.ABC):
    """One MCP server, whatever it is running on.

    Everything above the wire lives here — the handshake, what the server said
    it can do, and the four calls that matter — so adding a transport is a
    matter of saying how a request gets sent, not reimplementing the protocol.
    """

    transport = 'unknown'

    def __init__(self, name: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.name = name
        self.timeout = timeout

        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self.tools: list[MCPTool] = []
        self.resources: list[MCPResource] = []
        self.prompts: list[MCPPrompt] = []
        self.server_info: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        await self._open()
        try:
            result = await asyncio.wait_for(
                self._request('initialize', {
                    'protocolVersion': PROTOCOL_VERSION,
                    # Said plainly rather than left empty: a client that
                    # declares nothing gets servers that volunteer nothing.
                    'capabilities': {},
                    'clientInfo': {'name': 'openmirror', 'version': '0.1.0'},
                }),
                timeout=STARTUP_TIMEOUT,
            )
        except TimeoutError as exc:
            await self.stop()
            raise MCPError(f'{self.name}: did not answer initialize within {STARTUP_TIMEOUT:.0f}s') from exc
        except MCPError:
            await self.stop()
            raise

        self.server_info = result.get('serverInfo') or {}
        self.capabilities = result.get('capabilities') or {}
        await self._notify('notifications/initialized', {})
        await self.refresh()
        log.info(
            'mcp %s: %s over %s, %d tool(s), %d resource(s), %d prompt(s)',
            self.name, self.server_info.get('name', 'unnamed'), self.transport,
            len(self.tools), len(self.resources), len(self.prompts),
        )

    async def stop(self) -> None:
        await self._close()

    @property
    @abc.abstractmethod
    def running(self) -> bool: ...

    @abc.abstractmethod
    async def _open(self) -> None: ...

    @abc.abstractmethod
    async def _close(self) -> None: ...

    @abc.abstractmethod
    async def _request(self, method: str, params: dict[str, Any]) -> Any: ...

    @abc.abstractmethod
    async def _notify(self, method: str, params: dict[str, Any]) -> None: ...

    # -- plumbing shared by the transports that dispatch replies -------------

    def _claim_id(self) -> tuple[int, asyncio.Future[Any]]:
        self._next_id += 1
        msg_id = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = future
        return msg_id, future

    def _resolve(self, message: dict[str, Any]) -> None:
        """Hand one decoded JSON-RPC message to whoever is waiting for it."""
        msg_id = message.get('id')
        if msg_id is None:
            return  # a notification from the server; nothing wants it yet
        future = self._pending.pop(msg_id, None)
        if future is None or future.done():
            return
        if 'error' in message:
            future.set_exception(self._error(message['error']))
        else:
            future.set_result(message.get('result') or {})

    def _error(self, error: dict[str, Any]) -> MCPError:
        message = error.get('message', 'unknown error')
        if error.get('code') == METHOD_NOT_FOUND:
            return MCPUnsupported(f'{self.name}: {message}')
        return MCPError(f'{self.name}: {message}')

    def _abandon(self, why: str) -> None:
        """Fail everything still waiting. Called when the transport dies.

        Without this a caller waits for ever on a channel that will never
        answer, which is indistinguishable from a slow server and is how a
        request outlives the process it was sent to.
        """
        for future in self._pending.values():
            if not future.done():
                future.set_exception(MCPError(f'{self.name}: {why}'))
        self._pending.clear()

    # -- protocol -----------------------------------------------------------

    async def refresh(self) -> None:
        """Fetch everything the server said it has.

        Gated on the declared capabilities, and tolerant of a server that
        declares one and then does not implement it. Both halves matter: the
        gate keeps a tools-only server from being asked about resources at
        all, and the tolerance covers the servers that declare capabilities
        optimistically — of which there are many.
        """
        await self.refresh_tools()
        if 'resources' in self.capabilities:
            with contextlib.suppress(MCPUnsupported):
                await self.refresh_resources()
        if 'prompts' in self.capabilities:
            with contextlib.suppress(MCPUnsupported):
                await self.refresh_prompts()

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

    async def refresh_resources(self) -> list[MCPResource]:
        found: list[MCPResource] = []

        result = await self._request('resources/list', {})
        for r in result.get('resources', []):
            if r.get('uri'):
                found.append(MCPResource(
                    uri=r['uri'],
                    name=r.get('name', ''),
                    description=r.get('description', ''),
                    mime_type=r.get('mimeType', ''),
                    server=self.name,
                ))

        # Templates are a separate call and many servers have one without the
        # other, so its absence is not allowed to lose the concrete list.
        with contextlib.suppress(MCPUnsupported, MCPError):
            templates = await self._request('resources/templates/list', {})
            for r in templates.get('resourceTemplates', []):
                if r.get('uriTemplate'):
                    found.append(MCPResource(
                        uri=r['uriTemplate'],
                        name=r.get('name', ''),
                        description=r.get('description', ''),
                        mime_type=r.get('mimeType', ''),
                        server=self.name,
                        template=True,
                    ))

        self.resources = found
        return self.resources

    async def refresh_prompts(self) -> list[MCPPrompt]:
        result = await self._request('prompts/list', {})
        self.prompts = [
            MCPPrompt(
                name=p['name'],
                description=p.get('description', ''),
                arguments=list(p.get('arguments') or []),
                server=self.name,
            )
            for p in result.get('prompts', [])
            if p.get('name')
        ]
        return self.prompts

    async def call(self, tool: str, arguments: dict[str, Any]) -> MCPResult:
        result = await self._request('tools/call', {'name': tool, 'arguments': arguments})
        return _content_to_result(result.get('content') or [], is_error=bool(result.get('isError')))

    async def read_resource(self, uri: str) -> MCPResult:
        """Fetch one resource.

        The blocks come back shaped like tool content but named differently —
        `contents` rather than `content`, and carrying `text`/`blob` instead of
        a type — so they are normalised here into the same result a tool gives.
        That is what lets one wrapper attribute both on the way to the model.
        """
        result = await self._request('resources/read', {'uri': uri})
        parts: list[dict[str, Any]] = []
        for item in result.get('contents') or []:
            if item.get('text') is not None:
                parts.append({'type': 'text', 'text': item['text']})
            elif item.get('blob') and str(item.get('mimeType', '')).startswith('image/'):
                parts.append({'type': 'image', 'data': item['blob'], 'mimeType': item['mimeType']})
            else:
                parts.append({'type': 'text', 'text': f'[{item.get("mimeType", "binary")} at {item.get("uri", uri)}]'})
        out = _content_to_result(parts)
        if out.text == '(the tool returned nothing)':
            out.text = f'({uri} is empty)'
        return out

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Expand one of the server's prompts into text.

        Flattened to a string rather than returned as messages: a prompt from
        a third party is material for the model to read, not turns to splice
        into the conversation as though a person had said them.
        """
        result = await self._request('prompts/get', {'name': name, 'arguments': arguments or {}})
        lines: list[str] = []
        for message in result.get('messages') or []:
            content = message.get('content')
            blocks = content if isinstance(content, list) else [content]
            text = _content_to_result([b for b in blocks if isinstance(b, dict)]).text
            lines.append(f'{message.get("role", "user")}: {text}')
        return '\n\n'.join(lines) or '(the prompt expanded to nothing)'


class StdioServer(MCPServer):
    """One MCP server, running as a child process."""

    transport = 'stdio'

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(name, timeout)
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.cwd = cwd

        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stderr: asyncio.Task[None] | None = None

    # -- lifecycle ----------------------------------------------------------

    async def _open(self) -> None:
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

    async def _close(self) -> None:
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
            self._resolve(message)

        # The process died. Anything still waiting must be told, or it waits
        # for ever on a pipe that will never answer.
        self._abandon('server exited')

    async def _send(self, payload: dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            raise MCPError(f'{self.name}: not running')
        self._proc.stdin.write(json.dumps(payload).encode() + b'\n')
        await self._proc.stdin.drain()

    async def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        while True:
            line = await self._proc.stderr.readline()
            if not line:
                break
            log.debug('mcp %s: %s', self.name, line.decode('utf-8', 'replace').rstrip())

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        msg_id, future = self._claim_id()
        try:
            await self._send({'jsonrpc': '2.0', 'id': msg_id, 'method': method, 'params': params})
        except MCPError:
            self._pending.pop(msg_id, None)
            raise
        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except TimeoutError as exc:
            self._pending.pop(msg_id, None)
            raise MCPError(f'{self.name}: {method} timed out after {self.timeout:.0f}s') from exc

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({'jsonrpc': '2.0', 'method': method, 'params': params})


def _sse_messages(chunk: str, buffer: list[str]) -> list[dict[str, Any]]:
    """Decode whatever complete SSE events are in `chunk`.

    `buffer` holds the partial event across calls, because a chunk boundary
    falls wherever the network put it and an event split across two reads is
    the normal case rather than the exception.
    """
    out: list[dict[str, Any]] = []
    buffer.append(chunk)
    text = ''.join(buffer)
    buffer.clear()

    # An event ends at a blank line; anything after the last one is incomplete.
    *events, rest = text.split('\n\n')
    buffer.append(rest)

    for event in events:
        data = '\n'.join(
            line[5:].lstrip() for line in event.splitlines() if line.startswith('data:')
        )
        if not data or data == '[DONE]':
            continue
        try:
            out.append(json.loads(data))
        except json.JSONDecodeError:
            log.debug('sse: non-JSON event: %r', data[:120])
    return out


class HTTPServer(MCPServer):
    """A remote MCP server over the streamable HTTP transport.

    One endpoint, one POST per request, and the reply comes back in that
    POST's response — as JSON when there is one message to send, or as an SSE
    stream when the server wants to interleave progress with it. Both are
    handled, because which one you get is the server's choice and not
    something a client may require.

    The session id is the part that looks like boilerplate and is not: a
    server that issued one and does not see it again treats the next request
    as a new client, which fails after `initialize` succeeded — the most
    confusing possible place for it to fail.
    """

    transport = 'http'

    def __init__(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(name, timeout)
        self.url = url
        self.headers = dict(headers or {})
        self._session: Any = None
        self._session_id = ''
        self._closed = False

    async def _open(self) -> None:
        import aiohttp

        # transport-exempt: an MCP server is a tool endpoint the operator named
        # in .mcp.json, not a model server, so a provider connection's Tor
        # routing does not apply to it — routing somebody's issue tracker
        # through the proxy their GPU needs would be one connection's rule
        # silently becoming another's.
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        self._closed = False

    async def _close(self) -> None:
        self._closed = True
        self._abandon('connection closed')
        session = self._session
        self._session = None
        if session is not None:
            await session.close()

    @property
    def running(self) -> bool:
        return self._session is not None and not self._closed

    def _request_headers(self) -> dict[str, str]:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
            **self.headers,
        }
        if self._session_id:
            headers['Mcp-Session-Id'] = self._session_id
        if self.server_info:
            # Only after the handshake: sending a protocol version before one
            # has been agreed is how a server that pins versions rejects the
            # handshake itself.
            headers['MCP-Protocol-Version'] = PROTOCOL_VERSION
        return headers

    async def _post(self, payload: dict[str, Any]) -> Any:
        import aiohttp

        if self._session is None:
            raise MCPError(f'{self.name}: not connected')

        try:
            async with self._session.post(
                self.url, json=payload, headers=self._request_headers()
            ) as resp:
                issued = resp.headers.get('Mcp-Session-Id') or resp.headers.get('mcp-session-id')
                if issued:
                    self._session_id = issued

                if resp.status == 401 or resp.status == 403:
                    raise MCPError(
                        f'{self.name}: the server refused the request (HTTP {resp.status}). '
                        'If it needs a token, put it in the server\'s `headers` in .mcp.json.'
                    )
                if resp.status >= 400:
                    body = (await resp.text())[:300]
                    raise MCPError(f'{self.name}: HTTP {resp.status} from {self.url}: {body}')

                # 202 with no body is the correct answer to a notification.
                if resp.status == 202 or resp.content_length == 0:
                    return None

                ctype = resp.headers.get('Content-Type', '')
                if 'text/event-stream' in ctype:
                    return await self._read_stream(resp, payload.get('id'))
                return await resp.json(content_type=None)
        except aiohttp.ClientError as exc:
            raise MCPError(f'{self.name}: {self.url}: {exc}') from exc

    async def _read_stream(self, resp: Any, want_id: Any) -> Any:
        """Read an SSE response until the message we asked for arrives.

        Progress notifications share the stream with the answer, so the loop
        cannot stop at the first message — it stops at the first one carrying
        our id, and drops the rest on the floor along with the connection.
        """
        buffer: list[str] = []
        async for chunk in resp.content.iter_any():
            for message in _sse_messages(chunk.decode('utf-8', 'replace'), buffer):
                if message.get('id') == want_id:
                    return message
                self._resolve(message)
        raise MCPError(f'{self.name}: the stream ended before answering')

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        self._next_id += 1
        msg_id = self._next_id
        message = await self._post(
            {'jsonrpc': '2.0', 'id': msg_id, 'method': method, 'params': params}
        )
        if message is None:
            raise MCPError(f'{self.name}: {method} got an empty reply')
        if isinstance(message, list):
            # A batched reply. Ours is the one with our id.
            message = next((m for m in message if m.get('id') == msg_id), message[0])
        if 'error' in message:
            raise self._error(message['error'])
        return message.get('result') or {}

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._post({'jsonrpc': '2.0', 'method': method, 'params': params})


class SSEServer(MCPServer):
    """A remote MCP server over the older HTTP+SSE transport.

    Two channels: a GET that stays open and carries every reply, and a POST
    endpoint the server names on that stream. Superseded by streamable HTTP
    and still what a good number of deployed servers speak, including ones
    published before the change — which is the whole reason it is here.
    """

    transport = 'sse'

    def __init__(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(name, timeout)
        self.url = url
        self.headers = dict(headers or {})
        self._session: Any = None
        self._stream: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._post_url = ''
        self._ready: asyncio.Event = asyncio.Event()

    async def _open(self) -> None:
        import aiohttp

        # transport-exempt: an MCP server is a tool endpoint the operator named
        # in .mcp.json, not a model server, so a provider connection's Tor
        # routing does not apply to it — routing somebody's issue tracker
        # through the proxy their GPU needs would be one connection's rule
        # silently becoming another's.
        self._session = aiohttp.ClientSession(
            # No total timeout: the point of this socket is to stay open.
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
        )
        self._ready = asyncio.Event()
        try:
            self._stream = await self._session.get(
                self.url, headers={'Accept': 'text/event-stream', **self.headers}
            )
        except aiohttp.ClientError as exc:
            await self._close()
            raise MCPError(f'{self.name}: could not open {self.url}: {exc}') from exc

        if self._stream.status >= 400:
            status = self._stream.status
            await self._close()
            raise MCPError(f'{self.name}: HTTP {status} opening the event stream at {self.url}')

        self._reader = asyncio.create_task(self._read_loop())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=STARTUP_TIMEOUT)
        except TimeoutError as exc:
            await self._close()
            raise MCPError(
                f'{self.name}: the event stream never named a POST endpoint — '
                'it may be a streamable HTTP server, which is `"type": "http"`.'
            ) from exc

    async def _close(self) -> None:
        if self._reader and not self._reader.done():
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
        self._reader = None
        self._abandon('connection closed')

        if self._stream is not None:
            self._stream.close()
            self._stream = None
        session = self._session
        self._session = None
        if session is not None:
            await session.close()

    @property
    def running(self) -> bool:
        return self._session is not None and bool(self._post_url)

    async def _read_loop(self) -> None:
        """Dispatch replies, and pick the POST endpoint out of the stream.

        The endpoint arrives as a named SSE event whose data is a URL rather
        than JSON, which is why the event name is parsed here instead of
        reusing the decoder the other transport uses.
        """
        buffer = ''
        try:
            async for chunk in self._stream.content.iter_any():
                buffer += chunk.decode('utf-8', 'replace')
                while '\n\n' in buffer:
                    raw, buffer = buffer.split('\n\n', 1)
                    event = 'message'
                    data_lines: list[str] = []
                    for line in raw.splitlines():
                        if line.startswith('event:'):
                            event = line[6:].strip()
                        elif line.startswith('data:'):
                            data_lines.append(line[5:].lstrip())
                    data = '\n'.join(data_lines)
                    if not data:
                        continue
                    if event == 'endpoint':
                        # Relative in every implementation seen, absolute in
                        # the spec's examples. urljoin handles both.
                        self._post_url = urljoin(self.url, data)
                        self._ready.set()
                        continue
                    try:
                        self._resolve(json.loads(data))
                    except json.JSONDecodeError:
                        log.debug('mcp %s: non-JSON event: %r', self.name, data[:120])
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:  # noqa: BLE001 - any read failure ends the stream
            log.debug('mcp %s: event stream ended: %s', self.name, exc)
        self._abandon('the event stream closed')

    async def _send(self, payload: dict[str, Any]) -> None:
        import aiohttp

        if self._session is None or not self._post_url:
            raise MCPError(f'{self.name}: not connected')
        try:
            async with self._session.post(
                self._post_url,
                json=payload,
                headers={'Content-Type': 'application/json', **self.headers},
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status >= 400:
                    body = (await resp.text())[:300]
                    raise MCPError(f'{self.name}: HTTP {resp.status} posting to {self._post_url}: {body}')
        except aiohttp.ClientError as exc:
            raise MCPError(f'{self.name}: {self._post_url}: {exc}') from exc

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        msg_id, future = self._claim_id()
        try:
            await self._send({'jsonrpc': '2.0', 'id': msg_id, 'method': method, 'params': params})
        except MCPError:
            self._pending.pop(msg_id, None)
            raise
        try:
            return await asyncio.wait_for(future, timeout=self.timeout)
        except TimeoutError as exc:
            self._pending.pop(msg_id, None)
            raise MCPError(f'{self.name}: {method} timed out after {self.timeout:.0f}s') from exc

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({'jsonrpc': '2.0', 'method': method, 'params': params})
