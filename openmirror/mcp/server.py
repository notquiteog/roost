"""openmirror as an MCP server: the twin, as tools in somebody else's client.

The client half of this package is about openmirror using other people's tools.
This is the other direction, and it is the direction that makes a digital twin
worth having in more than one place. What openmirror holds that nothing else does
is *your* memory, *your* provider routing and *your* search backend — and
without this file all three are reachable only from openmirror's own UI. With it,
an editor, a desktop assistant or another agent can ask the twin what it knows
and get an answer grounded in the same memory the voice session uses.

Four decisions, all of them about not handing out more than was asked for.

**Off, until it is not.** The HTTP endpoint is not mounted unless
`OPENMIRROR_MCP_SERVE` says so, and then it wants a token. Memory is the most
personal thing in the project and a digital twin on an open port is a liability
rather than a feature; the stdio transport needs no token because a parent
process that spawned it already has everything this could give it.

**Scope, not a pile of switches.** `read` exposes the things that only look:
recall, search, fetch, research. `write` adds remembering. `all` adds media
generation, which spends money. One setting with three values, because the
decision a person actually makes is how much of themselves to lend out, not
eleven tool-shaped decisions.

**Nothing that acts on the machine, at any scope.** No shell, no files, no
desktop, no browser driving. Those are the tools openmirror guards with an
approval prompt in front of a human, and an MCP call has no human in front of
it — exposing them here would route around the one control that matters. The
approval policy is not reimplemented on this side; the dangerous tools are
simply not here.

**Generated media goes out as a resource, not as megabytes of base64.** A tool
result names what it made; `openmirror://media/{id}` hands over the bytes to a
client that wants them. Inlining a 4MB PNG into a tool result costs the client's
context window whether or not it wanted to look.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from openmirror.mcp.client import PROTOCOL_VERSION

log = logging.getLogger(__name__)

SERVER_NAME = 'openmirror'
SERVER_VERSION = '0.1.0'

# JSON-RPC's own codes. Only the three that can actually happen here.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# What each scope may reach. Ordered, and checked by index, so `write` implies
# `read` without the sets having to repeat themselves.
SCOPES = ('read', 'write', 'all')


def _scope_allows(have: str, need: str) -> bool:
    """Whether a server running at `have` may expose something needing `need`.

    An unrecognised scope is read — the narrowest — because a typo in a config
    file must not be a way to widen what is exposed. Read rather than nothing,
    though: a server that offers no tools at all reads as a broken install, and
    somebody who misspelt `write` would go looking for the wrong fault. The
    warning that says so is logged once, where the server is built.
    """
    if have not in SCOPES:
        have = 'read'
    return SCOPES.index(have) >= SCOPES.index(need)


class ProtocolError(Exception):
    """A JSON-RPC level failure — a bad method, bad params.

    Distinct from a tool that ran and failed: that comes back as a result with
    `isError` set, which is what the protocol asks for and what lets a model
    read the failure and try something else.
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


Handler = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(slots=True)
class Exposed:
    """One tool this server offers."""

    name: str
    description: str
    schema: dict[str, Any]
    handler: Handler
    scope: str = 'read'
    # Hints, which a client may or may not believe — openmirror's own client does
    # not, and says why in `agent/tools/mcp.py`. Declared honestly anyway: the
    # point of being distrusted is that honesty still costs nothing.
    read_only: bool = True
    destructive: bool = False
    # Said in the description when true, because a client cannot see a price.
    spends: bool = False


@dataclass(slots=True)
class Resource:
    uri: str
    name: str
    description: str
    mime_type: str
    reader: Callable[[dict[str, str]], Awaitable[dict[str, Any]]]
    # A URI with `{placeholders}`, listed separately by the protocol.
    template: bool = False
    scope: str = 'read'


@dataclass(slots=True)
class Prompt:
    name: str
    description: str
    arguments: list[dict[str, Any]]
    render: Callable[[dict[str, str]], str]


@dataclass
class OpenmirrorMCP:
    """The server, with no transport attached.

    Transport-free on purpose: the same object answers a pipe, an HTTP POST and
    an SSE session, and a bug fixed in one is fixed in all three. The transports
    below are thirty lines each because everything that decides anything is
    here.
    """

    scope: str = 'read'
    # Whatever the install already built. None for anything this process has
    # not got — a server with no memory service simply does not offer `recall`,
    # rather than offering it and failing.
    memory: Any = None
    media: Any = None
    registry: Any = None
    user_id: str = 'local'
    web: bool = True
    search_backend: str = 'headless'
    search_key: str = ''
    search_url: str = ''
    search_engine: str = 'auto'
    headless: bool = True
    web_allow_private: bool = False

    _web_tools: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    # -- what is on offer ---------------------------------------------------

    def tools(self) -> list[Exposed]:
        """Every tool this install can actually back, filtered by scope.

        Built per call rather than cached because what an install can do
        changes while it runs — a provider arriving gives it image generation,
        and a tool list built once at start-up would never mention it.
        """
        out: list[Exposed] = []

        if self.memory is not None:
            out.append(Exposed(
                name='recall',
                description=(
                    'Search what openmirror remembers about the person it belongs to — their '
                    'preferences, their projects, how they work, what they have told it. Use '
                    'this before answering anything personal, and before guessing.'
                ),
                schema={
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string', 'description': 'What you want to know.'},
                        'limit': {'type': 'integer', 'description': 'How many memories. Default 5.'},
                        'subject': {
                            'type': 'string',
                            'description': 'Narrow to one project or topic, if you know it.',
                        },
                    },
                    'required': ['query'],
                },
                handler=self._recall,
            ))
            out.append(Exposed(
                name='remember',
                description=(
                    'Store one durable fact about the person — something that will still be '
                    'true next week. Not a passing detail, and not something they only said '
                    'once in a conversation about something else.'
                ),
                schema={
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string', 'description': 'The fact, in one sentence.'},
                        'subject': {'type': 'string', 'description': 'What it is about, if not them.'},
                        'kind': {
                            'type': 'string',
                            'enum': ['fact', 'preference', 'episode'],
                            'description': 'Default fact.',
                        },
                    },
                    'required': ['text'],
                },
                handler=self._remember,
                scope='write',
                read_only=False,
            ))

        if self.web:
            out.append(Exposed(
                name='web_search',
                description=(
                    "Search the web through openmirror's configured backend and get titles, URLs "
                    'and snippets back. Follow up with web_fetch to read a result properly.'
                ),
                schema={
                    'type': 'object',
                    'properties': {
                        'query': {'type': 'string'},
                        'count': {'type': 'integer', 'description': 'Default 8, max 20.'},
                    },
                    'required': ['query'],
                },
                handler=self._web_search,
            ))
            out.append(Exposed(
                name='web_fetch',
                description=(
                    'Fetch a URL and return its text. Local and private addresses are refused '
                    'unless this install allows them.'
                ),
                schema={
                    'type': 'object',
                    'properties': {
                        'url': {'type': 'string'},
                        'max_chars': {'type': 'integer', 'description': 'Default 40000.'},
                    },
                    'required': ['url'],
                },
                handler=self._web_fetch,
            ))
            out.append(Exposed(
                name='research',
                description=(
                    'Look something up properly: several searches, the most promising pages '
                    'read, and the relevant text returned with its source next to each block. '
                    'Use this instead of web_search when you need an answer, not a list of links.'
                ),
                schema={
                    'type': 'object',
                    'properties': {
                        'question': {'type': 'string', 'description': 'What you are trying to find out.'},
                        'queries': {
                            'type': 'array', 'items': {'type': 'string'},
                            'description': 'Up to 5 differently-worded searches.',
                        },
                        'pages': {'type': 'integer', 'description': 'How many to read. Default 4, max 8.'},
                        'urls': {
                            'type': 'array', 'items': {'type': 'string'},
                            'description': 'Read these too, whatever the search finds.',
                        },
                    },
                    'required': ['question'],
                },
                handler=self._research,
            ))

        if self.media is not None and 'image' in self.media.can_generate():
            out.append(Exposed(
                name='generate_image',
                description=(
                    'Make an image with the provider this install is routed to. THIS SPENDS '
                    'MONEY unless that provider is local. The image is saved here; the result '
                    'names it and openmirror://media/{id} hands over the bytes.'
                ),
                schema={
                    'type': 'object',
                    'properties': {
                        'prompt': {'type': 'string'},
                        'provider': {'type': 'string', 'description': "Override the install's route."},
                        'model': {'type': 'string'},
                    },
                    'required': ['prompt'],
                },
                handler=self._generate_image,
                scope='all',
                read_only=False,
                spends=True,
            ))

        return [t for t in out if _scope_allows(self.scope, t.scope)]

    def resources(self) -> list[Resource]:
        out = [
            Resource(
                uri='openmirror://capabilities',
                name='What this openmirror can do',
                description=(
                    'Which modalities are routed, which tools this server exposes, and whether '
                    'memory is on. Read this first: it is the difference between asking for '
                    'something this install can do and guessing.'
                ),
                mime_type='application/json',
                reader=self._read_capabilities,
            ),
        ]
        if self.registry is not None:
            out.append(Resource(
                uri='openmirror://providers',
                name='Configured providers',
                description='Every registered provider, what it serves, and whether it runs locally.',
                mime_type='application/json',
                reader=self._read_providers,
            ))
        if self.memory is not None:
            out.append(Resource(
                uri='openmirror://memory/recent',
                name='Recently remembered',
                description='The most recent memories, newest first. Empty when memory is off.',
                mime_type='text/plain',
                reader=self._read_recent_memory,
            ))
            out.append(Resource(
                uri='openmirror://memory/search/{query}',
                name='Memory search',
                description='The same search `recall` runs, addressed as a URI.',
                mime_type='text/plain',
                reader=self._read_memory_search,
                template=True,
            ))
        if self.media is not None:
            out.append(Resource(
                uri='openmirror://media/{id}',
                name='Generated media',
                description='The bytes of something openmirror generated, by its media id.',
                mime_type='application/octet-stream',
                reader=self._read_media,
                template=True,
            ))
        return [r for r in out if _scope_allows(self.scope, r.scope)]

    def prompts(self) -> list[Prompt]:
        """Two, and both exist to stop a client guessing at the person.

        A prompt is the server saying "here is how to use me well". The failure
        they prevent is a model that has a `recall` tool, does not call it, and
        answers a question about someone it has never asked about.
        """
        return [
            Prompt(
                name='ask_the_twin',
                description="Answer a question about this person using openmirror's memory, not guesses.",
                arguments=[{'name': 'question', 'description': 'The question.', 'required': True}],
                render=lambda a: (
                    f'Answer this about the person openmirror belongs to: {a.get("question", "")}\n\n'
                    'Call recall first, with two or three differently-worded queries — what is '
                    'remembered is written in their words, not yours. If recall comes back empty, '
                    'say that nothing is remembered about it rather than answering from a general '
                    'impression: a confident guess about someone is worse than an admission. '
                    'Where the answer depends on something current, use research and say which '
                    'part came from the web.'
                ),
            ),
            Prompt(
                name='research_brief',
                description='Research a topic through this install\'s search backend and report with sources.',
                arguments=[
                    {'name': 'topic', 'description': 'What to look into.', 'required': True},
                    {'name': 'angle', 'description': 'What specifically matters about it.', 'required': False},
                ],
                render=lambda a: (
                    f'Research: {a.get("topic", "")}\n'
                    + (f'What matters about it: {a.get("angle")}\n' if a.get('angle') else '')
                    + '\nUse the research tool with two or three differently-worded queries rather '
                    'than one. Keep each claim next to the URL it came from — a bibliography at '
                    'the end loses which claim came from where. Say plainly what you could not '
                    'find; a gap reported is useful and a gap filled in is not.'
                ),
            ),
        ]

    # -- the protocol -------------------------------------------------------

    async def handle(self, message: Any) -> dict[str, Any] | None:
        """One JSON-RPC message in, at most one out.

        None means "nothing to send": a notification, which the protocol says
        gets no reply. Returning an empty reply to one is a common enough bug
        that it is worth a sentence — a client correlating by id hangs on it.
        """
        if not isinstance(message, dict) or message.get('jsonrpc') not in (None, '2.0'):
            return self._failure(None, INVALID_REQUEST, 'not a JSON-RPC 2.0 message')

        method = message.get('method')
        msg_id = message.get('id')
        params = message.get('params') or {}
        if not isinstance(method, str):
            return self._failure(msg_id, INVALID_REQUEST, 'no method')
        if msg_id is None:
            # A notification. `notifications/initialized` is the only one that
            # matters and nothing is owed in return for it.
            log.debug('mcp server: notification %s', method)
            return None

        try:
            result = await self._dispatch(method, params if isinstance(params, dict) else {})
        except ProtocolError as exc:
            return self._failure(msg_id, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001 - a crash must not kill the transport
            log.exception('mcp server: %s failed', method)
            return self._failure(msg_id, INTERNAL_ERROR, f'{type(exc).__name__}: {exc}')
        return {'jsonrpc': '2.0', 'id': msg_id, 'result': result}

    def _failure(self, msg_id: Any, code: int, message: str) -> dict[str, Any]:
        return {'jsonrpc': '2.0', 'id': msg_id, 'error': {'code': code, 'message': message}}

    async def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == 'initialize':
            return {
                'protocolVersion': params.get('protocolVersion') or PROTOCOL_VERSION,
                'capabilities': {
                    'tools': {'listChanged': False},
                    'resources': {'subscribe': False, 'listChanged': False},
                    'prompts': {'listChanged': False},
                },
                'serverInfo': {'name': SERVER_NAME, 'version': SERVER_VERSION},
                'instructions': (
                    'This is one person\'s openmirror: their memory, their provider routing, their '
                    'search backend. Read openmirror://capabilities to see what this install can '
                    'do. For anything about the person, call recall before answering.'
                ),
            }
        if method == 'ping':
            return {}
        if method == 'tools/list':
            return {'tools': [self._tool_json(t) for t in self.tools()]}
        if method == 'tools/call':
            return await self._call_tool(params)
        if method == 'resources/list':
            return {'resources': [self._resource_json(r) for r in self.resources() if not r.template]}
        if method == 'resources/templates/list':
            return {
                'resourceTemplates': [
                    self._resource_json(r, template=True) for r in self.resources() if r.template
                ]
            }
        if method == 'resources/read':
            return await self._read_resource(params)
        if method == 'prompts/list':
            return {
                'prompts': [
                    {'name': p.name, 'description': p.description, 'arguments': p.arguments}
                    for p in self.prompts()
                ]
            }
        if method == 'prompts/get':
            return self._get_prompt(params)
        raise ProtocolError(METHOD_NOT_FOUND, f'no method {method!r}')

    def _tool_json(self, tool: Exposed) -> dict[str, Any]:
        description = tool.description
        if tool.spends:
            description += '\n\nThis can cost money. Ask before calling it.'
        return {
            'name': tool.name,
            'description': description,
            'inputSchema': tool.schema,
            'annotations': {
                'readOnlyHint': tool.read_only,
                'destructiveHint': tool.destructive,
                'openWorldHint': tool.name in ('web_search', 'web_fetch', 'research'),
            },
        }

    def _resource_json(self, resource: Resource, *, template: bool = False) -> dict[str, Any]:
        key = 'uriTemplate' if template else 'uri'
        return {
            key: resource.uri,
            'name': resource.name,
            'description': resource.description,
            'mimeType': resource.mime_type,
        }

    async def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get('name')
        args = params.get('arguments') or {}
        if not isinstance(name, str):
            raise ProtocolError(INVALID_PARAMS, 'tools/call needs a name')
        if not isinstance(args, dict):
            raise ProtocolError(INVALID_PARAMS, 'arguments must be an object')

        tool = next((t for t in self.tools() if t.name == name), None)
        if tool is None:
            # Said precisely: a tool missing because of scope is a different
            # problem from a tool that does not exist, and a client told the
            # first can ask its operator rather than retrying for ever.
            blocked = next((t for t in self._all_tools() if t.name == name), None)
            if blocked is not None:
                raise ProtocolError(
                    METHOD_NOT_FOUND,
                    f'{name} needs the {blocked.scope!r} scope and this server is running at '
                    f'{self.scope!r} (OPENMIRROR_MCP_SERVE_SCOPE)',
                )
            raise ProtocolError(METHOD_NOT_FOUND, f'no tool named {name!r}')

        try:
            result = await tool.handler(args)
        except Exception as exc:  # noqa: BLE001 - a failed tool is a result
            # The protocol's own convention: a tool that ran and failed reports
            # it in the content, so the model reads the reason and adapts.
            # Raising would tell the model only that something broke.
            log.info('mcp server: %s failed: %s', name, exc)
            return {
                'content': [{'type': 'text', 'text': f'{name} failed: {exc}'}],
                'isError': True,
            }
        if isinstance(result, dict) and 'content' in result:
            return result
        return {'content': [{'type': 'text', 'text': str(result)}], 'isError': False}

    def _all_tools(self) -> list[Exposed]:
        """Every tool regardless of scope, for telling "not allowed" from "not a thing"."""
        keep, self.scope = self.scope, SCOPES[-1]
        try:
            return self.tools()
        finally:
            self.scope = keep

    async def _read_resource(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get('uri')
        if not isinstance(uri, str) or not uri:
            raise ProtocolError(INVALID_PARAMS, 'resources/read needs a uri')

        for resource in self.resources():
            bindings = _match_uri(resource.uri, uri)
            if bindings is None:
                continue
            content = await resource.reader(bindings)
            return {'contents': [{'uri': uri, **content}]}
        raise ProtocolError(INVALID_PARAMS, f'no resource at {uri}')

    def _get_prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get('name')
        args = params.get('arguments') or {}
        prompt = next((p for p in self.prompts() if p.name == name), None)
        if prompt is None:
            raise ProtocolError(INVALID_PARAMS, f'no prompt named {name!r}')
        missing = [
            a['name'] for a in prompt.arguments
            if a.get('required') and not str(args.get(a['name'], '')).strip()
        ]
        if missing:
            raise ProtocolError(INVALID_PARAMS, f'{name} needs: {", ".join(missing)}')
        return {
            'description': prompt.description,
            'messages': [{
                'role': 'user',
                'content': {'type': 'text', 'text': prompt.render({k: str(v) for k, v in args.items()})},
            }],
        }

    # -- the tools themselves ------------------------------------------------

    def _tool(self, which: str) -> Any:
        """The agent's own web tools, built once and shared.

        The agent's, not a second implementation: one set of keys, one
        private-address guard, and no way for this server to reach somewhere
        `web_fetch` would refuse.
        """
        if not self._web_tools:
            from openmirror.agent.tools.research import ResearchTool
            from openmirror.agent.tools.web import WebFetchTool, WebSearchTool

            fetch = WebFetchTool(allow_private=self.web_allow_private)
            search = WebSearchTool(
                backend=self.search_backend,
                api_key=self.search_key,
                base_url=self.search_url,
                engine=self.search_engine,
                headless=self.headless,
            )
            self._web_tools = {
                'fetch': fetch,
                'search': search,
                'research': ResearchTool(search, fetch),
            }
        return self._web_tools[which]

    def _context(self) -> Any:
        """A tool context that can reach the network and nothing else.

        No checkpoint, no ask, a root it never writes to. The web tools do not
        touch the filesystem, and building the context explicitly rather than
        borrowing a session's is what keeps that true when one of them changes.
        """
        from openmirror.agent.tools.base import ToolContext
        from openmirror.config import config

        async def emit(text: str, stream: str) -> None:
            return None

        async def ask(question: str, options: list[str], multi: bool) -> str:
            # Nobody is watching an MCP call. A tool that asked would hang.
            return ''

        return ToolContext(
            root=config.workspace, cwd=config.workspace,
            emit=emit, ask=ask, session_id='mcp-server',
        )

    async def _recall(self, args: dict[str, Any]) -> str:
        query = str(args.get('query') or '').strip()
        if not query:
            raise ValueError('query is required')
        limit = max(1, min(int(args.get('limit') or 5), 25))
        subject = str(args.get('subject') or '').strip() or None

        recall = await self.memory.recall(self.user_id, query, limit=limit, subject=subject)
        if not recall.memories:
            return recall.reason or f'Nothing remembered about {query!r}.'
        lines = [
            f'- {m.text}'
            + (f' (about {m.subject})' if m.subject else '')
            + f' [{m.kind}, {time.strftime("%Y-%m-%d", time.localtime(m.created_at))}]'
            for m in recall.memories
        ]
        return f'{len(lines)} memory/memories for {query!r}:\n' + '\n'.join(lines)

    async def _remember(self, args: dict[str, Any]) -> str:
        text = str(args.get('text') or '').strip()
        if not text:
            raise ValueError('text is required')
        kind = str(args.get('kind') or 'fact')
        subject = str(args.get('subject') or '').strip() or None
        memory = await self.memory.remember(
            self.user_id, text, kind=kind, subject=subject, source='mcp',
        )
        if memory is None:
            return (
                'Not stored: memory is switched off for this user, or the text was empty. '
                'Nothing was written.'
            )
        return f'Remembered: {memory.text}'

    async def _web_search(self, args: dict[str, Any]) -> str:
        query = str(args.get('query') or '').strip()
        if not query:
            raise ValueError('query is required')
        count = max(1, min(int(args.get('count') or 8), 20))
        results = await self._tool('search').search(query, count)
        if not results:
            return f'No results for {query!r}.'
        return '\n\n'.join(
            f'{i}. {r["title"]}\n   {r["url"]}\n   {r.get("snippet", "")}'.rstrip()
            for i, r in enumerate(results, 1)
        )

    async def _web_fetch(self, args: dict[str, Any]) -> str:
        output = await self._tool('fetch').run(
            {'url': str(args.get('url') or ''), 'max_chars': args.get('max_chars')},
            self._context(),
        )
        return output.content

    async def _research(self, args: dict[str, Any]) -> str:
        output = await self._tool('research').run(dict(args), self._context())
        return output.content

    async def _generate_image(self, args: dict[str, Any]) -> dict[str, Any]:
        prompt = str(args.get('prompt') or '').strip()
        if not prompt:
            raise ValueError('prompt is required')
        result = await self.media.generate_image(
            prompt,
            provider=str(args.get('provider') or '') or None,
            model=str(args.get('model') or '') or None,
        )
        made = result.get('media') or []
        lines = [f'Generated {len(made)} image(s) with {result.get("provider")} ({result.get("model")}).']
        for item in made:
            kind = item.get('media_type') or 'image'
            lines.append(f'- openmirror://media/{item.get("id")}  [{kind}, {item.get("bytes", 0)} bytes]')
        return {
            'content': [{'type': 'text', 'text': '\n'.join(lines)}],
            'isError': False,
        }

    # -- the resources themselves -------------------------------------------

    async def _read_capabilities(self, _: dict[str, str]) -> dict[str, Any]:
        from openmirror.config import config

        body: dict[str, Any] = {
            'server': {'name': SERVER_NAME, 'version': SERVER_VERSION, 'scope': self.scope},
            'tools': [t.name for t in self.tools()],
            'memory': {
                'available': self.memory is not None,
                'enabled': bool(
                    self.memory is not None and self.memory.settings(self.user_id).enabled
                ),
                'embedder': self.memory.embedder() if self.memory is not None else '',
            },
            'web': {
                'available': self.web,
                'backend': self.search_backend,
                'engine': self.search_engine if self.search_backend == 'headless' else '',
            },
            'media': {
                'available': self.media is not None,
                'can_generate': sorted(self.media.can_generate()) if self.media is not None else [],
            },
            # Said plainly, because a client that knows these are absent stops
            # asking for them.
            'not_exposed': [
                'shell', 'file editing', 'the desktop', 'browser control',
                'every tool openmirror puts behind a human approval',
            ],
            'workspace': str(config.workspace),
        }
        if self.registry is not None:
            body['routed'] = self.registry.capabilities()
        return {'mimeType': 'application/json', 'text': json.dumps(body, indent=2, default=str)}

    async def _read_providers(self, _: dict[str, str]) -> dict[str, Any]:
        rows = [self.registry.describe(info.id) for info in self.registry.list()]
        return {'mimeType': 'application/json', 'text': json.dumps(rows, indent=2, default=str)}

    async def _read_recent_memory(self, _: dict[str, str]) -> dict[str, Any]:
        if not self.memory.settings(self.user_id).enabled:
            return {'mimeType': 'text/plain', 'text': 'Memory is switched off for this user.'}
        rows = self.memory.store.list(self.user_id, limit=50)
        if not rows:
            return {'mimeType': 'text/plain', 'text': 'Nothing remembered yet.'}
        lines = [
            f'{time.strftime("%Y-%m-%d", time.localtime(m.created_at))}  [{m.kind}] {m.text}'
            + (f'  (about {m.subject})' if m.subject else '')
            for m in rows
        ]
        return {'mimeType': 'text/plain', 'text': '\n'.join(lines)}

    async def _read_memory_search(self, bindings: dict[str, str]) -> dict[str, Any]:
        text = await self._recall({'query': bindings.get('query', ''), 'limit': 10})
        return {'mimeType': 'text/plain', 'text': text}

    async def _read_media(self, bindings: dict[str, str]) -> dict[str, Any]:
        """The bytes of one generated file, base64 as the protocol wants them.

        The store resolves the id to a path; this never builds one from the
        URI. An id that is not in the store has no path, which is the whole
        traversal defence — `{id}` already cannot contain a slash, and even if
        it could there is nothing here that would concatenate it onto a
        directory.
        """
        media_id = bindings.get('id', '')
        record = self.media.store.get(media_id)
        path = self.media.store.path(media_id)
        if record is None or path is None:
            raise ProtocolError(INVALID_PARAMS, f'no media with id {media_id!r}')
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ProtocolError(INTERNAL_ERROR, f'could not read {media_id}: {exc}') from exc
        return {
            'mimeType': record.media_type or 'application/octet-stream',
            'blob': base64.b64encode(data).decode(),
        }


def _match_uri(pattern: str, uri: str) -> dict[str, str] | None:
    """Match a URI against a resource pattern, returning its `{placeholders}`.

    Written out rather than compiled to a regex because the patterns are three
    segments long and a regex built from user-ish text is a way to turn a typo
    into a catastrophic backtrack. `{id}` matches one path segment; nothing
    matches across a `/`, so `openmirror://media/{id}` cannot be talked into
    reading `openmirror://media/../secrets`.
    """
    if '{' not in pattern:
        return {} if pattern == uri else None

    # Split the pattern into literals and names: 'a{x}b' -> ['a', '{x}', 'b'].
    parts: list[str] = []
    rest = pattern
    while '{' in rest:
        head, _, tail = rest.partition('{')
        name, closed, tail = tail.partition('}')
        if not closed:
            return None
        if head:
            parts.append(head)
        parts.append('{' + name + '}')
        rest = tail
    if rest:
        parts.append(rest)

    bindings: dict[str, str] = {}
    position = 0
    for index, part in enumerate(parts):
        if not part.startswith('{'):
            if not uri.startswith(part, position):
                return None
            position += len(part)
            continue
        name = part[1:-1]
        following = parts[index + 1] if index + 1 < len(parts) else ''
        if following:
            end = uri.find(following, position)
            if end == -1:
                return None
        else:
            end = len(uri)
        value = uri[position:end]
        if not value or '/' in value:
            return None
        bindings[name] = value
        position = end
    return bindings if position == len(uri) else None


# ---------------------------------------------------------------------------
# Building one from this install's configuration
# ---------------------------------------------------------------------------


async def from_config(*, app: Any = None) -> OpenmirrorMCP:
    """The server this install should expose.

    Takes the running app's services when there is one — the HTTP endpoint
    shares the daemon's memory store and media service rather than opening a
    second handle on the same SQLite file — and builds what it needs when there
    is not, which is the stdio case.
    """
    from openmirror.config import config

    memory = getattr(app.state, 'memory', None) if app is not None else None
    media = getattr(app.state, 'media', None) if app is not None else None
    registry = None

    if app is not None:
        from openmirror.providers.registry import registry as live

        registry = live
    else:
        from openmirror.providers.bootstrap import bootstrap
        from openmirror.providers.registry import registry as live

        # A standalone stdio server still needs providers: recall embeds its
        # query, and an embedding is a provider call.
        await bootstrap(config, live)
        registry = live
        if config.memory_enabled:
            from openmirror.memory.service import MemoryService
            from openmirror.memory.store import MemoryStore

            memory = MemoryService(
                MemoryStore(config.memory_db), live, model=config.embed_model
            )

    scope = config.mcp_serve_scope
    if scope not in SCOPES:
        # Said once, loudly, at the only moment anyone can act on it. The
        # comparison falls back on its own; this is what tells a person why
        # their `write` tools are missing.
        log.warning(
            'OPENMIRROR_MCP_SERVE_SCOPE is %r, which is not one of %s. Serving at "read".',
            scope, ', '.join(SCOPES),
        )
        scope = 'read'

    return OpenmirrorMCP(
        scope=scope,
        memory=memory,
        media=media,
        registry=registry,
        user_id=config.default_user,
        web=config.web_enabled,
        search_backend=config.search_backend,
        search_key=config.search_key,
        search_url=config.search_url,
        search_engine=config.search_engine,
        headless=config.browser_headless,
        web_allow_private=config.web_allow_private,
    )


# ---------------------------------------------------------------------------
# The stdio transport
# ---------------------------------------------------------------------------


async def serve_stdio(server: OpenmirrorMCP | None = None) -> None:
    """Speak the protocol over stdin and stdout until stdin closes.

    The one rule of a stdio MCP server: **nothing but protocol on stdout.** A
    stray `print` corrupts the stream and the client's error says only that the
    JSON was invalid, which is a miserable thing to debug. Logging is pinned to
    stderr at the bottom of this file for exactly that reason.
    """
    if server is None:
        server = await from_config()

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break
        raw = line.strip()
        if not raw:
            continue
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            _write_stdout({'jsonrpc': '2.0', 'id': None,
                           'error': {'code': PARSE_ERROR, 'message': 'invalid JSON'}})
            continue

        # A batch is a list, and the protocol allows one.
        if isinstance(message, list):
            for item in message:
                reply = await server.handle(item)
                if reply is not None:
                    _write_stdout(reply)
            continue

        reply = await server.handle(message)
        if reply is not None:
            _write_stdout(reply)

    from openmirror.agent.headless_search import shutdown

    await shutdown()


def _write_stdout(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + '\n')
    sys.stdout.flush()


def main() -> None:
    """`openmirror-mcp` — this install as an MCP server on stdio."""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format='%(levelname)-7s openmirror-mcp: %(message)s',
    )
    try:
        asyncio.run(serve_stdio())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
