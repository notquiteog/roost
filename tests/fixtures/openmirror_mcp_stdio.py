#!/usr/bin/env python3
"""openmirror's own MCP server on stdio, with fakes behind it.

The real entrypoint bootstraps providers and opens a memory database, neither
of which a test wants. This runs the same `serve_stdio` over the same
`OpenmirrorMCP` object with a memory service that answers from a list — so what
is tested is the transport and the protocol, which is what a subprocess is for.
"""
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openmirror.mcp.server import OpenmirrorMCP, serve_stdio  # noqa: E402


@dataclass
class Memory:
    text: str
    kind: str = 'fact'
    subject: str | None = None
    created_at: float = 1_700_000_000.0


@dataclass
class Recall:
    memories: list
    reason: str = ''


@dataclass
class Settings:
    enabled: bool = True


@dataclass
class FakeStore:
    rows: list = field(default_factory=list)

    def list(self, user_id, *, limit=200, **kw):
        return self.rows[:limit]


@dataclass
class FakeMemory:
    rows: list = field(default_factory=lambda: [Memory('they take their coffee black')])
    store: FakeStore = field(default_factory=FakeStore)

    def __post_init__(self):
        self.store.rows = self.rows

    def settings(self, user_id):
        return Settings(enabled=True)

    def embedder(self):
        return 'fake-embed'

    async def recall(self, user_id, query, *, limit=5, subject=None, kind=None):
        hits = [m for m in self.rows if any(w in m.text for w in query.lower().split())]
        return Recall(memories=hits[:limit])

    async def remember(self, user_id, text, *, kind='fact', subject=None, source=None):
        memory = Memory(text=text, kind=kind, subject=subject)
        self.rows.append(memory)
        return memory


if __name__ == '__main__':
    server = OpenmirrorMCP(scope='write', memory=FakeMemory(), web=False, user_id='tester')
    asyncio.run(serve_stdio(server))
