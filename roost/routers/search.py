"""Searching the web yourself, without asking an agent to do it.

Small, and worth having for a reason that is not obvious: the agent's search
backend is configured once, with a key, and is often better than whatever the
person would get in a browser tab — a self-hosted SearxNG, or a Brave key that
does not track them. Making it reachable directly means "look this up for me"
does not have to cost a model turn, and a person who only wanted the links
gets the links.

It is the same tool the agent uses, deliberately. A second search path would
be a second set of keys, a second failure mode, and eventually a different
answer to the same question — and the private-address guard on fetching is one
of the few genuine security controls here, so nothing gets a way around it.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from roost.agent.tools.base import ToolContext, ToolError
from roost.agent.tools.web import WebFetchTool, WebSearchTool
from roost.config import config

log = logging.getLogger(__name__)

router = APIRouter(prefix='/api/search')


def _tools() -> tuple[WebSearchTool, WebFetchTool]:
    if not config.web_enabled:
        raise HTTPException(status_code=503, detail='the web is switched off on this install (ROOST_WEB)')
    return (
        WebSearchTool(backend=config.search_backend, api_key=config.search_key, base_url=config.search_url),
        WebFetchTool(allow_private=config.web_allow_private),
    )


async def _noop(text: str, stream: str) -> None:
    return None


async def _noask(question: str, options: list[str], multi: bool) -> str:
    return ''


def _context() -> ToolContext:
    # A tool context with nothing in it that could act on the machine. These
    # two tools only read the network, but constructing the context explicitly
    # rather than reusing a session's is what keeps it that way.
    return ToolContext(
        root=config.workspace, cwd=config.workspace, emit=_noop, ask=_noask, session_id='search'
    )


@router.get('')
async def search(q: str = Query(..., min_length=1), count: int = 8) -> dict[str, Any]:
    tool, _ = _tools()
    try:
        results = await tool.search(q, count)
    except ToolError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {'query': q, 'backend': config.search_backend, 'results': results}


@router.get('/fetch')
async def fetch(url: str = Query(..., min_length=1), max_chars: int = 40_000) -> dict[str, Any]:
    _, tool = _tools()
    try:
        output = await tool.run({'url': url, 'max_chars': max_chars}, _context())
    except ToolError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {'url': url, 'text': output.content, 'truncated': output.truncated}
