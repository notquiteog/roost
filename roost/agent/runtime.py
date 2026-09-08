"""Putting a session together.

One function, because assembling an agent is where the mistakes happen: a
session built with the wrong root, or with tools its policy will refuse
everything from, fails in ways that look like the model being stupid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from roost.agent import prompt as prompt_mod
from roost.agent.approval import ApprovalPolicy, Mode
from roost.agent.session import AgentSession
from roost.agent.tools.ask import AskUserTool
from roost.agent.tools.base import Tool
from roost.agent.tools.files import EditTool, ListDirTool, ReadTool, WriteTool
from roost.agent.tools.search import GlobTool, GrepTool
from roost.agent.tools.shell import ShellTool


def default_tools() -> list[Tool]:
    return [
        ReadTool(),
        WriteTool(),
        EditTool(),
        ListDirTool(),
        GlobTool(),
        GrepTool(),
        ShellTool(),
        AskUserTool(),
    ]


def web_tools(cfg: Any) -> list[Tool]:
    from roost.agent.tools.web import WebFetchTool, WebSearchTool

    return [
        WebFetchTool(allow_private=cfg.web_allow_private),
        WebSearchTool(backend=cfg.search_backend, api_key=cfg.search_key, base_url=cfg.search_url),
    ]


def mcp_tools(manager: Any) -> list[Tool]:
    from roost.agent.tools.mcp import mcp_tools as build

    return build(manager)


def desktop_tools() -> list[Tool]:
    from roost.agent.tools.desktop import desktop_tools as build

    return build()


def browser_tools(browser: Any) -> list[Tool]:
    from roost.agent.tools.browser import (
        BrowserAskHumanTool,
        BrowserClickTool,
        BrowserNavigateTool,
        BrowserReadTool,
        BrowserScreenshotTool,
        BrowserTypeTool,
    )

    return [
        BrowserNavigateTool(browser),
        BrowserReadTool(browser),
        BrowserClickTool(browser),
        BrowserTypeTool(browser),
        BrowserScreenshotTool(browser),
        BrowserAskHumanTool(browser),
    ]


def build_session(
    *,
    root: str | Path,
    provider: Any,
    model: str,
    mode: Mode | str = Mode.ASK,
    tools: list[Tool] | None = None,
    session_id: str | None = None,
    env: dict[str, str] | None = None,
    extra_prompt: str = '',
    title: str = '',
    memory: Any = None,
    user_id: str = '',
    confined: bool = True,
    allow_purchases: bool = False,
    allow_credentials: bool = False,
    web: Any = None,
    browser: Any = None,
    desktop: bool = False,
    mcp: Any = None,
    checkpoints: Any = None,
) -> AgentSession:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f'{root_path}: working root is not a directory')

    policy = ApprovalPolicy(
        mode=Mode(mode),
        allow_purchases=allow_purchases,
        allow_credentials=allow_credentials,
    )
    chosen = tools if tools is not None else default_tools()

    # In read-only mode the tools that can change things are left out entirely
    # rather than registered and refused. A model that can see `write_file`
    # will keep proposing it and spend the turn being told no.
    if policy.mode is Mode.READ_ONLY:
        chosen = [t for t in chosen if t.name not in {'write_file', 'edit_file'}]

    # Memory is added as tools and as a per-turn hook, and only when this
    # person has switched it on — so a model in a session without memory is
    # never shown a `remember` tool it will be refused for using.
    context_provider = None
    on_user_text = None
    if memory is not None and user_id and memory.settings(user_id).enabled:
        from roost.agent.tools.memory import RecallTool, RememberTool

        chosen = [*chosen, RememberTool(memory, user_id), RecallTool(memory, user_id)]

        async def context_provider(text: str) -> str:  # noqa: F811
            return await memory.context_for(user_id, text)

        async def on_user_text(text: str) -> None:  # noqa: F811
            await memory.capture(user_id, text, source='chat')

    if web is not None:
        chosen = [*chosen, *web_tools(web)]
    if browser is not None:
        chosen = [*chosen, *browser_tools(browser)]
    if desktop:
        chosen = [*chosen, *desktop_tools()]
    if mcp is not None:
        chosen = [*chosen, *mcp_tools(mcp)]

    context = prompt_mod.project_context(root_path)
    if extra_prompt:
        context = f'{context}\n\n{extra_prompt}' if context else extra_prompt

    return AgentSession(
        session_id=session_id,
        root=root_path,
        provider=provider,
        model=model,
        tools=chosen,
        policy=policy,
        system_prompt=prompt_mod.build(
            root_path, policy=policy.describe(), extra=context, confined=confined
        ),
        env=env,
        title=title,
        context_provider=context_provider,
        on_user_text=on_user_text,
        confined=confined,
        browser=browser,
        checkpoints=checkpoints,
    )
