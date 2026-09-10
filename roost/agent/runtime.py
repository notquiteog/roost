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
from roost.agent.tools.code import (
    ApplyPatchTool,
    MultiEditTool,
    OutlineTool,
    PlanTool,
    ReadFilesTool,
)
from roost.agent.tools.files import EditTool, ListDirTool, ReadTool, WriteTool
from roost.agent.tools.search import GlobTool, GrepTool
from roost.agent.tools.shell import ShellTool

# What each named toolset contains. Groups rather than individual names,
# because the choice a person actually makes is "this session is for browsing"
# rather than a list of eleven function names.
#
# This exists because of a measured failure, not a preference. gemma4:12b —
# a model this project explicitly targets — was given the full set of thirty
# and could not complete a five-step browser task: it opened the page, then
# lost the thread and reported that it had no way to browse. The same model,
# the same task, the same prompt, with only the browser tools: done in
# twenty-six seconds, correctly, first try. The tools were not the problem.
# The *list* was.
TOOLSETS: dict[str, tuple[str, ...]] = {
    'files': ('read_file', 'read_files', 'write_file', 'edit_file', 'multi_edit',
              'apply_patch', 'outline', 'list_dir', 'glob', 'grep'),
    'shell': ('shell',),
    'plan': ('plan',),
    'ask': ('ask_user',),
    'web': ('web_search', 'web_fetch', 'research'),
    'browser': ('browser_navigate', 'browser_read', 'browser_click', 'browser_type',
                'browser_screenshot', 'browser_hand_over'),
    'desktop': ('desktop_screenshot', 'desktop_click', 'desktop_type', 'desktop_key',
                'desktop_scroll'),
    'system': ('system_info', 'package_search', 'package_install', 'package_remove',
               'display_info', 'display_hdr'),
    'media': ('media_params', 'generate_image', 'generate_video', 'media_job', 'import_workflow'),
    'memory': ('remember', 'recall'),
}


def resolve_toolset(names: list[str]) -> set[str] | None:
    """Turn what a caller asked for into a set of tool names, or None for all.

    Unknown entries are kept as literal tool names rather than rejected: a
    caller who wants exactly `shell` and `read_file` should be able to say so
    without a group existing for it. `ask_user` is always in, whatever was
    asked for — a session that cannot ask a question is a session that guesses.
    """
    if not names:
        return None
    allowed: set[str] = set(TOOLSETS['ask'])
    for name in names:
        allowed.update(TOOLSETS.get(name, (name,)))
    return allowed


def default_tools() -> list[Tool]:
    return [
        ReadTool(),
        ReadFilesTool(),
        WriteTool(),
        EditTool(),
        MultiEditTool(),
        ApplyPatchTool(),
        OutlineTool(),
        ListDirTool(),
        GlobTool(),
        GrepTool(),
        ShellTool(),
        PlanTool(),
        AskUserTool(),
    ]


def web_tools(cfg: Any) -> list[Tool]:
    from roost.agent.tools.research import ResearchTool
    from roost.agent.tools.web import WebFetchTool, WebSearchTool

    fetch = WebFetchTool(allow_private=cfg.web_allow_private)
    search = WebSearchTool(backend=cfg.search_backend, api_key=cfg.search_key, base_url=cfg.search_url)
    # `research` is given the other two rather than its own: one set of keys,
    # one private-address guard, and no way for the composite to reach
    # somewhere the individual tools would refuse.
    return [fetch, search, ResearchTool(search, fetch)]


def mcp_tools(manager: Any) -> list[Tool]:
    from roost.agent.tools.mcp import mcp_tools as build

    return build(manager)


def system_tools() -> list[Tool]:
    from roost.agent.tools.system import system_tools as build

    return build()


def media_tools(service: Any) -> list[Tool]:
    from roost.agent.tools.media import media_tools as build

    return build(service)


def desktop_tools(stage: Any) -> list[Tool]:
    from roost.agent.tools.desktop import desktop_tools as build

    return build(stage)


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
    # None means "whatever ApprovalPolicy says", which is the only place the
    # default should live. Repeating it here is how these two ended up
    # meaning `False` after the policy changed — and under the new policy
    # `False` is a refusal rather than a prompt, so every session built
    # without an explicit config silently lost the ability to buy anything.
    allow_purchases: bool | None = None,
    allow_credentials: bool | None = None,
    web: Any = None,
    browser: Any = None,
    stage: Any = None,
    media: Any = None,
    system: bool = False,
    toolset: list[str] | None = None,
    mcp: Any = None,
    checkpoints: Any = None,
) -> AgentSession:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f'{root_path}: working root is not a directory')

    permissions = {
        name: value
        for name, value in (
            ('allow_purchases', allow_purchases),
            ('allow_credentials', allow_credentials),
        )
        if value is not None
    }
    policy = ApprovalPolicy(mode=Mode(mode), **permissions)
    chosen = tools if tools is not None else default_tools()

    # In read-only mode the tools that can change things are left out entirely
    # rather than registered and refused. A model that can see `write_file`
    # will keep proposing it and spend the turn being told no.
    if policy.mode is Mode.READ_ONLY:
        chosen = [
            t for t in chosen
            if t.name not in {'write_file', 'edit_file', 'multi_edit', 'apply_patch'}
        ]

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
    if stage is not None:
        chosen = [*chosen, *desktop_tools(stage)]
    if media is not None:
        chosen = [*chosen, *media_tools(media)]
    if system:
        chosen = [*chosen, *system_tools()]
    if mcp is not None:
        chosen = [*chosen, *mcp_tools(mcp)]

    # What this session actually has, for the prompt. Derived from the tools
    # that ended up on it rather than from the arguments, so a capability that
    # was asked for and could not be built is not claimed.
    names = {t.name for t in chosen}
    capabilities = [
        name for name, marker in (
            ('browser', 'browser_navigate'),
            ('desktop', 'desktop_screenshot'),
            ('media', 'generate_image'),
            ('system', 'system_info'),
            ('memory', 'recall'),
        )
        if marker in names
    ]

    # Narrowed last, after everything has been added, so a group name means
    # the same thing whichever capabilities happen to be attached.
    allowed = resolve_toolset(toolset or [])
    if allowed is not None:
        chosen = [t for t in chosen if t.name in allowed]

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
            root_path,
            policy=policy.describe(),
            extra=context,
            confined=confined,
            capabilities=capabilities,
        ),
        env=env,
        title=title,
        context_provider=context_provider,
        on_user_text=on_user_text,
        confined=confined,
        browser=browser,
        checkpoints=checkpoints,
        stage=stage,
    )
