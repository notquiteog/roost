"""What a tool is.

Two things here are less obvious than they look.

`assess` is separate from `run` because the approval decision has to be made
from the *arguments*, before anything happens. A tool that only declared a
risk level for itself would force `shell` to be permanently dangerous, and a
human asked to approve every `ls` stops reading the prompts within a day —
which is how a real destructive command gets approved by reflex.

`emit` gives a tool a way to stream while it runs. A four-minute build that
reports nothing until it finishes is indistinguishable from a hung one, and
the model does not need that output — the person watching does.
"""

from __future__ import annotations

import abc
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roost.protocol.agent import Risk

# Called by a tool with (text, stream) as output appears.
Emit = Callable[[str, str], Awaitable[None]]

# Called by a tool with (question, options, multi); resolves when a human
# answers. The loop is suspended for the duration, which is the point: an
# agent that guesses rather than asking is the failure this prevents.
Ask = Callable[[str, list[str], bool], Awaitable[str]]


class ToolError(Exception):
    """A failure the model should see and can act on.

    Distinct from an unexpected exception: this becomes a tool result the
    model reads and retries around, rather than an error that ends the turn.
    """


@dataclass(slots=True)
class Assessment:
    risk: Risk
    summary: str
    # Set when the tool knows the call is malformed. Checked before approval,
    # so a person is never asked to approve something that cannot run.
    invalid: str | None = None


@dataclass(slots=True)
class ToolContext:
    """Everything a tool is allowed to know about where it is running."""

    root: Path
    cwd: Path
    emit: Emit
    ask: Ask
    session_id: str
    # False gives the agent the whole filesystem. It is a declared mode rather
    # than an accident, and the distinction matters: before this existed the
    # file tools were confined and the shell was not, which is the worst of
    # both — a promise of containment that the most powerful tool ignored.
    confined: bool = True
    # Snapshots files before they change, so a turn can be undone. None when
    # checkpointing is off; the tools call it unconditionally and it does
    # nothing, which keeps the null check out of every write path.
    checkpoint: Any = None
    # Set for tools that need to reach a model of their own.
    user_routes: Any = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Output:
    content: str
    display: dict[str, Any] | None = None
    truncated: bool = False
    # Images for the *model*, as (base64, media_type). Separate from `display`
    # on purpose: display is what the UI renders and never reaches the model.
    # Conflating them is how a screenshot tool ends up showing a picture to the
    # human while the model receives only the words "Screenshot taken" — and
    # then describes, confidently, a screen it has never seen.
    images: list[tuple[str, str]] = field(default_factory=list)


class Tool(abc.ABC):
    name: str
    description: str
    input_schema: dict[str, Any]

    @abc.abstractmethod
    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment: ...

    @abc.abstractmethod
    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output: ...


# ---------------------------------------------------------------------------
# Path confinement
# ---------------------------------------------------------------------------


class PathEscape(ToolError):
    pass


def resolve_in_root(candidate: str | Path, ctx: ToolContext) -> Path:
    """Resolve a path, refusing anything outside the session root when confined.

    `Path.resolve()` is what does the work: it collapses `..` and follows
    symlinks, so both `../../etc/passwd` and a symlink planted inside the root
    that points at `/etc` land outside and are caught by the same check.
    Resolving first and comparing after is the only ordering that catches the
    symlink case — a textual check on the string would pass it.
    """
    p = Path(candidate).expanduser()
    if not p.is_absolute():
        p = ctx.cwd / p

    resolved = p.resolve()
    if not ctx.confined:
        return resolved

    root = ctx.root.resolve()
    if resolved != root and root not in resolved.parents:
        raise PathEscape(f'{candidate}: outside the session root ({root})')
    return resolved


def escapes_root(candidate: str, ctx: ToolContext) -> bool:
    """Whether a path string would land outside the root. Never raises.

    For tools that must *grade* a path rather than open one — the shell, which
    cannot be confined but can at least be honest about when it is leaving.
    """
    if not ctx.confined:
        return False
    try:
        resolve_in_root(candidate, ctx)
    except PathEscape:
        return True
    except (OSError, ValueError):
        # Unparseable as a path: not evidence of an escape.
        return False
    return False


def truncate(text: str, limit: int = 30_000, *, keep: str = 'both') -> tuple[str, bool]:
    """Cut output to something a model can read.

    Which end to keep is the caller's choice and it matters: a stack trace is
    useful at the top, a log is useful at the bottom, and a test run is useful
    at both ends and worthless in the middle.
    """
    if len(text) <= limit:
        return text, False

    if keep == 'head':
        return text[:limit] + f'\n\n[... {len(text) - limit} more characters]', True
    if keep == 'tail':
        return f'[... {len(text) - limit} characters ...]\n\n' + text[-limit:], True

    half = limit // 2
    omitted = len(text) - limit
    return f'{text[:half]}\n\n[... {omitted} characters omitted ...]\n\n{text[-half:]}', True
