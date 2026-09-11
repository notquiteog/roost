"""Choosing which browser the agent drives.

Three endpoints, and the shape of them is the argument: reading tells you what
is in force *and where that came from*, writing validates before it saves, and
deleting goes back to the environment's default rather than to a hardcoded one.

A setting that silently falls back is worse than no setting. So a saved choice
that has stopped being launchable — Chrome uninstalled, a path that moved — is
reported as exactly that, with the default that is being used instead, rather
than appearing to be in force while something else runs.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from openmirror.agent.browsers import (
    BrowserUnavailable,
    available,
    from_id,
    settings,
)
from openmirror.config import config

log = logging.getLogger(__name__)

router = APIRouter(prefix='/api/browser')


class ChoiceBody(BaseModel):
    # A catalog id: chromium, chrome, msedge, firefox, webkit, or custom.
    browser: str
    # Only for `custom`. The path to a Chromium-based binary.
    executable: str = ''


def _state() -> dict[str, Any]:
    store = settings()
    choice = store.load()
    return {
        'browser': choice.id,
        'engine': choice.engine,
        'channel': choice.channel,
        'executable': choice.executable,
        'label': choice.label(),
        # Which of the two is in force, said plainly. A person who set
        # OPENMIRROR_BROWSER_CHANNEL and then saved something else in the dialog
        # needs to know which one won.
        'source': 'saved' if store.saved else 'environment',
        'default': store.default.id,
        'headless': config.browser_headless,
        'profile': str(config.browser_profile),
    }


@router.get('')
async def get_browser() -> dict[str, Any]:
    """What is in force, and everything that could be."""
    return {**_state(), 'choices': await available()}


@router.put('')
async def set_browser(body: ChoiceBody) -> dict[str, Any]:
    """Save a choice, refusing one that cannot launch.

    Checked here rather than at launch: otherwise the error arrives from inside
    a tool, halfway through a task, as a Playwright traceback — and the person
    reading it is trying to book a flight.
    """
    try:
        choice = from_id(body.browser.strip(), body.executable.strip())
        settings().save(choice)
    except BrowserUnavailable as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log.info('browser: set to %s', choice.label())
    return {
        **_state(),
        # Said out loud, because a browser already open keeps running: a session
        # mid-task is not restarted under somebody because a setting changed.
        'note': 'Sessions already running keep the browser they started. New ones use this.',
    }


@router.delete('')
async def clear_browser() -> dict[str, Any]:
    """Forget the saved choice and go back to what the environment says."""
    settings().clear()
    return _state()
