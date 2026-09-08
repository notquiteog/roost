"""Sessions that outlive the client watching them.

A session is a unit of work, not a connection. Closing a tab must not kill an
agent four minutes into a build, and reopening one must not start over — so
sessions live here, clients attach and detach, and the event log in each
session is what makes reattaching cheap.

That has a consequence worth being explicit about: a session with nobody
attached can still be *waiting* on somebody. An approval has no timeout by
design, so a suspended session sits there indefinitely. The reaper below
therefore never closes a session that is waiting on a human, only ones that
have gone quiet on their own.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from roost.agent.approval import Mode
from roost.agent.runtime import build_session
from roost.agent.session import AgentSession
from roost.mcp.manager import manager as mcp_manager

log = logging.getLogger(__name__)

# A session nobody has touched for this long, that is not running and not
# waiting on anyone, is closed. Generous: the cost of keeping one is a little
# memory, and the cost of reaping one someone wanted is their work.
IDLE_TIMEOUT = 12 * 60 * 60
REAP_INTERVAL = 300


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self._reaper: asyncio.Task[None] | None = None

    async def create(
        self,
        *,
        root: str | Path,
        provider: Any,
        model: str,
        mode: Mode | str,
        session_id: str | None = None,
        title: str = '',
        memory: Any = None,
        user_id: str = '',
        cfg: Any = None,
    ) -> AgentSession:
        from roost.config import config as default_config

        cfg = cfg or default_config

        browser = None
        if cfg.browser_enabled:
            from roost.agent.browser import BrowserConfig, BrowserSession

            # Constructed, not started: launching Chromium takes a few hundred
            # milliseconds and most sessions never open a page.
            browser = BrowserSession(
                BrowserConfig(profile_dir=cfg.browser_profile, headless=cfg.browser_headless)
            )

        checkpoints = None
        if cfg.checkpoints_enabled:
            from roost.agent.checkpoint import CheckpointStore

            # Under the data directory, never inside the working root — a
            # snapshot in the tree the agent is editing gets read, grepped and
            # eventually committed.
            checkpoints = CheckpointStore(
                Path(cfg.memory_db).parent / 'checkpoints' / (session_id or 'session')
            )

        session = build_session(
            root=root, provider=provider, model=model, mode=mode, session_id=session_id,
            title=title or Path(root).name, memory=memory, user_id=user_id,
            confined=not cfg.unconfined,
            allow_purchases=cfg.allow_purchases,
            allow_credentials=cfg.allow_credentials,
            mcp=mcp_manager if cfg.mcp_enabled and mcp_manager.servers else None,
            checkpoints=checkpoints,
            web=cfg if cfg.web_enabled else None,
            browser=browser,
            desktop=cfg.desktop_enabled,
        )
        self._sessions[session.id] = session
        await session.start()
        log.info('session %s created at %s', session.id, session.root)
        return session

    def get(self, session_id: str) -> AgentSession | None:
        return self._sessions.get(session_id)

    def list(self) -> list[dict[str, Any]]:
        """A summary per session, for a client showing what is in flight."""
        return [
            {
                'id': s.id,
                'title': s.title,
                'root': str(s.root),
                'model': s.model,
                'policy': s.policy.mode.value,
                'busy': s.busy,
                'waiting_on': s.waiting_on,
                'attached': s.attached,
                'seq': s.seq,
                'turns': sum(1 for m in s.messages if m.role == 'user'),
                'idle_for': int(time.time() - s.last_active),
                'closed': s.closed,
            }
            for s in self._sessions.values()
        ]

    async def close(self, session_id: str, reason: str = 'closed') -> bool:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        await session.close(reason)
        return True

    async def close_all(self) -> None:
        for session_id in list(self._sessions):
            await self.close(session_id, 'server shutting down')

    # -- reaping ------------------------------------------------------------

    def start_reaper(self) -> None:
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.create_task(self._reap_loop())

    def stop_reaper(self) -> None:
        if self._reaper and not self._reaper.done():
            self._reaper.cancel()

    async def _reap_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(REAP_INTERVAL)
                await self.reap()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception('session reaper failed')

    async def reap(self) -> list[str]:
        """Close sessions that have gone quiet. Returns what was closed."""
        now = time.time()
        doomed = [
            s.id
            for s in self._sessions.values()
            # Never a session someone is expected to answer: it has been
            # waiting precisely because nobody has got to it yet.
            if not s.busy and s.waiting_on is None and s.attached == 0
            and now - s.last_active > IDLE_TIMEOUT
        ]
        for session_id in doomed:
            log.info('reaping idle session %s', session_id)
            await self.close(session_id, 'idle')
        return doomed


manager = SessionManager()
