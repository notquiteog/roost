"""One agent session: a conversation that can act on the machine.

The shape is a loop — ask the model, run what it asks for, give it the
results, ask again — and the interesting parts are all in what happens
between those steps.

**Approval suspends the loop rather than skipping the call.** When a tool
needs a human, the turn stops on a future and waits. It does not proceed
optimistically and it does not time out into a default, because both of those
turn "are you sure?" into a formality.

**A denial is a tool result, not an error.** The model is told, in the result,
that the person said no and why. That lets it try a different approach, which
is what a person who said no usually wants. Ending the turn instead just
makes them say the same thing again in words.

**Interruption cancels in-flight work.** Killing the task cancels the
provider stream and any running subprocess through normal cancellation, and
the session survives to take the next message with its history intact.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from roost.agent.approval import ApprovalPolicy, Decision
from roost.agent.tools.base import Assessment, Tool, ToolContext, ToolError
from roost.protocol.agent import (
    AgentError,
    QuestionAsked,
    Risk,
    SessionEnded,
    SessionStarted,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolCompleted,
    ToolDenied,
    ToolOutputDelta,
    ToolProposed,
    ToolResult,
    ToolStarted,
    ToolStatus,
    TurnCompleted,
    TurnStarted,
)
from roost.providers.base import (
    ChatRequest,
    ImageBlock,
    Message,
    StreamDone,
    StreamText,
    StreamThinking,
    StreamToolUse,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)

log = logging.getLogger(__name__)

# How many model round trips one user message may cost. A ceiling rather than
# a target: hitting it means the agent is looping, and stopping with an
# explanation beats burning tokens until someone notices the bill.
MAX_STEPS = 60

# How many events one session keeps for replay. Roughly an afternoon of work.
EVENT_LOG_LIMIT = 5000


def _denial_text(call: ToolCall, reason: str) -> str:
    """What the model is told when a person says no.

    For most tools, "try another way" is right — the approach was wrong, not
    the goal. For money and secrets it is exactly wrong, and observably so:
    told to place an order and refused, a local model immediately clicked a
    different button and then tried to fill in the card fields. It was not
    being devious; it was following the instruction to find another route.

    So a refusal on those two is terminal for that goal. Nothing else about
    the turn stops — it can carry on with everything else it was asked to do.
    """
    said = reason or 'No reason given.'
    if call.risk in (Risk.PURCHASE, Risk.CREDENTIAL):
        return (
            f'Refused: {said}\n\n'
            'This is final. Do not attempt it another way, do not try a different button, '
            'and do not try to enter the details yourself. Stop pursuing this particular '
            'action, tell the person plainly what you were about to do and why it stopped, '
            'and carry on with anything else they asked for.'
        )
    return (
        f'The user declined this. {said} '
        'Do not retry it as-is; either do it another way or ask them what they want.'
    )


class AgentSession:
    def __init__(
        self,
        *,
        session_id: str | None = None,
        root: Path,
        provider: Any,
        model: str,
        tools: list[Tool],
        policy: ApprovalPolicy | None = None,
        system_prompt: str = '',
        max_steps: int = MAX_STEPS,
        env: dict[str, str] | None = None,
        title: str = '',
        context_provider: Any = None,
        on_user_text: Any = None,
        confined: bool = True,
        browser: Any = None,
        checkpoints: Any = None,
    ) -> None:
        self.id = session_id or uuid.uuid4().hex[:16]
        self.title = title
        self.root = root.resolve()
        self.cwd = self.root
        # False means the whole filesystem. Declared at session creation and
        # reported in session.started, so a client can say so plainly.
        self.confined = confined
        # Owned here so it dies with the session: an abandoned Chromium is
        # 200 MB, and sessions outlive their client by design.
        self.browser = browser
        # Undo for this session's own file edits. See checkpoint.py for what it
        # does not cover — a shell command can write anywhere.
        self.checkpoints = checkpoints
        self.provider = provider
        self.model = model
        self.policy = policy or ApprovalPolicy()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.env = env or {}

        self.tools: dict[str, Tool] = {t.name: t for t in tools}
        self.messages: list[Message] = []

        # Called with the user's message before each turn; whatever it returns
        # is appended to the system prompt for that turn only. A hook rather
        # than a memory object, so the session does not have to know what
        # memory is — or that there is any.
        self.context_provider = context_provider
        # Called with the user's message after the turn is under way, for
        # anything that wants to learn from it. Never awaited on the critical
        # path: a slow write must not delay the answer.
        self.on_user_text = on_user_text
        self._turn_context: str = ''
        # Images produced by tools during the current step, attached to the
        # message carrying their results. Every provider accepts images on a
        # user turn; none of them accept one inside a tool result on all three
        # dialects, so this is the portable place to put them.
        self._pending_images: list[tuple[str, str]] = []

        # Events are logged and fanned out rather than queued to one reader.
        # A browser tab closing must not destroy the session — the agent may
        # be four minutes into a build — so a client detaches and reattaches,
        # and asks for everything after the last sequence number it saw.
        self._log: list[tuple[int, Any]] = []
        self._seq = 0
        self._subs: set[asyncio.Queue[tuple[int, Any]]] = set()
        self.last_active: float = time.time()

        self._approvals: dict[str, asyncio.Future[tuple[bool, str, bool]]] = {}
        self._questions: dict[str, asyncio.Future[str]] = {}
        self._turn: asyncio.Task[None] | None = None
        self._closed = False

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        await self._emit(
            SessionStarted(
                session_id=self.id,
                cwd=str(self.cwd),
                model=self.model,
                policy=self.policy.describe() + ('' if self.confined else '  ·  UNCONFINED: the whole filesystem'),
                tools=sorted(self.tools),
            )
        )

    async def close(self, reason: str = 'closed') -> None:
        if self._closed:
            return
        self._closed = True
        self.interrupt()
        # Anything still waiting on a human is resolved rather than left to
        # hang: a closed session must not leak a suspended turn.
        for fut in list(self._approvals.values()):
            if not fut.done():
                fut.set_result((False, 'session closed', False))
        for fut in list(self._questions.values()):
            if not fut.done():
                fut.set_result('(the session was closed before this was answered)')

        if self.browser is not None:
            try:
                await self.browser.close()
            except Exception:  # noqa: BLE001
                log.debug('browser did not close cleanly', exc_info=True)

        await self._emit(SessionEnded(session_id=self.id, reason=reason))

    async def events(self, since: int = 0) -> AsyncIterator[Any]:
        """Subscribe, replaying anything after `since`.

        The queue is registered *before* the replay so nothing emitted during
        it is lost; duplicates are then filtered by sequence number, which is
        the only ordering that has no gap.
        """
        queue: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()
        self._subs.add(queue)
        last = since
        try:
            for seq, event in list(self._log):
                if seq <= last:
                    continue
                last = seq
                yield event
                if isinstance(event, SessionEnded):
                    return

            while True:
                seq, event = await queue.get()
                if seq <= last:
                    continue
                last = seq
                yield event
                if isinstance(event, SessionEnded):
                    return
        finally:
            self._subs.discard(queue)

    async def _emit(self, event: Any) -> None:
        self._seq += 1
        self.last_active = time.time()
        event.seq = self._seq
        self._log.append((self._seq, event))
        # Bounded, because a session left running for a day should not grow
        # without limit. A client that has been away longer than this reattaches
        # to a partial history, which is why the cap is generous.
        if len(self._log) > EVENT_LOG_LIMIT:
            del self._log[: len(self._log) - EVENT_LOG_LIMIT]
        for queue in list(self._subs):
            queue.put_nowait((self._seq, event))

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def attached(self) -> int:
        return len(self._subs)

    @property
    def busy(self) -> bool:
        return self._turn is not None and not self._turn.done()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def waiting_on(self) -> str | None:
        """What a human owes this session, if anything.

        A reattaching client needs this: a session suspended on an approval
        looks identical to an idle one until you know it is waiting.
        """
        if self._approvals:
            return 'approval'
        if self._questions:
            return 'question'
        return None

    # -- client commands ----------------------------------------------------

    def submit(self, text: str, attachments: list[dict[str, Any]] | None = None) -> str:
        """Start a turn. Returns its id immediately; the work happens in a task."""
        if self._closed:
            raise RuntimeError('session is closed')
        if self._turn and not self._turn.done():
            raise RuntimeError('a turn is already running — interrupt it first')

        turn_id = uuid.uuid4().hex[:16]
        self._turn = asyncio.create_task(self._run_turn(turn_id, text, attachments or []))
        return turn_id

    def approve(self, call_id: str, remember: bool = False) -> bool:
        fut = self._approvals.get(call_id)
        if fut is None or fut.done():
            return False
        fut.set_result((True, '', remember))
        return True

    def deny(self, call_id: str, reason: str = '') -> bool:
        fut = self._approvals.get(call_id)
        if fut is None or fut.done():
            return False
        fut.set_result((False, reason, False))
        return True

    def answer(self, question_id: str, answer: str) -> bool:
        fut = self._questions.get(question_id)
        if fut is None or fut.done():
            return False
        fut.set_result(answer)
        return True

    def interrupt(self) -> bool:
        if self._turn and not self._turn.done():
            self._turn.cancel()
            return True
        return False

    # -- the loop -----------------------------------------------------------

    async def _run_turn(self, turn_id: str, text: str, attachments: list[dict[str, Any]]) -> None:
        await self._emit(TurnStarted(session_id=self.id, turn_id=turn_id, text=text))

        if self.checkpoints is not None:
            # Opened before anything runs, so the snapshot is of the tree as it
            # was when the person asked, not partway through the answer.
            self.checkpoints.begin(turn_id, text[:80])

        # Recall runs before the first request, so what is remembered is in
        # front of the model from the outset rather than arriving a step late.
        self._turn_context = ''
        if self.context_provider is not None:
            try:
                self._turn_context = await self.context_provider(text) or ''
            except Exception:  # noqa: BLE001
                # Never fatal. Not remembering is a worse turn; failing is no turn.
                log.exception('context provider failed; continuing without it')

        if self.on_user_text is not None:
            # Fire and forget: this is bookkeeping, not part of the answer.
            asyncio.create_task(self._safe_learn(text))

        blocks: list[Any] = [TextBlock(text=text)]
        for att in attachments:
            if att.get('type') == 'image' and att.get('data'):
                blocks.append(ImageBlock(data=att['data'], media_type=att.get('media_type', 'image/png')))
        self.messages.append(Message(role='user', content=blocks))

        usage_total = {'input_tokens': 0, 'output_tokens': 0}
        stop_reason = 'end_turn'

        try:
            for _step in range(self.max_steps):
                assistant_blocks, calls, done = await self._one_exchange(turn_id)

                for key in usage_total:
                    usage_total[key] += done.usage.get(key, 0)

                if assistant_blocks:
                    self.messages.append(Message(role='assistant', content=assistant_blocks))

                if not calls:
                    stop_reason = 'end_turn'
                    break

                self._pending_images = []
                results = await self._run_calls(turn_id, calls)

                # Tool results go back as a user turn: that is where every
                # provider in use expects them, Anthropic and OpenAI alike.
                blocks: list[Any] = list(results)
                for data, media_type in self._pending_images:
                    blocks.append(ImageBlock(data=data, media_type=media_type))
                self._pending_images = []
                self.messages.append(Message(role='user', content=blocks))
            else:
                stop_reason = 'max_steps'
                await self._emit(
                    AgentError(
                        session_id=self.id,
                        turn_id=turn_id,
                        message=f'Stopped after {self.max_steps} steps without finishing.',
                    )
                )

        except asyncio.CancelledError:
            stop_reason = 'interrupted'
            if self.checkpoints is not None:
                # Kept, not discarded: a turn cut off halfway is the one most
                # likely to have left the tree somewhere nobody wanted.
                self.checkpoints.commit()
            # Recorded in the history so the next turn's context shows the work
            # was cut off rather than completed.
            self.messages.append(
                Message(role='assistant', content=[TextBlock(text='[interrupted by the user]')])
            )
            await self._emit(
                TurnCompleted(session_id=self.id, turn_id=turn_id, stop_reason='interrupted', usage=usage_total)
            )
            raise

        except Exception as exc:  # noqa: BLE001 - the turn boundary is where errors become events
            log.exception('turn %s failed', turn_id)
            stop_reason = 'error'
            await self._emit(
                AgentError(
                    session_id=self.id,
                    turn_id=turn_id,
                    message=str(exc),
                    retryable=isinstance(exc, (asyncio.TimeoutError, ConnectionError)),
                )
            )

        if self.checkpoints is not None:
            self.checkpoints.commit()

        await self._emit(
            TurnCompleted(session_id=self.id, turn_id=turn_id, stop_reason=stop_reason, usage=usage_total)
        )

    async def _one_exchange(self, turn_id: str) -> tuple[list[Any], list[ToolCall], StreamDone]:
        """One request to the model, streamed."""
        req = ChatRequest(
            model=self.model,
            messages=self.messages,
            system=(f'{self.system_prompt}\n\n{self._turn_context}' if self._turn_context else self.system_prompt)
            or None,
            tools=[
                ToolSpec(name=t.name, description=t.description, input_schema=t.input_schema)
                for t in self.tools.values()
            ],
        )

        assistant_blocks: list[Any] = []
        calls: list[ToolCall] = []
        text_buf: list[str] = []
        think_buf: list[str] = []
        done = StreamDone()

        async for event in self.provider.stream(req):
            if isinstance(event, StreamText):
                text_buf.append(event.text)
                await self._emit(TextDelta(session_id=self.id, turn_id=turn_id, text=event.text))
            elif isinstance(event, StreamThinking):
                think_buf.append(event.text)
                await self._emit(ThinkingDelta(session_id=self.id, turn_id=turn_id, text=event.text))
            elif isinstance(event, StreamToolUse):
                calls.append(ToolCall(id=event.id, name=event.name, arguments=event.input))
            elif isinstance(event, StreamDone):
                done = event

        # Thinking first: providers that accept it back require it before the
        # text and tool blocks of the same turn.
        if think_buf:
            assistant_blocks.append(ThinkingBlock(text=''.join(think_buf)))
        if text_buf:
            assistant_blocks.append(TextBlock(text=''.join(text_buf)))
        for call in calls:
            assistant_blocks.append(ToolUseBlock(id=call.id, name=call.name, input=call.arguments))

        return assistant_blocks, calls, done

    async def _safe_learn(self, text: str) -> None:
        try:
            await self.on_user_text(text)
        except Exception:  # noqa: BLE001
            log.exception('learning from the message failed')

    async def _run_calls(self, turn_id: str, calls: list[ToolCall]) -> list[ToolResultBlock]:
        """Assess, approve and run each call, in order.

        Sequential on purpose. Two tools running at once can edit the same file
        or race on the same directory, and the model has no way to know they
        will overlap when it proposes them.
        """
        results: list[ToolResultBlock] = []

        for call in calls:
            tool = self.tools.get(call.name)
            if tool is None:
                results.append(
                    ToolResultBlock(
                        tool_use_id=call.id,
                        content=f'No such tool: {call.name}. Available: {", ".join(sorted(self.tools))}',
                        is_error=True,
                    )
                )
                continue

            ctx = self._context(turn_id, call.id)

            try:
                assessment = tool.assess(call.arguments, ctx)
            except Exception as exc:  # noqa: BLE001
                assessment = Assessment(risk=Risk.READ, summary='', invalid=str(exc))

            if assessment.invalid:
                results.append(
                    ToolResultBlock(tool_use_id=call.id, content=f'Invalid call: {assessment.invalid}', is_error=True)
                )
                continue

            call.risk = assessment.risk
            call.summary = assessment.summary

            decision, why = self.policy.decide(call)

            if decision is Decision.DENY:
                call.status = ToolStatus.DENIED
                await self._emit(ToolDenied(session_id=self.id, turn_id=turn_id, call_id=call.id, reason=why))
                results.append(
                    ToolResultBlock(tool_use_id=call.id, content=f'Not permitted: {why}', is_error=True)
                )
                continue

            await self._emit(
                ToolProposed(
                    session_id=self.id,
                    turn_id=turn_id,
                    call=call,
                    needs_approval=decision is Decision.ASK,
                )
            )

            if decision is Decision.ASK and call.name != 'ask_user':
                allowed, reason, remember = await self._await_approval(call.id)
                if not allowed:
                    call.status = ToolStatus.DENIED
                    await self._emit(
                        ToolDenied(session_id=self.id, turn_id=turn_id, call_id=call.id, reason=reason)
                    )
                    results.append(
                        ToolResultBlock(tool_use_id=call.id, content=_denial_text(call, reason), is_error=True)
                    )
                    continue
                if remember:
                    self.policy.remember(call)

            results.append(await self._execute(turn_id, tool, call, ctx))

        return results

    async def _execute(self, turn_id: str, tool: Tool, call: ToolCall, ctx: ToolContext) -> ToolResultBlock:
        call.status = ToolStatus.RUNNING
        await self._emit(ToolStarted(session_id=self.id, turn_id=turn_id, call_id=call.id))
        started = time.monotonic()

        images: list[tuple[str, str]] = []
        try:
            output = await tool.run(call.arguments, ctx)
            ok, content, display, truncated = True, output.content, output.display, output.truncated
            images = output.images
        except ToolError as exc:
            # Expected failure: the model reads it and adjusts.
            ok, content, display, truncated = False, str(exc), None, False
        except asyncio.CancelledError:
            call.status = ToolStatus.CANCELLED
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception('tool %s failed', call.name)
            ok, content, display, truncated = False, f'{type(exc).__name__}: {exc}', None, False

        duration = int((time.monotonic() - started) * 1000)
        call.status = ToolStatus.COMPLETED if ok else ToolStatus.FAILED

        await self._emit(
            ToolCompleted(
                session_id=self.id,
                turn_id=turn_id,
                result=ToolResult(
                    id=call.id,
                    name=call.name,
                    ok=ok,
                    content=content,
                    display=display,
                    duration_ms=duration,
                    truncated=truncated,
                ),
            )
        )
        self._pending_images.extend(images)
        return ToolResultBlock(tool_use_id=call.id, content=content, is_error=not ok)

    # -- suspension points --------------------------------------------------

    async def _await_approval(self, call_id: str) -> tuple[bool, str, bool]:
        fut: asyncio.Future[tuple[bool, str, bool]] = asyncio.get_running_loop().create_future()
        self._approvals[call_id] = fut
        try:
            # No timeout. An agent left running overnight should still be
            # waiting in the morning, not have decided for itself.
            return await fut
        finally:
            self._approvals.pop(call_id, None)

    def _context(self, turn_id: str, call_id: str) -> ToolContext:
        async def emit(text: str, stream: str) -> None:
            await self._emit(
                ToolOutputDelta(
                    session_id=self.id, turn_id=turn_id, call_id=call_id, text=text, stream=stream
                )
            )

        async def ask(question: str, options: list[str], multi: bool) -> str:
            question_id = uuid.uuid4().hex[:16]
            fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            self._questions[question_id] = fut
            await self._emit(
                QuestionAsked(
                    session_id=self.id,
                    turn_id=turn_id,
                    question_id=question_id,
                    question=question,
                    options=options,
                    multi=multi,
                )
            )
            try:
                return await fut
            finally:
                self._questions.pop(question_id, None)

        return ToolContext(
            root=self.root,
            cwd=self.cwd,
            emit=emit,
            ask=ask,
            session_id=self.id,
            env=self.env,
            confined=self.confined,
            checkpoint=self.checkpoints,
        )
