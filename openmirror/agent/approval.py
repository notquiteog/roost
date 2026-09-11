"""Deciding what may run without asking.

The whole point of grading risk is so that the question a person is asked is
rare enough to be read. Five modes, each a line drawn at a different risk
level, and the line is the only thing that varies:

    read_only     nothing that changes anything, ever
    ask           reads run; everything else is asked          (default)
    auto_edit     reads and file writes run; commands are asked
    trusted       commands run too; destructive things are asked
    unrestricted  nothing is asked

`unrestricted` exists because people ask for it and will otherwise approve
everything by reflex, which is worse — a policy that is nominally strict but
answered without reading is a policy that does nothing while claiming to. It
is a deliberate, visible choice, and it never applies to `ask_user`.

Money and secrets sit on a second axis and are not on that ladder at all.
This harness is meant to be able to finish a task that ends at a checkout, so
both are *possible*; neither is ever automatic. A purchase is confirmed in
every mode, including `unrestricted`, and including a run nobody is watching —
there is no flag that turns that off, because a flag like that is one somebody
sets during a demo and still has set six months later. Secrets are confirmed
too, and the value is kept out of the prompt, the transcript and the log.

Remembered approvals are scoped to the tool *and its arguments*, not to the
tool. Otherwise one "yes, and don't ask again" on `rm -rf build/` becomes
permanent unattended `rm -rf` on anything — and a purchase or a secret is
never remembered at all, whatever the person ticks, because "don't ask again"
about spending money is the one answer nobody should be able to give once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum

from openmirror.protocol.agent import Risk, ToolCall

log = logging.getLogger(__name__)


class Mode(StrEnum):
    READ_ONLY = 'read_only'
    ASK = 'ask'
    AUTO_EDIT = 'auto_edit'
    TRUSTED = 'trusted'
    UNRESTRICTED = 'unrestricted'


# What each mode allows without asking. Note what is absent from every row:
# PURCHASE and CREDENTIAL. Both are *possible* by default — this harness is
# meant to be able to finish a task that ends in a checkout — but neither is
# ever automatic. `unrestricted` means "stop asking me about this machine",
# which is a different sentence from "spend my money", and autopilot means
# "do not check in about steps", which is a different sentence again.
AUTO: dict[Mode, set[Risk]] = {
    Mode.READ_ONLY: {Risk.READ},
    Mode.ASK: {Risk.READ},
    Mode.AUTO_EDIT: {Risk.READ, Risk.WRITE},
    Mode.TRUSTED: {Risk.READ, Risk.WRITE, Risk.EXECUTE, Risk.NETWORK},
    Mode.UNRESTRICTED: {Risk.READ, Risk.WRITE, Risk.EXECUTE, Risk.NETWORK, Risk.DESTRUCTIVE},
}


class Decision(StrEnum):
    ALLOW = 'allow'    # run it, no question
    ASK = 'ask'        # a human decides
    DENY = 'deny'      # refused outright; never reaches a human


@dataclass(slots=True)
class Rule:
    """An operator-set rule, checked before the mode.

    `pattern` is matched against the call's summary — the same one-line text a
    human would be shown — so a rule reads the way the prompt reads.
    """

    pattern: str
    decision: Decision
    tool: str | None = None

    def matches(self, call: ToolCall) -> bool:
        if self.tool and self.tool != call.name:
            return False
        try:
            return re.search(self.pattern, call.summary) is not None
        except re.error:
            log.warning('approval rule has an invalid pattern, ignoring: %s', self.pattern)
            return False


def _fingerprint(call: ToolCall) -> str:
    """Identity of a call for the purpose of remembering a decision.

    Arguments are included and sorted, so `shell(ls)` and `shell(rm -rf /)`
    can never share a remembered yes.
    """
    payload = json.dumps({'name': call.name, 'args': call.arguments}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


@dataclass
class ApprovalPolicy:
    mode: Mode = Mode.ASK
    rules: list[Rule] = field(default_factory=list)
    # Whether spending is possible at all — not whether it is automatic, which
    # it never is. Off turns a purchase into a refusal rather than a prompt,
    # for an install that should never be able to buy anything.
    allow_purchases: bool = True
    # The same, for typing a password, a card number or a one-time code.
    # `browser_hand_over` is still there and is still the better move when the
    # person is at the keyboard: a value the agent never receives cannot end
    # up in a transcript, a log or a model's context window.
    allow_credentials: bool = True
    # Tools that are always asked about regardless of mode or risk, for an
    # operator who wants a hard stop on one specific thing.
    always_ask: set[str] = field(default_factory=set)
    _remembered: set[str] = field(default_factory=set, repr=False)

    def decide(self, call: ToolCall) -> tuple[Decision, str]:
        # Asking a question is never auto-answered, in any mode.
        if call.name == 'ask_user':
            return Decision.ASK, 'the agent is asking you something'

        for rule in self.rules:
            if rule.matches(call):
                return rule.decision, f'matched rule {rule.pattern!r}'

        if call.name in self.always_ask:
            return Decision.ASK, f'{call.name} is always confirmed here'

        if _fingerprint(call) in self._remembered:
            return Decision.ALLOW, 'you approved this exact call earlier in the session'

        # A mode whose whole promise is "this session cannot change anything"
        # must not be escapable by clicking yes — and that has to include the
        # two risks that bypass the mode ladder, or `read_only` would offer a
        # checkout prompt while claiming to be read-only.
        if self.mode is Mode.READ_ONLY and call.risk in (Risk.PURCHASE, Risk.CREDENTIAL):
            return Decision.DENY, 'this session is read-only'

        if call.risk is Risk.CREDENTIAL:
            if not self.allow_credentials:
                return Decision.DENY, 'entering secrets is switched off — type it yourself in the browser'
            # Asked, never auto-run, and the value is never in the question.
            # See `Assessment.summary` at every call site that grades
            # CREDENTIAL: an approval prompt that prints the secret has
            # defeated the point of guarding it.
            return Decision.ASK, 'this enters a secret'

        if call.risk is Risk.PURCHASE:
            if not self.allow_purchases:
                return Decision.DENY, 'spending money is switched off on this install'
            # Always. There is no mode and no setting that skips this, and
            # that is the whole design: `unrestricted` means "stop asking me
            # about this machine", autopilot means "do not check in about
            # steps" — neither has ever meant "spend without asking", and a
            # single switch that made them mean it is a switch someone would
            # set once and forget while an agent runs unattended.
            return Decision.ASK, 'this spends money'

        if call.risk in AUTO[self.mode]:
            return Decision.ALLOW, f'{call.risk.value} is allowed in {self.mode.value} mode'

        # read_only refuses rather than asking. A mode whose promise is "this
        # session cannot change anything" must not be escapable by clicking yes.
        if self.mode is Mode.READ_ONLY:
            return Decision.DENY, 'this session is read-only'

        return Decision.ASK, f'{call.risk.value} needs your approval in {self.mode.value} mode'

    def remember(self, call: ToolCall) -> None:
        """Record a 'don't ask again' for this exact call.

        Refused for destructive calls: 'always' and 'destructive' should not be
        combinable by a single click, and a person who genuinely wants that has
        `unrestricted`, which at least says what it is.
        """
        if call.risk in (Risk.DESTRUCTIVE, Risk.PURCHASE, Risk.CREDENTIAL):
            log.info('not remembering approval for a %s call: %s', call.risk.value, call.summary)
            return
        self._remembered.add(_fingerprint(call))

    def describe(self) -> str:
        """What this session may do without stopping, in a sentence.

        Purchases are never in the list, whatever `allow_purchases` says: that
        flag decides whether spending is *possible*, not whether it is
        automatic. Saying otherwise in the one line a person reads at the top
        of a session would be the most consequential lie this file could tell.
        """
        auto = sorted(r.value for r in AUTO[self.mode])
        line = f'{self.mode.value} (runs without asking: {", ".join(auto)})'

        confirmed = []
        if self.allow_purchases:
            confirmed.append('purchases')
        if self.allow_credentials:
            confirmed.append('entering secrets')
        if confirmed:
            line += f'; {" and ".join(confirmed)} are possible and always confirmed'
        return line
