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

Remembered approvals are scoped to the tool *and its arguments*, not to the
tool. Otherwise one "yes, and don't ask again" on `rm -rf build/` becomes
permanent unattended `rm -rf` on anything.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum

from roost.protocol.agent import Risk, ToolCall

log = logging.getLogger(__name__)


class Mode(StrEnum):
    READ_ONLY = 'read_only'
    ASK = 'ask'
    AUTO_EDIT = 'auto_edit'
    TRUSTED = 'trusted'
    UNRESTRICTED = 'unrestricted'


# What each mode allows without asking. Note what is absent from every row:
# PURCHASE and CREDENTIAL. `unrestricted` means "stop asking me about this
# machine", which is not the same sentence as "spend my money", and the two
# should not be bought with one click.
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
    # Let it spend money without asking. Its own switch, orthogonal to `mode`,
    # because an agent that can run any command on your laptop and an agent
    # that can place orders on your card are different things to consent to.
    allow_purchases: bool = False
    # Let it type secrets into forms. Off, and the browser tool's default is
    # to hand the field to you instead — which is better than an approval
    # prompt, because then the agent never sees the value at all.
    allow_credentials: bool = False
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

        if call.risk is Risk.CREDENTIAL and not self.allow_credentials:
            # Refused rather than asked. The alternative to the agent typing
            # your password is not you approving it typing your password; it
            # is you typing it, in a window it cannot read.
            return Decision.DENY, 'entering secrets is disabled — type it yourself in the browser'

        if call.risk is Risk.PURCHASE:
            if self.allow_purchases:
                return Decision.ALLOW, 'purchases are allowed without asking on this install'
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
        auto = sorted(r.value for r in AUTO[self.mode])
        if self.allow_purchases:
            auto.append('purchase')
        return f'{self.mode.value} (runs without asking: {", ".join(auto)})'
