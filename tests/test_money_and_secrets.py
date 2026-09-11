"""Spending money and typing secrets: possible, and never automatic.

Both used to be refused outright. They are capabilities now, because a
harness meant to finish a real task has to be able to reach the end of one —
but the way they are granted is the whole of the design, and these are the
assertions that hold it:

* a purchase is confirmed in **every** mode, including `unrestricted`, and in
  a run nobody is watching. There is no setting that skips it;
* neither is ever remembered, so "don't ask again" cannot be answered once
  about spending money;
* and the secret itself never reaches the approval prompt, the transcript,
  the tool result or the log — the agent may type a password without anyone
  else ever being able to read it back.
"""

from __future__ import annotations

import json
import shutil

import pytest

from openmirror.agent.approval import ApprovalPolicy, Decision, Mode
from openmirror.agent.browser import BrowserConfig, BrowserSession
from openmirror.agent.tools.base import ToolContext
from openmirror.agent.tools.browser import (
    BrowserNavigateTool,
    BrowserReadTool,
    BrowserTypeTool,
    classify_field,
)
from openmirror.protocol.agent import Risk, ToolCall
from tests.fixtures.hotels import HotelSite

SECRET = 'hunter2-do-not-print-me'
CARD = '4111111111111111'


def buying(summary: str = 'click "Place your order" — this looks like it completes a purchase'):
    return ToolCall(id='c1', name='browser_click', risk=Risk.PURCHASE, summary=summary)


def entering():
    return ToolCall(id='c2', name='browser_type', risk=Risk.CREDENTIAL, summary="type into 'Password'")


# -- the policy -------------------------------------------------------------


@pytest.mark.parametrize('mode', list(Mode))
def test_a_purchase_is_confirmed_in_every_mode(mode):
    """Including `unrestricted`, which is where a mistake would actually cost.

    `unrestricted` means "stop asking me about this machine". It has never
    meant "spend my money", and there is no flag that makes it mean that —
    a flag like that is one somebody sets during a demo and still has set six
    months later.
    """
    decision, why = ApprovalPolicy(mode=mode).decide(buying())
    if mode in (Mode.READ_ONLY, Mode.PLAN):
        # A mode whose promise is that nothing changes must not offer a
        # checkout prompt while claiming to be read-only. Plan mode makes the
        # same promise until its plan is approved, and keeps it the same way.
        assert decision is Decision.DENY, why
    else:
        assert decision is Decision.ASK, f'{mode.value} would have bought it: {why}'


@pytest.mark.parametrize('mode', list(Mode))
def test_a_secret_is_confirmed_in_every_mode(mode):
    decision, why = ApprovalPolicy(mode=mode).decide(entering())
    expected = Decision.DENY if mode in (Mode.READ_ONLY, Mode.PLAN) else Decision.ASK
    assert decision is expected, f'{mode.value}: {why}'


def test_either_can_be_switched_off_entirely():
    """Off makes it a refusal rather than a prompt, for an install that should
    not be able to do it at all."""
    no_money = ApprovalPolicy(mode=Mode.TRUSTED, allow_purchases=False)
    assert no_money.decide(buying())[0] is Decision.DENY

    no_secrets = ApprovalPolicy(mode=Mode.TRUSTED, allow_credentials=False)
    assert no_secrets.decide(entering())[0] is Decision.DENY


def test_neither_can_ever_be_remembered():
    """"Don't ask again" about spending money is the one answer nobody should
    be able to give once."""
    policy = ApprovalPolicy(mode=Mode.TRUSTED)
    for call in (buying(), entering()):
        policy.remember(call)
        assert policy.decide(call)[0] is Decision.ASK, (
            f'a {call.risk.value} approval was remembered and will not be asked again'
        )


def test_the_session_banner_never_claims_purchases_are_automatic():
    """It is the one line a person reads at the top of a session."""
    for mode in Mode:
        described = ApprovalPolicy(mode=mode).describe()
        auto = described.split(';')[0]
        assert 'purchase' not in auto, f'{mode.value} banner implies purchases run unasked: {described}'
    assert 'always confirmed' in ApprovalPolicy(mode=Mode.UNRESTRICTED).describe()


def test_an_approval_prompt_never_contains_the_secret():
    """The summary is what the person is shown. A prompt that prints the
    password has defeated the point of guarding it."""
    field = {'name': 'password', 'type': 'password', 'label': 'Password'}
    risk, _ = classify_field(field)
    assert risk is Risk.CREDENTIAL

    tool = BrowserTypeTool.__new__(BrowserTypeTool)
    tool.browser = type('B', (), {'last_elements': {1: field}})()
    assessment = tool.assess({'ref': 1, 'text': SECRET}, None)
    assert assessment.risk is Risk.CREDENTIAL
    assert SECRET not in assessment.summary, f'the prompt would print it: {assessment.summary}'


# -- and the value really does not come back --------------------------------


async def noop(*args):
    return None


async def noask(*args):
    return ''


@pytest.mark.asyncio
async def test_a_typed_secret_is_not_echoed_anywhere(tmp_path):
    """The agent may type a password. Nobody may read it back.

    Enabling this without the check would have leaked on every use: the tool
    reads the field back to confirm what landed, and for a password field
    that read *is* the password — into the tool result, the transcript, and
    the model's own context.
    """
    if not shutil.which('Xvfb'):
        pytest.skip('Xvfb is not installed')
    try:
        from playwright.async_api import async_playwright  # noqa: F401
    except ImportError:
        pytest.skip('playwright is not installed')

    from openmirror.agent.stage import VirtualStage

    stage = VirtualStage(1024, 768)
    browser = BrowserSession(
        BrowserConfig(profile_dir=tmp_path / 'p', headless=False,
                      viewport=(1024, 768), env=stage.env())
    )
    ctx = ToolContext(root=tmp_path, cwd=tmp_path, emit=noop, ask=noask, session_id='s')

    try:
        with HotelSite() as site:
            await BrowserNavigateTool(browser).run({'url': f'{site.url}/login'}, ctx)
            page = await BrowserReadTool(browser).run({}, ctx)

            refs = {e['name']: r for r, e in browser.last_elements.items() if e.get('name')}
            typing = BrowserTypeTool(browser)

            # An ordinary field still reports what landed — the read-back is
            # useful and only suppressed where it would be a leak.
            plain = await typing.run({'ref': refs['email'], 'text': 'someone@example.com'}, ctx)
            assert 'someone@example.com' in plain.content

            for name, value in (('password', SECRET), ('card_number', CARD)):
                out = await typing.run({'ref': refs[name], 'text': value}, ctx)
                haystack = out.content + json.dumps(out.display)
                assert value not in haystack, f'{name} leaked into the tool result: {out.content}'
                assert out.display['secret'] is True
                assert out.display['value'] is None

            # Nor through reading the page afterwards, which is the other way
            # a value could reach the model.
            after = await BrowserReadTool(browser).run({}, ctx)
            assert SECRET not in after.content
            assert CARD not in after.content
            assert SECRET not in page.content
    finally:
        await browser.close()
        stage.close()
