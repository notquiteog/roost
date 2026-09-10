"""Answering out loud: the questions, the approvals, and never guessing.

This mode's worst failure was silence. The agent would ask something or stop
for an approval, only assistant prose was being spoken, and the call went
quiet — leaving someone who is not looking at a screen listening to nothing,
with no way to know the machine was waiting on them.

Speaking the question is half of it. The other half is that the reply has to
reach the thing that is waiting, rather than starting a new turn while the
agent sits on a future nobody will resolve.
"""

from __future__ import annotations

import asyncio

import pytest

from roost.voice.answers import Answer, how_to_answer, interpret
from roost.voice.pipeline import Waiting, _spoken_approval, _spoken_question

# -- reading a reply --------------------------------------------------------


@pytest.mark.parametrize('said', ['yes', 'yeah', 'yep', 'sure', 'go ahead', 'do it',
                                  'ok', 'please do', 'approve', 'carry on', 'confirm'])
def test_an_affirmative_is_understood(said):
    assert interpret(said) is Answer.YES


@pytest.mark.parametrize('said', ['no', 'nope', 'nah', "don't", 'do not do that', 'stop',
                                  'cancel', 'deny', 'wait', 'never mind', 'skip it'])
def test_a_refusal_is_understood(said):
    assert interpret(said) is Answer.NO


@pytest.mark.parametrize('said', ['', '   ', 'maybe', 'what?', 'go', 'hmm', 'the weather'])
def test_anything_else_is_unclear_rather_than_a_guess(said):
    """"Go" and "no" are one phoneme apart. The likelier reading is not a good
    enough reason to run something."""
    assert interpret(said) is Answer.UNCLEAR
    assert interpret(said, strict=True) is Answer.UNCLEAR


def test_a_refusal_wins_when_both_appear():
    """"Don't do it" contains "do it". Refusing on an ambiguous utterance costs
    a repeat; approving on one costs whatever the tool was about to do."""
    assert interpret("don't do it") is Answer.NO
    assert interpret('no, go ahead') is Answer.NO


def test_money_needs_a_word_that_is_not_a_mishearing():
    """A bare "yes" is one transcription error from a purchase, and "yes" is a
    word people say while thinking. "Confirm" is neither."""
    assert interpret('yes', strict=True) is Answer.UNCLEAR
    assert interpret('yeah go ahead', strict=True) is Answer.UNCLEAR
    assert interpret('confirm', strict=True) is Answer.YES
    assert interpret('yes, confirm the purchase', strict=True) is Answer.YES
    # And a refusal still stops it, whatever else is in the sentence.
    assert interpret('no, do not confirm that', strict=True) is Answer.NO


def test_the_person_is_told_what_will_be_understood():
    assert 'confirm' in how_to_answer(strict=True)
    assert 'yes' in how_to_answer(strict=False)


# -- what gets said aloud ---------------------------------------------------


def test_a_question_reads_its_options_out():
    """On a screen they are buttons. In a voice call they are the only way to
    know what answers are accepted."""
    said = _spoken_question('Which hotel?', ['Metekhi View', 'Rooms Tbilisi'])
    assert 'Which hotel?' in said
    assert 'Metekhi View' in said and 'Rooms Tbilisi' in said


def test_a_long_option_list_is_not_read_out():
    """Nine options read aloud is worse than none."""
    said = _spoken_question('Pick one', [f'option {n}' for n in range(9)])
    assert 'option 5' not in said
    assert '9 options' in said


def test_an_approval_says_what_it_is_and_how_to_answer():
    ordinary = _spoken_approval('run npm test', strict=False)
    assert 'run npm test' in ordinary
    assert 'yes' in ordinary

    money = _spoken_approval('click "Book this room — pay now"', strict=True)
    assert 'spends money' in money
    assert 'confirm' in money


def test_an_approval_with_no_summary_still_says_something():
    assert _spoken_approval('', strict=False).strip()


# -- the routing ------------------------------------------------------------


class FakeAgent:
    """Records what the pipeline did to it."""

    def __init__(self):
        self.answered: list[tuple[str, str]] = []
        self.approved: list[str] = []
        self.denied: list[tuple[str, str]] = []

    def answer(self, question_id, text):
        self.answered.append((question_id, text))
        return True

    def approve(self, call_id, remember=False):
        self.approved.append(call_id)
        return True

    def deny(self, call_id, reason=''):
        self.denied.append((call_id, reason))
        return True


def session(agent):
    """A VoiceSession with nothing real behind it.

    Built without calling `__init__`, because everything that matters here is
    `_answer`, and the constructor would want an STT, a TTS and a model.
    """
    from roost.voice.pipeline import VoiceSession

    voice = VoiceSession.__new__(VoiceSession)
    voice.agent = agent
    voice._awaiting = None
    voice.said: list[str] = []

    async def emit_event(event):
        return None

    async def say_aloud(text):
        voice.said.append(text)

    voice.emit_event = emit_event
    voice.say_aloud = say_aloud
    return voice


@pytest.mark.asyncio
async def test_an_answer_reaches_the_question_not_a_new_turn():
    """Otherwise the person replies and it becomes a fresh instruction while
    the agent sits on a future nobody is going to resolve."""
    agent = FakeAgent()
    voice = session(agent)
    voice._awaiting = Waiting(kind='question', id='q1', options=['a', 'b'])

    await voice._answer('the second one')

    assert agent.answered == [('q1', 'the second one')]
    assert voice._awaiting is None, 'it should stop waiting once answered'


@pytest.mark.asyncio
async def test_a_spoken_yes_approves_and_a_no_denies():
    agent = FakeAgent()
    voice = session(agent)

    voice._awaiting = Waiting(kind='approval', id='c1', summary='run npm test')
    await voice._answer('yes go ahead')
    assert agent.approved == ['c1']

    voice._awaiting = Waiting(kind='approval', id='c2', summary='delete it')
    await voice._answer('no, stop')
    assert [c for c, _ in agent.denied] == ['c2']


@pytest.mark.asyncio
async def test_an_unclear_reply_asks_again_and_keeps_waiting():
    """It does not approve, it does not deny, and it does not silently drop
    the question — all three of which would be worse than asking twice."""
    agent = FakeAgent()
    voice = session(agent)
    voice._awaiting = Waiting(kind='approval', id='c1', summary='run npm test')

    await voice._answer('uh, hmm')

    assert agent.approved == [] and agent.denied == []
    assert voice._awaiting is not None, 'the question must stay open'
    assert voice.said and 'did not catch' in voice.said[0]


@pytest.mark.asyncio
async def test_a_bare_yes_does_not_spend_money():
    """The one that matters. A misheard syllable must not buy anything."""
    agent = FakeAgent()
    voice = session(agent)
    voice._awaiting = Waiting(kind='approval', id='buy', strict=True,
                              summary='click "Place your order"')

    await voice._answer('yes')
    assert agent.approved == [], 'a bare yes bought something'
    assert voice._awaiting is not None

    await voice._answer('confirm')
    assert agent.approved == ['buy']


@pytest.mark.asyncio
async def test_a_refusal_stops_a_purchase_immediately():
    agent = FakeAgent()
    voice = session(agent)
    voice._awaiting = Waiting(kind='approval', id='buy', strict=True, summary='pay now')

    await voice._answer('no')
    assert [c for c, _ in agent.denied] == ['buy']
    assert voice._awaiting is None


# -- the whole path, with a real agent --------------------------------------


class Silence:
    """A synthesiser that produces nothing, and records what it was asked to say."""

    output_sample_rate = 24_000

    def __init__(self):
        self.spoken: list[str] = []

    async def synthesize(self, text, *, model, voice, fmt='pcm16', sample_rate=16_000):
        self.spoken.append(text)
        return
        yield b''       # pragma: no cover - makes this an async generator

    async def voices(self):
        return []


@pytest.mark.asyncio
async def test_a_suspended_turn_speaks_and_then_finishes_when_answered(tmp_path):
    """The whole path, against a real AgentSession.

    A model that calls a tool needing approval, the pipeline speaking that
    fact, a spoken "yes" routed back, and the turn *finishing* — which is the
    part that would break if the generator stopped when the turn suspended,
    because the agent would be left waiting on a future nobody resolves.
    """
    from roost.agent.approval import Mode
    from roost.agent.runtime import build_session
    from roost.agent.tools.base import Assessment, Output, Tool
    from roost.protocol.agent import Risk
    from roost.providers.base import StreamDone, StreamText, StreamToolUse
    from roost.voice.pipeline import VoiceConfig, VoiceSession
    from tests.test_agent import ScriptedProvider

    class Spender(Tool):
        name = 'buy_it'
        description = 'Buy the thing.'
        input_schema = {'type': 'object', 'properties': {}}

        def __init__(self):
            self.ran = 0

        def assess(self, args, ctx):
            return Assessment(risk=Risk.PURCHASE, summary='place the order, 96 GEL')

        async def run(self, args, ctx):
            self.ran += 1
            return Output(content='ordered')

    tool = Spender()
    provider = ScriptedProvider([
        [StreamToolUse(id='c1', name='buy_it', input={}), StreamDone(stop_reason='tool_use')],
        [StreamText(text='Done, that is booked.'), StreamDone()],
    ])

    agent = build_session(
        root=str(tmp_path), provider=provider, model='x', mode=Mode.TRUSTED, tools=[tool],
    )
    await agent.start()

    tts = Silence()
    events: list = []

    async def emit_event(event):
        events.append(event)

    async def emit_audio(utterance_id, chunk):
        return None

    voice = VoiceSession(
        stt=None, tts=tts, llm=provider, config=VoiceConfig(), emit_event=emit_event,
        emit_audio=emit_audio, agent=agent,
    )

    await voice.say('book the cheap one')

    # It should be speaking the approval and waiting, not finished.
    for _ in range(200):
        if voice._awaiting is not None:
            break
        await asyncio.sleep(0.01)

    assert voice._awaiting is not None, 'the turn suspended without saying so'
    assert voice._awaiting.strict is True, 'a purchase should need the strict answer'
    said = ' '.join(tts.spoken)
    assert 'place the order' in said, f'it did not say what it was waiting on: {said!r}'
    assert 'confirm' in said, 'it did not say how to answer'
    assert tool.ran == 0

    # A bare yes must not buy it.
    await voice._answer('yes')
    assert tool.ran == 0, 'a bare yes bought something'
    assert voice._awaiting is not None

    # And "confirm" finishes the turn.
    await voice._answer('confirm')
    for _ in range(300):
        if tool.ran:
            break
        await asyncio.sleep(0.01)

    assert tool.ran == 1, 'the approval never reached the suspended turn'
    assert voice._awaiting is None

    if voice._reply is not None:
        await asyncio.wait_for(asyncio.shield(voice._reply), timeout=10)
    assert 'Done, that is booked.' in ' '.join(tts.spoken), 'the turn did not carry on'

    await agent.close()
