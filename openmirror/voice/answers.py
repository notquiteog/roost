"""Reading a spoken reply as yes, no, or neither.

Neither is the important one. A voice approval is a transcript of a noisy
room, and the failure that matters is not misunderstanding — it is *guessing*.
"Go" and "no" are one phoneme apart; "don't" and "do" survive a dropped
consonant as each other. So anything that is not clearly one or the other is
`UNCLEAR`, and the caller asks again rather than picking the likelier reading.
Nothing here ever defaults to yes.

Money and secrets are stricter still, and deliberately not by requiring a
longer sentence — a long sentence is more transcript to go wrong, not less.
They require one distinct word, **confirm**, which is not a word that turns up
in conversation by accident and is not a plausible mishearing of "no". A bare
"yes" is not enough to spend money, and the prompt says so before it is
needed rather than after it was not given.
"""

from __future__ import annotations

import re
from enum import StrEnum


class Answer(StrEnum):
    YES = 'yes'
    NO = 'no'
    UNCLEAR = 'unclear'


# Checked before the affirmatives, always: "don't do it" contains "do it", and
# "no, go ahead" is a thing people say but "no" is the part that must win when
# the two collide. Refusing on an ambiguous utterance costs a repeat; approving
# on one costs whatever the tool was about to do.
_NO = re.compile(
    r'\b(?:no|nope|nah|negative|stop|cancel|don\'?t|do not|deny|denied|refuse|'
    r'abort|wait|hold on|not now|never mind|nevermind|skip(?: it)?)\b',
    re.I,
)

_YES = re.compile(
    r'\b(?:yes|yeah|yep|yup|sure|ok|okay|okey|fine|please do|go ahead|go for it|'
    r'do it|carry on|continue|proceed|approve[d]?|allow(?:ed)?|permit(?:ted)?|'
    r'sounds good|that\'?s right|correct|confirm(?:ed)?)\b',
    re.I,
)

# The one word that stands between a misheard syllable and somebody's card.
_CONFIRM = re.compile(r'\bconfirm(?:ed|s|ing)?\b', re.I)


def interpret(said: str, *, strict: bool = False) -> Answer:
    """What a spoken reply means.

    `strict` is for anything that spends money or types a secret: an
    affirmative then has to contain "confirm", and a bare "yes" reads as
    UNCLEAR — which is a repeat of the question, not a refusal, because the
    person almost certainly meant yes and should be told what to say.
    """
    text = ' '.join((said or '').split())
    if not text:
        return Answer.UNCLEAR

    if _NO.search(text):
        return Answer.NO

    if strict:
        return Answer.YES if _CONFIRM.search(text) else Answer.UNCLEAR

    return Answer.YES if _YES.search(text) else Answer.UNCLEAR


def how_to_answer(strict: bool) -> str:
    """What to say aloud so the person knows what will be understood."""
    if strict:
        return 'Say "confirm" to go ahead, or "no" to stop.'
    return 'Say "yes" to go ahead, or "no" to stop.'
