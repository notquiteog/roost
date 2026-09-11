"""No ceiling by default, and the three ways an API is told so.

`ChatRequest.max_tokens` defaults to 0, meaning no limit. Each adapter has to
express that in its own API's terms, and the rule is the same everywhere:
**omit the field**. A large stand-in number is still a ceiling — a less visible
one, and the wrong one on any model whose real limit differs.

Anthropic is the exception, because `max_tokens` is required there. It sends
the model's own maximum, read from the Models API and remembered per model.

── Why some of this is a source scan ──────────────────────────────────────

Because a wrongly-sent ceiling does not fail. The request succeeds, the
response is well-formed, and the truncation shows up as a `stop_reason` nobody
reads — so it looks exactly like the model deciding to stop. There is no
response to assert against, the same argument the transport guard makes about
proxies. What can be checked is that the field is written under a condition.

The absence assertions carry a positive control in the same test: one fixture
holding a correctly guarded write and an unguarded one, asserting exactly which
comes back. An empty result is otherwise indistinguishable from a scan that has
stopped matching.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from openmirror.providers.base import ChatRequest


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'openmirror' / 'providers' / 'base.py').exists():
            return parent
    raise AssertionError(f'cannot find the source tree, starting from {here}')


def source(rel: str) -> str:
    text = (repo_root() / rel).read_text(encoding='utf-8')
    assert len(text) > 300, f'{rel} is too short to have been read properly'
    return text


def test_the_default_is_no_ceiling():
    # The value the rest of this file is about. 4096 before, which was a guess
    # about how long an agent turn takes.
    assert ChatRequest(model='m', messages=[]).max_tokens == 0


def test_a_ceiling_that_is_set_is_still_honoured():
    # Uncapped by default must not mean uncappable: an operator or a caller
    # that sets one is making a decision, and every adapter still sends it.
    assert ChatRequest(model='m', messages=[], max_tokens=512).max_tokens == 512


#: A line that writes the ceiling into a request payload.
#:
#: Two forms, because the adapters use both and an earlier version of this
#: matched only the first — which made the scan blind to two of the three
#: files it exists to check, while still reporting clean. The positive control
#: below is what caught that:
#:
#:   * a dict literal   -- ``'max_tokens': value``
#:   * a subscript assign -- ``payload['max_tokens'] = value``
WRITES_CEILING = re.compile(r"""(['"])(max_tokens|num_predict)\1\s*(?:\]\s*=|:)""")

#: The guard written on the write itself: `req.max_tokens or <fallback>`.
INLINE_GUARD = re.compile(r'req\.max_tokens\s+or\s+')

#: The guard written as an enclosing statement.
BLOCK_GUARD = re.compile(r'^\s*if\s+req\.max_tokens\s*:')


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def unguarded_ceiling_writes(text: str) -> list[str]:
    """Ceiling writes that nothing makes conditional.

    A guard counts only if it actually governs THIS write: either it is on the
    same line, or it is an `if req.max_tokens:` at a **smaller indentation**,
    which is what enclosing it means in Python.

    The indentation test is the point. A plain three-line window accepted the
    neighbouring line's guard as this one's -- two adjacent entries in the same
    dict literal, one correctly guarded, and the unguarded one directly below
    it was excused. That is the same mistake in the opposite direction from a
    window too small to see a real guard, and only the block structure rules
    out both.
    """
    out: list[str] = []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.lstrip().startswith('#'):
            continue
        if not WRITES_CEILING.search(line):
            continue
        if INLINE_GUARD.search(line):
            continue
        here = _indent(line)
        # Walk outwards: only a line indented LESS than this one can enclose
        # it, and the first such line decides. Stop at the enclosing `def`.
        guarded = False
        for j in range(i - 1, max(-1, i - 12), -1):
            prev = lines[j]
            if not prev.strip():
                continue
            if _indent(prev) >= here:
                continue
            if BLOCK_GUARD.match(prev):
                guarded = True
            break
        if not guarded:
            out.append(line.strip())
    return out


def test_the_scanner_tells_a_guarded_write_from_a_bare_one():
    # Positive control, beside the real scan. The guarded forms are the
    # important half: a scanner reading too loosely flags them, and a scanner
    # that has stopped matching reports every half clean.
    fixture = '\n'.join([
        '    async def stream(self, req):',
        "        payload['max_tokens'] = req.max_tokens",           # bare
        '        if req.max_tokens:',
        "            payload['max_tokens'] = req.max_tokens",       # guarded (enclosed)
        '        payload = {',
        "            'max_tokens': req.max_tokens or self._limit(m),",  # guarded (inline)
        "            'num_predict': req.max_tokens,",               # bare, and the
        # important one: it sits directly under a correctly guarded line at the
        # SAME indentation. A window-based scan reads that neighbour's guard as
        # this line's and stays quiet -- a false negative, which is the failure
        # a decoy-only fixture cannot see.
        '        }',
    ])
    caught = unguarded_ceiling_writes(fixture)
    assert len(caught) == 2, caught
    assert any('num_predict' in c for c in caught), caught
    assert any("payload['max_tokens'] = req.max_tokens" == c for c in caught), caught


@pytest.mark.parametrize('rel', [
    'openmirror/providers/ollama.py',
    'openmirror/providers/openai_compat.py',
    'openmirror/providers/anthropic.py',
])
def test_no_adapter_sends_a_ceiling_unconditionally(rel):
    found = unguarded_ceiling_writes(source(rel))
    assert not found, (
        f'{rel} writes a token ceiling with nothing checking whether one was '
        f'asked for. Sent when the caller wanted none, it does not fail — the '
        f'answer is simply cut short with the stop reason buried:\n'
        + '\n'.join(found)
    )


def test_the_scan_is_looking_at_files_that_really_contain_these_writes():
    # Two ways this suite goes vacuous, and the first fix does not cover the
    # second: the wrong directory, and a pattern that has stopped matching a
    # rewritten file. The parametrised test above passes in both cases.
    hits = sum(len(WRITES_CEILING.findall(source(rel))) for rel in (
        'openmirror/providers/ollama.py',
        'openmirror/providers/openai_compat.py',
        'openmirror/providers/anthropic.py',
    ))
    assert hits >= 3, f'expected each adapter to write a ceiling somewhere; found {hits}'


def test_anthropic_asks_the_api_rather_than_guessing():
    # This adapter cannot omit the field, so it needs a number. The number has
    # to come from the model, because one ABOVE the real ceiling is a 400 —
    # a broken adapter rather than a short answer — and ceilings differ per
    # model and move each generation.
    text = source('openmirror/providers/anthropic.py')
    # `_model_info` since it also reads the capability tree from the same
    # reply — which thinking shapes the model accepts — so the name says both.
    assert '_model_info' in text
    assert '/models/' in text, 'the ceiling is not read from the Models API'
    # And the fallback must err downwards, for the same reason.
    m = re.search(r'_FALLBACK_MAX_OUTPUT\s*=\s*(\d+)', text)
    assert m, 'no documented fallback ceiling'
    assert int(m.group(1)) <= 8192, 'a fallback above the smallest real ceiling risks a 400'


def test_the_voice_path_does_not_reimpose_one():
    # A spoken reply that stops mid-sentence has no scrollback and no visible
    # stop reason — just a voice trailing off. It carried max_tokens=1024.
    text = source('openmirror/voice/pipeline.py')
    assert not re.search(r'max_tokens\s*=\s*\d+', text), (
        'the voice pipeline sets a fixed token ceiling again'
    )
