"""What Roost asks the Messages API for, and the three ways that shape moved.

None of these rules is visible in a response body, which is why they are tested
directly rather than left to a live call to exercise:

* a missing ``display`` returns thinking blocks that are present, billed, and
  EMPTY — a 200 with a blank working-out panel behind it;
* ``budget_tokens`` and an unknown effort level are each a 400;
* sampling parameters are a 400 on the current models rather than an ignore;
* and ``{'type': 'disabled'}`` produces the worst failure available to an agent
  harness — a tool call written into visible text, so the turn succeeds, the
  call never runs, and nothing reports it.
"""

from __future__ import annotations

import ast
from pathlib import Path

from roost.providers.anthropic import _TRANSLATED_EXTRAS, reasoning_params
from roost.providers.base import ChatRequest

CURRENT = 'claude-opus-5'
OLD = 'claude-3-5-sonnet-20241022'


def req(model: str = CURRENT, **kw) -> ChatRequest:
    return ChatRequest(model=model, messages=[], **kw)


def test_reasoning_is_asked_to_be_shown():
    # Without `display`, the blocks arrive empty and StreamThinking emits empty
    # strings — the panel stays blank through a long turn with nothing to say
    # why. `omitted` is the API's default now, a silent change from Opus 4.6.
    params = reasoning_params(req())
    assert params['thinking'] == {'type': 'adaptive', 'display': 'summarized'}


def test_no_token_budget_is_ever_sent():
    # `{'type': 'enabled', 'budget_tokens': N}` is a 400 on the current models.
    for r in (req(), req(extra={'think': False}), req(extra={'think': 'max'})):
        assert 'budget_tokens' not in repr(reasoning_params(r))
        assert reasoning_params(r)['thinking']['type'] == 'adaptive'


def test_thinking_is_never_disabled_even_when_the_caller_asks():
    # The voice path sets think: False because reasoning is silence in a call.
    # It must not become {'type': 'disabled'}: with thinking off the model
    # sometimes writes a tool call into its visible text instead of a tool_use
    # block, and the turn then succeeds with the call never having run.
    params = reasoning_params(req(extra={'think': False}))
    assert params['thinking']['type'] == 'adaptive'
    assert params['thinking']['display'] == 'omitted'
    # What the caller actually wanted was latency, and that is what they get.
    assert params['output_config'] == {'effort': 'low'}


def test_an_effort_level_rides_along_and_a_bogus_one_does_not():
    assert reasoning_params(req(extra={'think': 'xhigh'}))['output_config'] == {'effort': 'xhigh'}
    assert reasoning_params(req(extra={'effort': 'max'}))['output_config'] == {'effort': 'max'}
    # Unknown levels are a 400, and `high` is already this API's default, so
    # dropping one is better than inventing or forwarding it.
    assert 'output_config' not in reasoning_params(req(extra={'think': 'exhaustive'}))


def test_sampling_goes_only_to_models_that_still_take_it():
    # Removed on the current generation: a 400, not an ignore, so one setting
    # turns into a failed request.
    assert 'temperature' not in reasoning_params(req(temperature=0.5))
    assert reasoning_params(req(OLD, temperature=0.5))['temperature'] == 0.5
    # And nothing is invented when the caller set none.
    assert 'temperature' not in reasoning_params(req(OLD))


def test_the_keys_this_adapter_translates_are_held_back_from_the_splat():
    # `ChatRequest.extra` is splatted into the payload. `think` is Ollama's
    # spelling and has no meaning here — passed through it is an unknown
    # top-level parameter, which this API answers with a 400. That is what
    # broke the voice path against Anthropic outright.
    assert 'think' in _TRANSLATED_EXTRAS
    assert 'effort' in _TRANSLATED_EXTRAS


def _code_strings(path: Path) -> list[str]:
    """Every string literal in a module that is not a docstring.

    On the syntax tree rather than the text, and the reason is concrete: the
    first version of this stripped comment lines and then flagged the word
    'disabled' inside `reasoning_params`' own docstring — the paragraph that
    exists to explain why the adapter must never send it. Prose describing a
    hazard sits exactly where a text scan looks for the hazard, and dropping
    comments still leaves docstrings, which are strings and not comments.
    """
    tree = ast.parse(path.read_text(encoding='utf-8'))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    return [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
    ]


def test_the_adapter_never_writes_a_disabled_thinking_type():
    # A source check, because the danger is a future edit rather than current
    # behaviour: `{'type': 'disabled'}` is the obvious way to honour a
    # think: False, and it is the one that breaks tool calling silently. If it
    # ever has to change, it should have to change here too.
    src = Path(__file__).resolve().parents[1] / 'roost' / 'providers' / 'anthropic.py'
    assert len(src.read_text(encoding='utf-8')) > 500, 'the adapter source was not read properly'
    literals = _code_strings(src)

    assert 'disabled' not in literals, (
        "the adapter sets thinking type 'disabled', which makes the model write "
        'tool calls into visible text instead of tool_use blocks'
    )
    # Positive control, in the same test: the scan must still be finding real
    # literals from this file, or "no 'disabled'" is what an empty scan says.
    assert 'adaptive' in literals, 'the scan found no literals — it has stopped working'
    assert 'summarized' in literals


def test_the_literal_scan_can_tell_code_from_prose():
    # The two halves that defeated the text version, in one fixture: a
    # docstring naming the hazard (must NOT be found) and a real assignment
    # using it (must be found). A scan that reads prose reports the first; a
    # scan that has died reports neither.
    import tempfile
    fixture = (
        'def f():\n'
        '    """Never send {\'type\': \'disabled\'} here."""\n'
        "    return {'type': 'adaptive'}\n"
    )
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as fh:
        fh.write(fixture)
        tmp = Path(fh.name)
    try:
        found = _code_strings(tmp)
        assert 'disabled' not in found, 'prose in a docstring was read as code'
        assert 'adaptive' in found, 'the real literal was missed'
    finally:
        tmp.unlink()
