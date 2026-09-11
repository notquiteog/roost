"""How hard the model thinks, in the spelling the server on the other end accepts.

openmirror has one dial — ``off`` … ``max`` — and the servers a connection can
point at share the OpenAI request shape while agreeing about nothing on this
one field:

=================  =====================================================
api.openai.com     ``reasoning_effort``, on a ladder that differs per model
OpenRouter         ``reasoning: {effort}``, which OpenRouter maps itself
Groq               ``reasoning_effort``, with a different set per model
Fireworks          ``reasoning_effort`` none|low|medium|high, on any model
Together           ``reasoning: {enabled}`` on hybrids, effort on gpt-oss
SiliconFlow        ``enable_thinking`` + ``thinking_budget``
Alibaba (Qwen)     ``enable_thinking`` + ``thinking_budget``
Gemini (compat)    ``reasoning_effort``, and 2.5 Pro / 3.x cannot be off
=================  =====================================================

On several of them the wrong value is not ignored. Groq answers 400 to
``reasoning_effort: "low"`` on a Qwen that takes only none|default; OpenAI to
``none`` on GPT-6 Astra and to any effort at all on a model that does not
reason. So the dial stays one dial and this is the only place it is
translated. Which host is read off the address, because that is the one thing
about a connection that cannot be mislabelled.

Two rules every table follows:

* **Never above what was asked, never a value the model refuses.** A level a
  model lacks clamps DOWN to the nearest one it has, so ``max`` on a local
  model is its hardest setting rather than a 400. The exception is a model
  whose lowest rung is above the ask — ``gpt-5-pro`` takes only ``high``.
* **``off`` means as little as the model allows.** On a model with no off
  switch that is its lowest rung, not its default.

A model this does not recognise, on a host that needs telling per model, is
sent nothing: the host's own default is a correct answer, and an invented
parameter might be a 400.

``None`` — no level chosen — is different from ``off`` throughout: it leaves
every host at its own default, which is what openmirror did before a level
existed.

The same tables live in tern's ``ai/reasoning.ts`` and cryptostore's
``services/ai/reasoning.js``; a host added to one belongs in all three.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

#: The dial, in order.
LEVELS: tuple[str, ...] = ('off', 'low', 'medium', 'high', 'xhigh', 'max')

#: Every effort name any host uses, in order of how hard it thinks.
ORDER: tuple[str, ...] = ('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')

_THREE: tuple[str, ...] = ('low', 'medium', 'high')
_EFFORTS: tuple[str, ...] = ('low', 'medium', 'high', 'xhigh', 'max')

#: Thinking allowance in tokens, for the hosts that take a budget rather than a
#: level. Only ever sent because a level was chosen.
BUDGETS: dict[str, int] = {'low': 1024, 'medium': 4096, 'high': 16384, 'xhigh': 32768}


#: Keys of ``ChatRequest.extra`` that are this dial under older spellings, so
#: an adapter translates them rather than splatting them into a body — where
#: ``think`` is an unknown parameter to every host but Ollama.
TRANSLATED_EXTRAS = frozenset({'think', 'effort'})


def normalise(value: object) -> str | None:
    """A level from anything a caller might hold, or None for "no opinion".

    ``False`` is what the voice path has always sent (reasoning is silence in a
    call). ``True`` is Ollama's "think, at your default", which is what None
    already means everywhere else.
    """
    if value is False:
        return 'off'
    if isinstance(value, str):
        v = value.strip().lower()
        if v in LEVELS:
            return v
    return None


def request_level(effort: object, extra: dict[str, Any] | None) -> str | None:
    """The level one request asks for: its ``effort``, else the older spellings in ``extra``."""
    if effort is not None:
        return normalise(effort)
    extra = extra or {}
    if 'think' in extra:
        return normalise(extra['think'])
    return normalise(extra.get('effort'))


def _bare(model: str) -> str:
    """The last path segment, lower-cased: ``openai/gpt-5`` and ``Qwen/Qwen3-8B`` alike."""
    return (model or '').strip().lower().rsplit('/', 1)[-1]


def dialect_for(base_url: str) -> str:
    """Which host's rules apply, from the address alone.

    ``generic`` is everything else — vLLM, llama.cpp, LM Studio, Open WebUI,
    Perch — which gets the conservative treatment in ``_generic``.
    """
    try:
        host = (urlparse(base_url or '').hostname or '').lower()
    except ValueError:
        host = ''
    if host == 'api.openai.com':
        return 'openai'
    if host == 'openrouter.ai' or host.endswith('.openrouter.ai'):
        return 'openrouter'
    if host == 'api.groq.com':
        return 'groq'
    if host == 'api.fireworks.ai':
        return 'fireworks'
    if host in ('api.together.xyz', 'api.together.ai'):
        return 'together'
    if host in ('api.siliconflow.com', 'api.siliconflow.cn'):
        return 'siliconflow'
    # Model Studio moved to per-workspace hosts under maas.aliyuncs.com, and
    # the regional dashscope*.aliyuncs.com ones still answer. The suffix
    # covers both, in every region.
    if host.endswith('.aliyuncs.com'):
        return 'dashscope'
    if host == 'generativelanguage.googleapis.com':
        return 'gemini'
    if host == 'api.deepseek.com':
        return 'deepseek'
    return 'generic'


def clamp_effort(level: str, ladder: tuple[str, ...] | None) -> str | None:
    """The nearest rung a model accepts, never above ``level``.

    ``ladder`` is ascending in ORDER; ``none`` and ``minimal`` are only for ``off``.
    """
    if not ladder:
        return None
    if level == 'off':
        if 'none' in ladder:
            return 'none'
        if 'minimal' in ladder:
            return 'minimal'
        return ladder[0]
    want = ORDER.index(level)
    graded = [r for r in ladder if r not in ('none', 'minimal')]
    best = None
    for rung in graded:
        if ORDER.index(rung) <= want:
            best = rung
    return best or (graded[0] if graded else ladder[0])


# ---------- api.openai.com ----------


def openai_ladder(model: str) -> tuple[str, ...] | None:
    """Which efforts one OpenAI model takes, or None for a model that does not reason.

    As OpenAI documents it in September 2026: the 5.6 family takes none … max,
    GPT-6 Astra takes low … max and refuses ``none``, 5.2–5.5 stop at xhigh, 5.1
    at high, and the original gpt-5 has ``minimal`` where later ones have
    ``none``. An unrecognised later generation gets the three rungs every
    reasoning model since o1 has taken, which is never a 400.
    """
    m = _bare(model)
    if m.startswith('gpt-oss'):
        return _THREE
    if re.match(r'gpt-5(?:\.\d+)?-chat', m):
        return None
    if m.startswith('gpt-5-pro'):
        return ('high',)
    if re.match(r'gpt-5\.\d+-pro', m):
        return ('medium', 'high', 'xhigh')
    if m.startswith('gpt-6') or re.match(r'gpt-5\.\d+-cyber', m):
        return _EFFORTS
    minor = re.match(r'gpt-5\.(\d+)', m)
    if minor:
        n = int(minor.group(1))
        if n >= 6:
            return ('none', *_EFFORTS)
        if n >= 2:
            return ('none', 'low', 'medium', 'high', 'xhigh')
        return ('none', *_THREE)
    if re.match(r'gpt-5(?:$|-)', m):
        return ('minimal', *_THREE)
    if re.match(r'o[1-9](?:$|-)', m):
        return _THREE
    if re.match(r'gpt-(?:[7-9]|\d{2})', m):
        return _THREE
    return None


# ---------- the other OpenAI-shaped hosts ----------

_NO_SWITCH = re.compile(r'thinking|instruct|coder|(?:^|[-/])r1\b|qwq')
_TOGETHER_HYBRID = re.compile(r'deepseek-v(?:3\.[1-9]|[4-9])|glm-(?:4\.[5-9]|[5-9])|kimi-k[2-9]|qwen3|minimax-m[1-9]')
_SF_HYBRID = re.compile(r'qwen3|glm-|hunyuan|deepseek-v3\.[1-9]|deepseek-v[4-9]')
_DS_HYBRID = re.compile(r'qwen3|qwen-plus|qwen-flash|qwen-turbo|deepseek-v3\.[1-9]|deepseek-v[4-9]|kimi-k2|glm-')
_DS_OPEN_QWEN3 = re.compile(r'^qwen3(?:\.\d+)?-(?:next-)?\d+(?:\.\d+)?b')
_GENERIC_REASONERS = re.compile(r'^(?:o[1-9](?:-|$)|gpt-[5-9]|gpt-oss|deepseek|qwen|qwq|glm|minimax|kimi)')


def _groq(level: str, model: str) -> dict[str, Any]:
    """A different set per model, and a value outside it is a 400.

    ``reasoning_format: 'parsed'`` on Qwen and MiniMax is not decoration: with
    ``raw`` the reasoning arrives inside ``<think>`` tags in the ANSWER, and
    ``raw`` with tools is refused outright. GPT-OSS takes no format at all.
    """
    m = _bare(model)
    if 'gpt-oss' in m:
        return {'reasoning_effort': clamp_effort(level, _THREE)}
    if 'qwen' in m:
        graded = re.search(r'qwen3\.(?:[89]|\d{2,})', m) is not None
        effort = 'none' if level == 'off' else (clamp_effort(level, _THREE) if graded else 'default')
        return {'reasoning_effort': effort, 'reasoning_format': 'parsed'}
    if 'minimax' in m:
        return {'reasoning_format': 'parsed'}
    return {}


def _together(level: str, model: str) -> dict[str, Any]:
    m = _bare(model)
    if 'gpt-oss' in m:
        return {'reasoning_effort': clamp_effort(level, _THREE)}
    if _TOGETHER_HYBRID.search(m) and not _NO_SWITCH.search(m):
        return {'reasoning': {'enabled': level != 'off'}}
    return {}


def _siliconflow(level: str, model: str) -> dict[str, Any]:
    """A switch on hybrids and a budget (128–32,768) on anything that reasons.

    SiliconFlow's default budget is 4,096, not its ceiling, so ``max`` sends
    the ceiling rather than nothing.
    """
    m = _bare(model)
    hybrid = bool(_SF_HYBRID.search(m)) and not _NO_SWITCH.search(m)
    reasoning_only = re.search(r'(?:^|[-/])r1\b|thinking|qwq', m) is not None
    out: dict[str, Any] = {}
    if hybrid:
        out['enable_thinking'] = level != 'off'
    spend = ('low' if reasoning_only else None) if level == 'off' else level
    if spend and (hybrid or reasoning_only):
        out['thinking_budget'] = 32768 if spend == 'max' else BUDGETS[spend]
    return out


def _dashscope(level: str, model: str, stream: bool) -> dict[str, Any]:
    """The same switch and budget, with two differences that each fail a request if missed.

    Some hybrids think by default and some do not (qwen3.6-plus on, qwen3-max
    off), so ``off`` is always said out loud; and the open-source Qwen3 builds
    refuse a NON-streamed call with thinking on. The default budget here IS the
    model's ceiling, so ``max`` sends none.
    """
    m = _bare(model)
    reasoning_only = re.search(r'qwq|(?:^|[-/])(?:deepseek-)?r1\b|thinking', m) is not None
    hybrid = bool(_DS_HYBRID.search(m)) and not _NO_SWITCH.search(m)
    out: dict[str, Any] = {}
    thinks = reasoning_only
    if hybrid:
        thinks = level != 'off' and (stream or not _DS_OPEN_QWEN3.search(m))
        out['enable_thinking'] = thinks
    if thinks:
        spend = 'low' if level == 'off' else level
        if spend != 'max':
            out['thinking_budget'] = BUDGETS[spend]
    return out


def gemini_ladder(model: str) -> tuple[str, ...] | None:
    """``none`` only on the 2.5 Flash models; 2.5 Pro and every 3.x always think."""
    m = _bare(model).removeprefix('models/')
    if not m.startswith('gemini-') or re.match(r'gemini-(?:1|2\.0)', m):
        return None
    if m.startswith('gemini-2.5-flash'):
        return ('none', *_THREE)
    return _THREE


def _generic(level: str, model: str) -> dict[str, Any]:
    """vLLM, llama.cpp and friends accept or ignore the field rather than refusing it.

    Only a model that reasons does anything with it, so it goes to those
    families and nobody else, and never at ``off``.
    """
    if level == 'off' or not _GENERIC_REASONERS.search(_bare(model)):
        return {}
    return {'reasoning_effort': clamp_effort(level, _THREE)}


def openai_compat_reasoning(level: str | None, model: str, base_url: str = '', *, stream: bool = True) -> dict[str, Any]:
    """The reasoning half of an OpenAI-shaped body, for one host and one model.

    ``None`` sends nothing, leaving the host at its default.
    """
    lv = normalise(level)
    if lv is None:
        return {}
    dialect = dialect_for(base_url)
    if dialect == 'openai':
        effort = clamp_effort(lv, openai_ladder(model))
        return {'reasoning_effort': effort} if effort else {}
    if dialect == 'openrouter':
        # OpenRouter maps one effort onto every vendor behind it and drops the
        # field for models that do not reason.
        return {'reasoning': {'effort': 'none' if lv == 'off' else lv}}
    if dialect == 'groq':
        return _groq(lv, model)
    if dialect == 'fireworks':
        # Accepted on every model; Fireworks handles the ones that cannot use it.
        return {'reasoning_effort': 'none' if lv == 'off' else clamp_effort(lv, _THREE)}
    if dialect == 'together':
        return _together(lv, model)
    if dialect == 'siliconflow':
        return _siliconflow(lv, model)
    if dialect == 'dashscope':
        return _dashscope(lv, model, stream)
    if dialect == 'gemini':
        effort = clamp_effort(lv, gemini_ladder(model))
        return {'reasoning_effort': effort} if effort else {}
    if dialect == 'deepseek':
        # DeepSeek's own API picks reasoning by model name and has no field for it.
        return {}
    return _generic(lv, model)


def openai_takes_sampling(model: str) -> bool:
    """Whether an OpenAI-shaped call may carry temperature, top_p and ``stop``.

    OpenAI's reasoning models refuse them with a 400 rather than ignoring them.
    Decided by name rather than host, because the same models arrive through
    OpenRouter and other proxies under a vendor prefix and refuse there too.
    """
    return re.match(r'(?:o[1-9](?:-|$)|gpt-(?:[5-9]|\d{2}))', _bare(model)) is None


def openai_max_tokens_field(base_url: str) -> str:
    """OpenAI itself refuses ``max_tokens`` on reasoning models; most others know only that name."""
    return 'max_completion_tokens' if dialect_for(base_url) == 'openai' else 'max_tokens'


# ---------- Ollama ----------


def ollama_think(level: str | None, model: str = '') -> bool | str | None:
    """Ollama's ``think``: ``False`` or low|medium|high, or None for no opinion.

    gpt-oss IGNORES booleans and thinks at its default whatever ``False`` says,
    so ``off`` there is ``low``, the least it will do. Anything above high
    clamps to high.
    """
    lv = normalise(level)
    if lv is None:
        return None
    if lv == 'off':
        return 'low' if 'gpt-oss' in (model or '').lower() else False
    if lv in ('low', 'medium'):
        return lv
    return 'high'


# ---------- Anthropic ----------


def anthropic_family(model: str) -> dict[str, Any]:
    """What one Claude model accepts, from its id.

    Used until the Models API has said, and as the whole answer when it cannot
    be asked. Three request shapes, and every mismatch is a 400: adaptive +
    ``output_config.effort`` on 4.6 and later (``xhigh`` arriving at Opus 4.7,
    so 4.6 has four efforts); a token budget on Haiku 4.5, Opus 4.5 and older,
    which reject adaptive; and Fable/Mythos, which always think.
    """
    m = (model or '').lower()
    if re.match(r'claude-(?:fable|mythos)', m):
        return {'adaptive': True, 'budget': False, 'efforts': _EFFORTS, 'always_thinks': True}
    if re.match(r'claude-(?:opus-5|opus-4-[78]|sonnet-5)', m):
        return {'adaptive': True, 'budget': False, 'efforts': _EFFORTS}
    if re.match(r'claude-(?:opus|sonnet)-4-6', m):
        return {'adaptive': True, 'budget': True, 'efforts': ('low', 'medium', 'high', 'max')}
    if re.match(r'claude-opus-4-5', m):
        return {'adaptive': False, 'budget': True, 'efforts': _THREE}
    if re.match(r'claude-(?:haiku-4|sonnet-4|opus-4|3)', m):
        return {'adaptive': False, 'budget': True, 'efforts': ()}
    # A model released after this was written: the current generation's shape.
    return {'adaptive': True, 'budget': False, 'efforts': _EFFORTS}


def anthropic_capabilities(capabilities: Any) -> dict[str, Any] | None:
    """The same facts from ``GET /v1/models/{id}``'s capability tree, or None when it carried none."""
    if not isinstance(capabilities, dict):
        return None
    thinking = capabilities.get('thinking') or {}
    effort = capabilities.get('effort') or {}
    if not thinking and not effort:
        return None
    types = thinking.get('types') or {}

    def supported(node: Any) -> bool:
        return bool(isinstance(node, dict) and node.get('supported'))

    return {
        'adaptive': supported(types.get('adaptive')),
        'budget': supported(types.get('enabled')),
        'efforts': tuple(lv for lv in _EFFORTS if supported(effort.get(lv))) if supported(effort) else (),
    }


def anthropic_budget(level: str, max_tokens: int) -> int:
    """The thinking allowance on a budget-only model, always below ``max_tokens``."""
    ceiling = max_tokens if max_tokens > 0 else 8192
    want = ceiling // 2 if level == 'max' else BUDGETS.get(level, BUDGETS['medium'])
    return max(1024, min(want, ceiling // 2))


def anthropic_reasoning(
    model: str,
    level: str | None,
    *,
    capabilities: Any = None,
    max_tokens: int = 0,
    takes_sampling: bool = False,
) -> dict[str, Any]:
    """The reasoning half of a Messages API body.

    Returns ``{'body', 'thinking', 'sampling', 'min_max_tokens'}``: the fields
    to merge, whether the model deliberates on this call, whether temperature
    may be sent (never while thinking, which is a 400), and — for a budget-only
    model — what ``max_tokens`` must be at least.

    This is an agent harness, so ``off`` never becomes a disabled thinking
    type on a model that thinks adaptively. With thinking switched off that
    way the model sometimes writes a tool call into its VISIBLE TEXT instead of
    a tool_use block: the turn succeeds, the call never runs, nothing errors.
    Thinking stays on at the least effort with the working-out hidden, which
    gets the latency the caller wanted without the failure mode.
    """
    lv = normalise(level)
    fam = dict(anthropic_family(model))
    fam.update(anthropic_capabilities(capabilities) or {})
    efforts: tuple[str, ...] = tuple(fam['efforts'])
    ask = 'low' if lv == 'off' else lv
    effort = clamp_effort(ask, efforts) if efforts and ask else None

    def with_effort(body: dict[str, Any]) -> dict[str, Any]:
        return {**body, 'output_config': {'effort': effort}} if effort else body

    if lv == 'off':
        if fam.get('always_thinks'):
            # No off switch, so no thinking parameter at all; with no display
            # asked for the working-out comes back empty.
            return {'body': with_effort({}), 'thinking': False, 'sampling': False}
        if fam['adaptive']:
            return {
                'body': with_effort({'thinking': {'type': 'adaptive', 'display': 'omitted'}}),
                'thinking': False,
                'sampling': False,
            }
        # A budget-only model does not think unless asked to.
        return {'body': {}, 'thinking': False, 'sampling': takes_sampling}

    if fam['adaptive']:
        # `display: 'summarized'` is what makes the working-out non-empty; the
        # current models default to `omitted`, a silent change from 4.6. With
        # no level chosen, no effort is sent and the model's default applies.
        return {
            'body': with_effort({'thinking': {'type': 'adaptive', 'display': 'summarized'}}),
            'thinking': True,
            'sampling': False,
        }
    if fam['budget'] and lv is not None:
        budget = anthropic_budget(lv, max_tokens)
        return {
            'body': with_effort({'thinking': {'type': 'enabled', 'budget_tokens': budget}}),
            'thinking': True,
            'sampling': False,
            'min_max_tokens': budget + 1024,
        }
    # A budget-only model with no level chosen, or one with no thinking at all:
    # left at its default, which is not thinking.
    return {'body': {}, 'thinking': False, 'sampling': takes_sampling}
