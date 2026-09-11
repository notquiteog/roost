"""The thinking dial, spelled for each host.

Every table here guards a refusal rather than a degradation: a value a host
does not take is a 400, and a 400 is the whole turn gone. None of these
servers can be run locally with somebody's key, so the request fragments are
asserted directly.
"""

from __future__ import annotations

from openmirror.providers import reasoning as R

OPENAI = 'https://api.openai.com/v1'
OPENROUTER = 'https://openrouter.ai/api/v1'
GROQ = 'https://api.groq.com/openai/v1'
FIREWORKS = 'https://api.fireworks.ai/inference/v1'
TOGETHER = 'https://api.together.xyz/v1'
SILICONFLOW = 'https://api.siliconflow.com/v1'
DASHSCOPE = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
GEMINI = 'https://generativelanguage.googleapis.com/v1beta/openai'
VLLM = 'http://10.0.0.5:8000/v1'


def effort(level, model, url, **kw):
    return R.openai_compat_reasoning(level, model, url, **kw).get('reasoning_effort')


def test_the_host_is_read_off_the_address():
    assert R.dialect_for(OPENAI) == 'openai'
    assert R.dialect_for(OPENROUTER) == 'openrouter'
    assert R.dialect_for(GROQ) == 'groq'
    assert R.dialect_for(FIREWORKS) == 'fireworks'
    assert R.dialect_for(TOGETHER) == 'together'
    assert R.dialect_for('https://api.siliconflow.cn/v1') == 'siliconflow'
    assert R.dialect_for(DASHSCOPE) == 'dashscope'
    assert R.dialect_for('https://ws1.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1') == 'dashscope'
    assert R.dialect_for(GEMINI) == 'gemini'
    assert R.dialect_for(VLLM) == 'generic'
    assert R.dialect_for('not a url') == 'generic'


def test_none_is_no_opinion_and_is_not_off():
    # None leaves every host at its default — what openmirror did before a
    # level existed. `off` overrides it downwards. Collapsing them would switch
    # reasoning off for every session that never chose.
    assert R.openai_compat_reasoning(None, 'gpt-5.6', OPENAI) == {}
    assert effort('off', 'gpt-5.6', OPENAI) == 'none'
    assert R.ollama_think(None, 'qwen3.5:9b') is None
    assert R.ollama_think('off', 'qwen3.5:9b') is False
    # The older boolean spellings still mean what they meant.
    assert R.normalise(False) == 'off'
    assert R.normalise(' HIGH ') == 'high'
    assert R.normalise('exhaustive') is None


def test_a_level_clamps_down_never_up_and_off_is_the_least_allowed():
    assert R.clamp_effort('max', ('low', 'medium', 'high')) == 'high'
    assert R.clamp_effort('xhigh', ('low', 'medium', 'high', 'max')) == 'high'
    assert R.clamp_effort('off', ('none', 'low')) == 'none'
    assert R.clamp_effort('off', ('minimal', 'low')) == 'minimal'
    assert R.clamp_effort('off', ('low', 'medium')) == 'low'
    assert R.clamp_effort('low', ('minimal', 'low', 'medium')) == 'low'
    assert R.clamp_effort('low', ('high',)) == 'high'
    assert R.clamp_effort('high', None) is None


def test_openai_each_model_gets_a_rung_from_its_own_ladder():
    assert effort('off', 'gpt-6-astra', OPENAI) == 'low', 'Astra refuses none'
    assert effort('max', 'gpt-6-astra', OPENAI) == 'max'
    assert effort('off', 'gpt-5.6-luna', OPENAI) == 'none'
    assert effort('max', 'gpt-5.2', OPENAI) == 'xhigh'
    assert effort('xhigh', 'gpt-5.1', OPENAI) == 'high'
    assert effort('off', 'gpt-5', OPENAI) == 'minimal'
    assert effort('off', 'o3', OPENAI) == 'low'


def test_openai_a_model_that_does_not_reason_is_sent_nothing_at_any_level():
    for level in R.LEVELS:
        assert R.openai_compat_reasoning(level, 'gpt-4o', OPENAI) == {}, level
        # Positive control: the reasoning model beside it does get one.
        assert effort(level, 'gpt-5.6', OPENAI), level


def test_openai_sampling_and_the_cap_field():
    for model in ('gpt-5.6', 'o3', 'openai/gpt-5', 'gpt-5-chat-latest'):
        assert R.openai_takes_sampling(model) is False, model
    for model in ('gpt-4o', 'qwen3:8b', 'openai/gpt-oss-120b'):
        assert R.openai_takes_sampling(model) is True, model
    assert R.openai_max_tokens_field(OPENAI) == 'max_completion_tokens'
    assert R.openai_max_tokens_field(VLLM) == 'max_tokens'


def test_openrouter_groq_fireworks_together():
    assert R.openai_compat_reasoning('off', 'anthropic/claude-opus-5', OPENROUTER) == {'reasoning': {'effort': 'none'}}
    assert R.openai_compat_reasoning('max', 'openai/gpt-oss-120b', GROQ) == {'reasoning_effort': 'high'}
    assert R.openai_compat_reasoning('low', 'qwen/qwen3.6-27b', GROQ) == {'reasoning_effort': 'default', 'reasoning_format': 'parsed'}
    assert R.openai_compat_reasoning('medium', 'qwen/qwen3.8-27b', GROQ) == {'reasoning_effort': 'medium', 'reasoning_format': 'parsed'}
    assert R.openai_compat_reasoning('high', 'llama-3.3-70b-versatile', GROQ) == {}
    assert R.openai_compat_reasoning('off', 'accounts/fireworks/models/qwen3-8b', FIREWORKS) == {'reasoning_effort': 'none'}
    assert R.openai_compat_reasoning('off', 'zai-org/GLM-5.2', TOGETHER) == {'reasoning': {'enabled': False}}
    assert R.openai_compat_reasoning('off', 'deepseek-ai/DeepSeek-R1', TOGETHER) == {}


def test_siliconflow_and_alibaba_take_a_switch_and_a_budget():
    assert R.openai_compat_reasoning('off', 'Qwen/Qwen3-32B', SILICONFLOW) == {'enable_thinking': False}
    assert R.openai_compat_reasoning('high', 'Qwen/Qwen3-32B', SILICONFLOW) == {'enable_thinking': True, 'thinking_budget': 16384}
    assert R.openai_compat_reasoning('max', 'Qwen/Qwen3-32B', SILICONFLOW) == {'enable_thinking': True, 'thinking_budget': 32768}
    assert R.openai_compat_reasoning('off', 'qwen3.6-plus', DASHSCOPE) == {'enable_thinking': False}
    assert R.openai_compat_reasoning('max', 'qwen3-max', DASHSCOPE) == {'enable_thinking': True}
    # Refused non-streamed with thinking on — switched off rather than failed.
    assert R.openai_compat_reasoning('high', 'qwen3-32b', DASHSCOPE, stream=False) == {'enable_thinking': False}
    assert R.openai_compat_reasoning('off', 'qwq-plus', DASHSCOPE) == {'thinking_budget': 1024}


def test_gemini_off_is_none_only_where_there_is_an_off_switch():
    assert effort('off', 'gemini-2.5-flash', GEMINI) == 'none'
    assert effort('off', 'gemini-2.5-pro', GEMINI) == 'low'
    assert effort('max', 'gemini-3.1-pro', GEMINI) == 'high'
    assert R.openai_compat_reasoning('high', 'gemini-2.0-flash', GEMINI) == {}


def test_an_unknown_host_hears_about_reasoning_only_when_the_model_reasons():
    assert R.openai_compat_reasoning('high', 'qwen3:8b', VLLM) == {'reasoning_effort': 'high'}
    assert R.openai_compat_reasoning('off', 'qwen3:8b', VLLM) == {}
    assert R.openai_compat_reasoning('high', 'llama3.3:70b', VLLM) == {}


def test_no_host_is_sent_an_effort_above_the_level_asked_for():
    cases = [
        (OPENAI, ('gpt-6-astra', 'gpt-5.6', 'gpt-5.2', 'gpt-5.1', 'gpt-5', 'o3')),
        (GROQ, ('openai/gpt-oss-120b', 'qwen/qwen3.8-27b')),
        (FIREWORKS, ('accounts/fireworks/models/qwen3-8b',)),
        (GEMINI, ('gemini-2.5-flash', 'gemini-3.1-pro')),
        (VLLM, ('qwen3:8b',)),
    ]
    checked = 0
    for url, models in cases:
        for model in models:
            for level in R.LEVELS[1:]:
                got = effort(level, model, url)
                if got in (None, 'default'):
                    continue
                assert R.ORDER.index(got) <= R.ORDER.index(level), (url, model, level, got)
                checked += 1
    assert checked > 40, f'only {checked} combinations were checked — the tables stopped answering'


def test_ollama_three_names_and_false_except_gpt_oss():
    assert R.ollama_think('off', 'gpt-oss:20b') == 'low', 'gpt-oss ignores false'
    assert R.ollama_think('medium', 'qwen3.5:9b') == 'medium'
    assert R.ollama_think('max', 'qwen3.5:9b') == 'high'


def test_anthropic_each_family_is_asked_in_the_one_form_it_accepts():
    opus = R.anthropic_reasoning('claude-opus-5', 'xhigh')
    assert opus['body'] == {'thinking': {'type': 'adaptive', 'display': 'summarized'}, 'output_config': {'effort': 'xhigh'}}
    assert opus['sampling'] is False
    assert R.anthropic_reasoning('claude-sonnet-4-6', 'xhigh')['body']['output_config'] == {'effort': 'high'}, '4.6 predates xhigh'
    haiku = R.anthropic_reasoning('claude-haiku-4-5', 'high', max_tokens=64000)
    assert haiku['body']['thinking']['type'] == 'enabled', 'Haiku 4.5 rejects adaptive'
    assert 'output_config' not in haiku['body'], 'and rejects effort'
    assert haiku['min_max_tokens'] > haiku['body']['thinking']['budget_tokens']


def test_anthropic_off_never_disables_a_model_that_thinks_adaptively():
    # This is an agent harness: a disabled thinking type is how a tool call
    # ends up written into visible text and never run. Checked over every
    # family, with a positive control that the adaptive ones DO get a body.
    for model in ('claude-opus-5', 'claude-opus-4-8', 'claude-sonnet-5', 'claude-sonnet-4-6', 'claude-fable-5-1', 'claude-something-9'):
        out = R.anthropic_reasoning(model, 'off')
        assert 'disabled' not in repr(out['body']), model
        assert out['body'], f'{model} got nothing at all for off'
    assert R.anthropic_reasoning('claude-opus-5', 'off')['body'] == {
        'thinking': {'type': 'adaptive', 'display': 'omitted'}, 'output_config': {'effort': 'low'}}
    assert 'thinking' not in R.anthropic_reasoning('claude-fable-5-1', 'off')['body'], 'Fable refuses any off switch'


def test_anthropic_no_level_leaves_the_model_at_its_default():
    assert R.anthropic_reasoning('claude-opus-5', None)['body'] == {'thinking': {'type': 'adaptive', 'display': 'summarized'}}
    # A budget-only model does not think by default, and keeps its sampling.
    old = R.anthropic_reasoning('claude-haiku-4-5', None, takes_sampling=True)
    assert old['body'] == {} and old['sampling'] is True


def test_anthropic_the_models_api_overrides_the_id():
    caps = {
        'thinking': {'types': {'adaptive': {'supported': True}, 'enabled': {'supported': False}}},
        'effort': {'supported': True, 'low': {'supported': True}, 'medium': {'supported': True},
                   'high': {'supported': True}, 'xhigh': {'supported': False}, 'max': {'supported': True}},
    }
    assert R.anthropic_reasoning('claude-something-9', 'xhigh', capabilities=caps)['body']['output_config'] == {'effort': 'high'}
    for level in ('low', 'medium', 'high', 'xhigh', 'max'):
        for ceiling in (8192, 64000, 128000):
            b = R.anthropic_budget(level, ceiling)
            assert 1024 <= b <= ceiling // 2, (level, ceiling, b)
