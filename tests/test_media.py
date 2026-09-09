"""Generation: the parameter schema, the workflow tokeniser, and the store.

The through-line is that a picture nobody can make again is a picture nobody
can iterate on. So the recipe is what these check: that settings survive the
round trip to the backend, that what came back is recorded with the seed that
was actually used, and that a template's controls follow from the template.
"""

from __future__ import annotations

import json

import pytest

from roost.media import workflow as wf
from roost.media.params import Param, coerce, common_image, defaults, unknown_keys
from roost.media.store import MediaStore

# -- parameters -------------------------------------------------------------

SCHEMA = [
    Param('steps', 'Steps', 'int', default=25, minimum=1, maximum=150),
    Param('cfg_scale', 'CFG', 'float', default=7.0, minimum=1.0, maximum=30.0),
    Param('hires', 'High-res', 'bool', default=False),
    Param('sampler', 'Sampler', 'enum', default='Euler', options=['Euler', 'DPM++ 2M']),
    Param('prompt', 'Prompt', 'text'),
]


def test_values_are_coerced_to_the_type_the_backend_wants():
    """A string where an integer belongs fails inside the queue rather than at
    the request, which is a much worse place to find out."""
    clean = coerce(SCHEMA, {'steps': '40', 'cfg_scale': '6.5', 'hires': 'true'})
    assert clean == {'steps': 40, 'cfg_scale': 6.5, 'hires': True}
    assert isinstance(clean['steps'], int)


def test_out_of_range_is_clamped_rather_than_refused():
    """Someone dragging a slider to 400 meant the maximum, not an error page."""
    assert coerce(SCHEMA, {'steps': 400})['steps'] == 150
    assert coerce(SCHEMA, {'steps': -5})['steps'] == 1


def test_empty_values_are_dropped_not_sent():
    """An explicit null is a 400 on several backends that accept the request
    without the field at all."""
    assert coerce(SCHEMA, {'steps': None, 'cfg_scale': '', 'prompt': 'x'}) == {'prompt': 'x'}


def test_a_parameter_the_backend_does_not_have_is_reported():
    """Dropped silently, a model believes it set the sampler and reports a
    choice it never made."""
    given = {'steps': 20, 'made_up': 3}
    assert 'made_up' not in coerce(SCHEMA, given)
    assert unknown_keys(SCHEMA, given) == ['made_up']


def test_the_common_form_is_the_same_wherever_you_are():
    """Switching backend should not move the field you were typing in."""
    names = [p.name for p in common_image()]
    assert names[:2] == ['prompt', 'negative_prompt']
    assert 'seed' in names and 'n' in names


def test_defaults_are_only_the_ones_that_have_one():
    """`False` is a default, not the absence of one — the filter is `is not
    None` rather than a truth test, because a switch that defaults to off is a
    real answer and dropping it would send the backend no value at all."""
    assert defaults(SCHEMA) == {'steps': 25, 'cfg_scale': 7.0, 'hires': False, 'sampler': 'Euler'}
    assert 'prompt' not in defaults(SCHEMA), 'no default means no entry'


# -- workflows --------------------------------------------------------------

GRAPH = {
    '3': {'class_type': 'KSampler', 'inputs': {
        'seed': 1, 'steps': 20, 'cfg': 8.0, 'sampler_name': 'euler', 'scheduler': 'normal',
        'denoise': 1, 'model': ['4', 0], 'positive': ['6', 0], 'negative': ['7', 0],
        'latent_image': ['5', 0]}},
    '4': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': 'x.safetensors'}},
    '5': {'class_type': 'EmptyHunyuanLatentVideo',
          'inputs': {'width': 848, 'height': 480, 'length': 49, 'batch_size': 1}},
    '6': {'class_type': 'CLIPTextEncode', 'inputs': {'text': 'a good thing', 'clip': ['4', 1]}},
    '7': {'class_type': 'CLIPTextEncode', 'inputs': {'text': 'a bad thing', 'clip': ['4', 1]}},
    '9': {'class_type': 'SaveAnimatedWEBP', 'inputs': {'fps': 16, 'images': ['8', 0]}},
}


def test_the_positive_prompt_is_told_apart_from_the_negative():
    """Both are the same node class with the same input name. The only thing
    distinguishing them is which sampler input they are wired to — get it
    wrong and every generation is conditioned on what it was meant to avoid.
    """
    template, found = wf.tokenise(GRAPH)
    assert template['6']['inputs']['text'] == '%prompt%'
    assert template['7']['inputs']['text'] == '%negative%'
    assert 'prompt' in found and 'negative' in found


def test_the_form_follows_from_the_template():
    """A video template gets a frame count; an image one does not."""
    template, _ = wf.tokenise(GRAPH)
    names = {p.name for p in wf.describe(template)}
    assert {'prompt', 'seed', 'steps', 'cfg', 'width', 'height', 'frames', 'fps'} <= names

    image_only = {k: v for k, v in template.items() if k != '5'}
    assert 'frames' not in {p.name for p in wf.describe(image_only)}


def test_a_whole_token_keeps_its_type():
    """ComfyUI validates types: a string where an int belongs fails inside the
    queue, which is a long way from the request that caused it."""
    template, _ = wf.tokenise(GRAPH)
    filled = wf.inject(template, {'prompt': 'a hotel', 'steps': 30})
    assert filled['3']['inputs']['steps'] == 30
    assert isinstance(filled['3']['inputs']['steps'], int)
    assert filled['6']['inputs']['text'] == 'a hotel'


def test_an_unfilled_text_token_becomes_empty_not_its_own_name():
    """Otherwise the image is conditioned on the literal word "negative"."""
    filled = wf.inject({'a': '%negative%'}, {'prompt': 'x'})
    assert filled['a'] == ''


def test_a_token_inside_a_sentence_is_substituted():
    filled = wf.inject({'a': '%prompt%, cinematic lighting'}, {'prompt': 'a hotel'})
    assert filled['a'] == 'a hotel, cinematic lighting'


def test_a_wired_input_is_left_alone():
    """`["4", 0]` is a link to another node, not a value. Tokenising it would
    disconnect the graph."""
    template, _ = wf.tokenise(GRAPH)
    assert template['3']['inputs']['model'] == ['4', 0]
    assert template['6']['inputs']['clip'] == ['4', 1]


def test_a_sidecar_can_move_a_range_but_not_add_a_control():
    """A sidecar overriding a label must not become a way to add a parameter
    the graph has nowhere to put."""
    template, _ = wf.tokenise(GRAPH)
    params = {p.name: p for p in wf.describe(template, {'frames': {'maximum': 121},
                                                        'nonexistent': {'default': 1}})}
    assert params['frames'].maximum == 121
    assert 'nonexistent' not in params


# -- the store --------------------------------------------------------------


def test_the_recipe_is_stored_next_to_the_result(tmp_path):
    store = MediaStore(tmp_path)
    media = store.add(
        b'\x89PNG fake', kind='image', media_type='image/png', prompt='a hotel in Tbilisi',
        provider='automatic1111', model='sdxl', seed=4242,
        params={'steps': 40, 'sampler': 'DPM++ 2M'},
    )
    back = store.get(media.id)
    assert back is not None
    assert back.seed == 4242
    assert back.params['steps'] == 40
    assert back.prompt == 'a hotel in Tbilisi'

    sidecar = next(tmp_path.glob(f'*/{media.id}.json'))
    assert json.loads(sidecar.read_text())['model'] == 'sdxl'


def test_two_identical_generations_are_two_files(tmp_path):
    """Disk is cheaper than losing the one you liked."""
    store = MediaStore(tmp_path)
    a = store.add(b'x', kind='image', media_type='image/png', prompt='same')
    b = store.add(b'x', kind='image', media_type='image/png', prompt='same')
    assert a.id != b.id
    assert len(store.list()) == 2


@pytest.mark.parametrize('bad', ['../../etc/passwd', 'a/b', '..', ''])
def test_an_id_is_never_joined_onto_a_path(tmp_path, bad):
    store = MediaStore(tmp_path)
    assert store.path(bad) is None
    assert store.get(bad) is None
