"""Generation: the parameter schema, the workflow tokeniser, and the store.

The through-line is that a picture nobody can make again is a picture nobody
can iterate on. So the recipe is what these check: that settings survive the
round trip to the backend, that what came back is recorded with the seed that
was actually used, and that a template's controls follow from the template.
"""

from __future__ import annotations

import json

import pytest

from openmirror.media import workflow as wf
from openmirror.media.params import Param, coerce, common_image, defaults, unknown_keys
from openmirror.media.store import MediaStore

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


def test_a_reference_image_does_not_bury_the_gallery(tmp_path):
    """The store holds uploads too, because an image-to-video reference has to
    live somewhere this server can read later. What it must not do is put every
    picture you dropped in alongside every picture that was made."""
    store = MediaStore(tmp_path)
    store.add(b'made', kind='image', media_type='image/png', prompt='a result')
    store.add(b'given', kind='image', media_type='image/png', source='upload')

    assert [m.prompt for m in store.list()] == ['a result']
    assert len(store.list(source='')) == 2
    assert len(store.list(source='upload')) == 1


def test_the_library_can_be_paged_and_searched(tmp_path):
    store = MediaStore(tmp_path)
    for index in range(5):
        store.add(b'x', kind='image', media_type='image/png',
                  prompt=f'a cat number {index}', model='sdxl')
    store.add(b'x', kind='video', media_type='video/mp4', prompt='a dog running', model='wan')

    assert len(store.list(limit=2)) == 2
    assert len(store.list(limit=2, offset=4)) == 2
    assert len(store.list(limit=10, offset=10)) == 0
    assert {m.prompt for m in store.list(search='dog')} == {'a dog running'}
    # The search covers what a person would actually remember about a result.
    assert len(store.list(search='sdxl')) == 5
    assert len(store.list(kind='video')) == 1


# -- the service ------------------------------------------------------------


class _Recorder:
    """A provider that records what it was told and returns one small PNG."""

    provider_id = 'recorder'

    def __init__(self, *, stall: bool = False) -> None:
        self.seen: dict = {}
        self.stall = stall
        self.interrupted = False
        self.notes: list = []

    async def models(self, kind: str = ''):
        return [{'id': f'{kind}-model'}]

    async def describe(self, model: str = ''):
        return [
            Param('prompt', 'Prompt', 'text'),
            Param('image', 'Starting image', 'image'),
            Param('steps', 'Steps', 'int', default=20, minimum=1, maximum=50),
        ]

    async def generate(self, prompt, *, model='', n=1, size='', progress=None, **kw):
        from openmirror.providers.base import GeneratedMedia

        self.seen = {'prompt': prompt, 'model': model, 'n': n, 'size': size, **kw}
        if progress:
            progress(None, 'queued somewhere')
            progress(0.5, 'halfway')
        if self.stall:
            import asyncio

            await asyncio.sleep(30)
        return [GeneratedMedia(data=b'\x89PNG fake', media_type='image/png', seed=99)]

    async def interrupt(self):
        self.interrupted = True


def _service(tmp_path, impl, modality=None):
    from openmirror.media.service import MediaService
    from openmirror.providers.base import Modality, ProviderInfo
    from openmirror.providers.registry import ProviderRegistry

    modality = modality or Modality.IMAGE
    registry = ProviderRegistry()
    registry.register(
        ProviderInfo(id='recorder', label='Recorder', modalities={modality}, local=True),
        {modality: impl},
    )
    return MediaService(MediaStore(tmp_path), registry)


async def test_a_reference_image_reaches_the_provider_as_bytes(tmp_path):
    """A control of kind `image` carries a media id through the JSON, and the
    provider gets the picture. Ids rather than paths or data URIs is the point:
    the only thing a client can name is something this server already stored."""
    impl = _Recorder()
    service = _service(tmp_path, impl)
    reference = service.store.add(b'\x89PNG reference', kind='image', media_type='image/png',
                                  source='upload')

    job = service.start('image', 'a cat', params={'image': reference.id, 'steps': 30})
    await job.task

    assert job.state == 'done'
    assert impl.seen['image'] == b'\x89PNG reference'
    assert impl.seen['steps'] == 30


async def test_an_id_for_something_that_is_not_there_is_dropped_rather_than_sent(tmp_path):
    impl = _Recorder()
    service = _service(tmp_path, impl)
    job = service.start('image', 'a cat', params={'image': 'deadbeefdeadbeef'})
    await job.task
    assert job.state == 'done'
    assert 'image' not in impl.seen


async def test_the_recipe_records_the_settings_and_not_the_picture(tmp_path):
    """A reference image is megabytes and belongs in the store, not repeated
    inside the sidecar of everything it was used to make."""
    impl = _Recorder()
    service = _service(tmp_path, impl)
    reference = service.store.add(b'\x89PNG reference', kind='image', media_type='image/png',
                                  source='upload')
    job = service.start('image', 'a cat', params={'image': reference.id, 'steps': 30})
    await job.task

    stored = service.store.get(job.media[0].id)
    assert stored.params['steps'] == 30
    assert 'image' not in stored.params
    # And the seed the backend actually used, which is the whole point of
    # storing any of it.
    assert stored.seed == 99


async def test_a_job_carries_what_the_backend_honestly_said(tmp_path):
    impl = _Recorder()
    service = _service(tmp_path, impl)
    job = service.start('image', 'a cat')
    await job.task
    assert job.state == 'done'
    assert job.progress == 1.0
    assert job.elapsed >= 0


async def test_stopping_a_job_tells_the_backend_too(tmp_path):
    """A job abandoned rather than cancelled keeps a GPU busy and keeps a
    hosted account billing."""
    import asyncio

    impl = _Recorder(stall=True)
    service = _service(tmp_path, impl)
    job = service.start('image', 'a cat')
    await asyncio.sleep(0.05)

    assert service.cancel(job.id) is True
    with pytest.raises(asyncio.CancelledError):
        await job.task
    assert job.state == 'cancelled'
    assert impl.interrupted is True


async def test_a_failed_job_is_a_state_rather_than_a_crash(tmp_path):
    class Broken(_Recorder):
        async def generate(self, prompt, **kw):
            raise RuntimeError('CUDA out of memory')

    service = _service(tmp_path, Broken())
    job = service.start('image', 'a cat')
    await job.task
    assert job.state == 'failed'
    # The backend's own words: "out of memory" and "no such checkpoint" want
    # different reactions from the person reading it.
    assert 'CUDA out of memory' in job.error


async def test_the_video_picker_is_not_offered_the_image_models(tmp_path):
    """The aggregators host both and their lists barely overlap, so `kind` is
    passed to any provider whose listing takes it."""
    from openmirror.providers.base import Modality

    impl = _Recorder()
    service = _service(tmp_path, impl, modality=Modality.VIDEO)
    described = await service.describe('video')
    assert described['models'] == [{'id': 'video-model'}]


async def test_finished_jobs_do_not_accumulate_forever(tmp_path):
    from openmirror.media import service as service_mod

    impl = _Recorder()
    service = _service(tmp_path, impl)
    original = service_mod.KEEP_JOBS
    service_mod.KEEP_JOBS = 3
    try:
        for _ in range(6):
            job = service.start('image', 'a cat')
            await job.task
        assert len(service.jobs) <= 4
    finally:
        service_mod.KEEP_JOBS = original


async def test_a_recipe_records_only_what_could_reproduce_the_result(tmp_path):
    """Found by running it: a model that sizes by aspect ratio was recording
    `size 1024x1024`, which it neither asked for nor has a control for, and
    `seed -1`, which is a request for a random seed rather than a seed. Both
    read as settings to somebody trying to get the same picture again."""
    class Ratio(_Recorder):
        async def describe(self, model: str = ''):
            return [
                Param('prompt', 'Prompt', 'text'),
                Param('aspect_ratio', 'Aspect ratio', 'enum', default='16:9', options=['16:9', '1:1']),
                Param('seed', 'Seed', 'seed', default=-1),
                Param('n', 'How many', 'int', default=1, minimum=1, maximum=4),
            ]

    service = _service(tmp_path, Ratio())
    job = service.start('image', 'a lighthouse', params={'aspect_ratio': '16:9', 'seed': -1, 'n': 1})
    await job.task

    recipe = service.store.get(job.media[0].id).params
    assert recipe == {'aspect_ratio': '16:9'}
    assert 'size' not in recipe
    assert 'seed' not in recipe


async def test_a_backend_that_does_size_by_pixels_still_records_it(tmp_path):
    class Pixels(_Recorder):
        async def describe(self, model: str = ''):
            return [
                Param('prompt', 'Prompt', 'text'),
                Param('width', 'Width', 'int', default=1024, minimum=64, maximum=2048),
                Param('height', 'Height', 'int', default=1024, minimum=64, maximum=2048),
                Param('n', 'How many', 'int', default=1, minimum=1, maximum=4),
            ]

    service = _service(tmp_path, Pixels())
    job = service.start('image', 'a lighthouse', params={'width': 1344, 'height': 768, 'n': 2})
    await job.task

    recipe = service.store.get(job.media[0].id).params
    assert recipe['size'] == '1344x768'
    assert recipe['n'] == 2
