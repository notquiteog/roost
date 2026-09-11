"""The parts of image and video generation that are worth pinning.

Nothing here talks to a provider. What it tests is the three things that go
wrong without a network: the translation from a model's published schema into
a form, the one polling loop that every hosted service shares, and the rules
about what a generation is allowed to be told.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json

import pytest

from openmirror.media.params import coerce
from openmirror.providers.hosted_media import HostedMedia, Poll, Submitted, sniff, urls_in
from openmirror.providers.openapi_form import params_from_openapi

# ---------------------------------------------------------------------------
# A model's schema becomes a form
# ---------------------------------------------------------------------------

SCHEMA = {
    'components': {
        'schemas': {
            'aspect_ratio': {'type': 'string', 'enum': ['1:1', '16:9', '9:16'], 'title': 'aspect_ratio'},
            'Input': {
                'required': ['prompt'],
                'properties': {
                    'prompt': {'type': 'string', 'x-order': 0, 'title': 'Prompt',
                               'description': 'What to make.'},
                    'aspect_ratio': {'allOf': [{'$ref': '#/components/schemas/aspect_ratio'}],
                                     'default': '16:9', 'x-order': 2,
                                     'description': 'The shape of it.'},
                    'image': {'type': 'string', 'format': 'uri', 'x-order': 1,
                              'description': 'A picture to work from.'},
                    'steps': {'type': 'integer', 'minimum': 1, 'maximum': 50, 'default': 28,
                              'x-order': 3},
                    'guidance': {'type': 'number', 'minimum': 0, 'maximum': 1, 'default': 0.5,
                                 'x-order': 4},
                    'go_fast': {'type': 'boolean', 'default': True, 'x-order': 5},
                    'seed': {'type': 'integer', 'x-order': 6},
                    'lora_weights': {'type': 'object', 'x-order': 7},
                },
            },
        }
    }
}


def _by_name(params):
    return {p.name: p for p in params}


def test_an_enum_behind_a_reference_is_still_an_enum():
    """These schemas spell an enum as `allOf: [$ref]`. A reader that only looks
    at `type` sees a string and renders a text box — and a text box for an
    aspect ratio is a field where 16:9 and 16x9 look equally plausible."""
    param = _by_name(params_from_openapi(SCHEMA))['aspect_ratio']
    assert param.kind == 'enum'
    assert param.options == ['1:1', '16:9', '9:16']
    assert param.default == '16:9'
    # The field's own description wins over the referenced component's.
    assert param.help == 'The shape of it.'


def test_a_file_input_becomes_a_picture_control():
    """`format: uri` is how an image-to-image model asks for an image. As a
    text box it looks broken rather than unsupplied."""
    assert _by_name(params_from_openapi(SCHEMA))['image'].kind == 'image'


def test_a_type_nobody_understands_is_left_out_rather_than_guessed():
    """A control that sends the wrong shape is a 422 the person cannot connect
    to anything they touched. A missing one is merely missing."""
    assert 'lora_weights' not in _by_name(params_from_openapi(SCHEMA))


def test_a_required_field_is_never_folded_away():
    """`advanced` hides a control behind a disclosure, and a form whose one
    mandatory field is hidden fails on submit for no visible reason."""
    assert _by_name(params_from_openapi(SCHEMA))['prompt'].advanced is False


def test_the_authors_own_ordering_is_kept():
    names = [p.name for p in params_from_openapi(SCHEMA)]
    assert names[:4] == ['prompt', 'image', 'aspect_ratio', 'steps']


def test_a_seed_is_a_seed_and_not_just_an_integer():
    assert _by_name(params_from_openapi(SCHEMA))['seed'].kind == 'seed'


def test_a_narrow_float_gets_a_finer_step_than_a_wide_one():
    """A slider from 0 to 1 in tenths has eleven positions, which is not a
    guidance control."""
    params = _by_name(params_from_openapi(SCHEMA))
    assert params['guidance'].step == 0.01
    assert params['steps'].step == 1


def test_fal_names_its_input_schema_after_the_endpoint():
    """Replicate calls it `Input`; fal references it from the POST body under
    whatever name it likes. Both have to be found."""
    doc = {
        'paths': {'/': {'post': {'requestBody': {'content': {'application/json': {
            'schema': {'$ref': '#/components/schemas/FluxDevInput'}}}}}}},
        'components': {'schemas': {'FluxDevInput': {
            'properties': {'prompt': {'type': 'string'}, 'num_images': {'type': 'integer', 'default': 1}}}}},
    }
    assert {p.name for p in params_from_openapi(doc)} == {'prompt', 'num_images'}


def test_a_schema_that_makes_no_sense_produces_no_controls_rather_than_an_error():
    assert params_from_openapi({}) == []
    assert params_from_openapi({'components': {'schemas': {'Input': {}}}}) == []


# ---------------------------------------------------------------------------
# Reading a response that could be shaped any of six ways
# ---------------------------------------------------------------------------


def test_an_output_is_found_wherever_the_service_decided_to_put_it():
    assert urls_in('https://x/a.png') == ['https://x/a.png']
    assert urls_in(['https://x/a.png', 'https://x/b.png']) == ['https://x/a.png', 'https://x/b.png']
    assert urls_in({'video': {'url': 'https://x/v.mp4'}}) == ['https://x/v.mp4']
    assert urls_in({'images': [{'url': 'https://x/a.png'}]}) == ['https://x/a.png']
    assert urls_in({'result': {'sample': 'https://x/s.jpg'}}) == ['https://x/s.jpg']


def test_a_thumbnail_is_never_mistaken_for_the_result():
    """Several of these return a preview alongside the output, and a gallery
    showing the 128px version as the result is a gallery that looks broken."""
    found = urls_in({'url': 'https://x/full.mp4', 'thumbnail_url': 'https://x/thumb.jpg'})
    assert found == ['https://x/full.mp4']


def test_a_non_url_string_is_not_an_output():
    assert urls_in({'status': 'succeeded', 'id': 'abc123'}) == []


def test_what_a_file_is_comes_from_its_bytes_not_its_header():
    """Several of these services label everything application/octet-stream,
    and a PNG stored as .bin is a PNG no gallery will show."""
    assert sniff(b'\x89PNG\r\n\x1a\n' + b'0' * 20, 'application/octet-stream') == 'image/png'
    assert sniff(b'\xff\xd8\xff\xe0' + b'0' * 20) == 'image/jpeg'
    assert sniff(b'\x00\x00\x00\x20ftypisom' + b'0' * 20) == 'video/mp4'
    assert sniff(b'RIFF\x00\x00\x00\x00WEBP') == 'image/webp'
    # Nothing recognised: the header, then the extension.
    assert sniff(b'nonsense', 'image/gif') == 'image/gif'
    assert sniff(b'nonsense', '', 'https://x/y.mp4?sig=1') == 'video/mp4'


# ---------------------------------------------------------------------------
# The loop every hosted provider shares
# ---------------------------------------------------------------------------


class Fake(HostedMedia):
    """A hosted service that answers from a script rather than from a socket."""

    provider_id = 'fake'

    def __init__(self, script, **kw):
        super().__init__('https://example.invalid', 'k', timeout=5, **kw)
        self.script = list(script)
        self.cancelled = False
        self.submitted = None
        self.downloads = []

    async def submit(self, prompt, *, model, **kw):
        self.submitted = {'prompt': prompt, 'model': model, **kw}
        return Submitted(id='job-1', poll_url='https://example.invalid/j/1')

    async def check(self, job):
        return self.script.pop(0) if self.script else Poll(state='running')

    async def cancel(self, job):
        self.cancelled = True

    async def download(self, url):
        self.downloads.append(url)
        return b'\x89PNG\r\n\x1a\n', 'image/png'


@pytest.fixture(autouse=True)
def _fast_polling(monkeypatch):
    """The loop's own waits, shortened. Without this every test here pays a
    real second per poll for a delay that is not what is under test."""
    monkeypatch.setattr('openmirror.providers.hosted_media.POLL_START', 0.001)
    monkeypatch.setattr('openmirror.providers.hosted_media.POLL_MAX', 0.001)


async def test_it_polls_until_it_stops_moving_and_then_downloads():
    fake = Fake([
        Poll(state='queued', note='waiting'),
        Poll(state='running', progress=0.5),
        Poll(state='done', outputs=['https://cdn/x.png']),
    ])
    out = await fake.run('a cat', model='m')
    assert len(out) == 1
    assert out[0].media_type == 'image/png'
    assert fake.downloads == ['https://cdn/x.png']


async def test_progress_is_reported_on_every_poll_including_when_it_has_not_moved():
    """"Third in the queue" becoming "second" is the useful signal on a busy
    service, and a caller watching only the number would miss it."""
    seen = []
    fake = Fake([
        Poll(state='queued', note='third in the queue'),
        Poll(state='queued', note='second in the queue'),
        Poll(state='done', outputs=['https://cdn/x.png']),
    ])
    await fake.run('a cat', model='m', progress=lambda f, n: seen.append((f, n)))
    notes = [n for _, n in seen]
    # Accepted, then each poll, then the download. The two queue positions are
    # both reported even though no fraction ever moved.
    assert notes[0] == 'queued'
    assert notes[1:3] == ['third in the queue', 'second in the queue']
    assert notes[-1] == 'downloading'
    # And no fraction was invented for a job that had not started.
    assert [f for f, _ in seen][:3] == [None, None, None]


async def test_a_failure_carries_the_providers_own_words():
    """"Generation failed" is not an error message. `insufficient credits` and
    `NSFW content detected` want different reactions."""
    fake = Fake([Poll(state='failed', error='insufficient credits')])
    with pytest.raises(RuntimeError, match='insufficient credits'):
        await fake.run('a cat', model='m')
    assert fake.cancelled is False


async def test_cancelling_reaches_the_other_end():
    """A job cancelled here is still running there — and on a hosted service
    still billing there — until the provider is told."""
    fake = Fake([Poll(state='running')] * 50)
    task = asyncio.create_task(fake.run('a cat', model='m'))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake.cancelled is True


async def test_giving_up_also_cancels_rather_than_leaving_it_running():
    fake = Fake([Poll(state='running')] * 10_000)
    fake.timeout = 0
    with pytest.raises(TimeoutError, match='job-1'):
        await fake.run('a cat', model='m')
    assert fake.cancelled is True


async def test_bytes_already_in_hand_are_not_re_downloaded():
    """Stability returns the video inside the status body. Turning that into a
    URL to fetch would be a round trip in the direction nobody wants."""
    fake = Fake([Poll(state='done', inline=[(b'\x00\x00\x00\x20ftypisom', 'video/mp4')])])
    out = await fake.run('a cat', model='m')
    assert fake.downloads == []
    assert out[0].media_type == 'video/mp4'


# ---------------------------------------------------------------------------
# Kling signs rather than authenticates
# ---------------------------------------------------------------------------


def test_the_kling_token_is_a_real_signed_jwt():
    """Three lines of HMAC rather than a dependency, which is only a good trade
    if the three lines are right."""
    from openmirror.providers.kling import mint

    token = mint('my-access-key', 'my-secret')
    header_b64, payload_b64, signature_b64 = token.split('.')

    def unpad(chunk):
        return base64.urlsafe_b64decode(chunk + '=' * (-len(chunk) % 4))

    assert json.loads(unpad(header_b64)) == {'alg': 'HS256', 'typ': 'JWT'}
    payload = json.loads(unpad(payload_b64))
    assert payload['iss'] == 'my-access-key'
    # Valid slightly in the past: the clock here and the clock at Kling are not
    # the same clock, and a token that is not yet valid reads as a bad secret.
    assert payload['nbf'] < payload['exp']

    expected = hmac.new(b'my-secret', f'{header_b64}.{payload_b64}'.encode(), hashlib.sha256).digest()
    assert unpad(signature_b64) == expected


def test_kling_says_what_its_credential_is_rather_than_failing_at_the_first_call():
    from openmirror.providers.kling import KlingProvider

    provider = KlingProvider(api_key='just-one-value')
    with pytest.raises(RuntimeError, match='access_key:secret_key'):
        provider.headers()


def test_kling_picks_its_endpoint_from_how_it_was_built_not_from_the_model_id():
    """`kling-v1-5` is both an image model and a video model, so the id cannot
    decide. Two instances, one per kind — see connections.build."""
    from openmirror.providers.kling import KlingProvider

    image = KlingProvider(api_key='a:b', kind='image')
    video = KlingProvider(api_key='a:b', kind='video')
    assert image.kind != video.kind


# ---------------------------------------------------------------------------
# What a generation may be told
# ---------------------------------------------------------------------------


def test_a_picture_parameter_survives_coercion_as_the_id_it_is():
    """An image control carries a media id. Coercion must not turn it into a
    number, drop it, or try to read it."""
    from openmirror.media.params import Param

    schema = [Param('image', 'Image', 'image')]
    assert coerce(schema, {'image': 'abc123def456'}) == {'image': 'abc123def456'}


async def test_every_media_provider_offers_a_form_without_being_asked_anything():
    """`describe()` is what the panel is built from, so a provider that cannot
    answer it offline is a provider whose panel is empty until a key is typed
    in — which is the wrong order to find out the backend is wrong."""
    from openmirror.providers.bfl import BFLProvider
    from openmirror.providers.ideogram import IdeogramProvider
    from openmirror.providers.kling import KlingProvider
    from openmirror.providers.luma import LumaProvider
    from openmirror.providers.minimax import MiniMaxProvider
    from openmirror.providers.runway import RunwayProvider
    from openmirror.providers.sora import SoraProvider
    from openmirror.providers.stability import StabilityProvider

    providers = [
        BFLProvider(api_key='k'), IdeogramProvider(api_key='k'), KlingProvider(api_key='a:b'),
        LumaProvider(api_key='k'), MiniMaxProvider(api_key='k'), RunwayProvider(api_key='k'),
        SoraProvider(api_key='k'), StabilityProvider(api_key='k'),
    ]
    for provider in providers:
        params = await provider.describe('')
        assert params, provider.provider_id
        # Every form has somewhere to say what to make, whether that is words
        # or a picture.
        kinds = {p.name for p in params}
        assert 'prompt' in kinds or 'image' in kinds, provider.provider_id
        for param in params:
            assert param.label, f'{provider.provider_id}.{param.name}'
            assert param.group, f'{provider.provider_id}.{param.name}'
