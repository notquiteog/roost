"""OpenRouter, against a fake that answers the way the real one did.

Every shape here was measured against the live API on 2026-09-11 before it was
written down: `/models` serving the full catalogue to a wrong key while `/key`
refuses it, speech arriving as `audio/pcm;rate=24000` from Kokoro and
`rate=44100` from Fish Audio, `/images` answering with base64 and a
`media_type`, and `/videos` accepting a job with a 202 and handing back
`unsigned_urls` on its own host that need the key to fetch.
"""

from __future__ import annotations

import base64
import json
import threading
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openmirror.providers import connections
from openmirror.providers.base import Modality, ProviderRefused
from openmirror.providers.connections import ConnectionStore
from openmirror.providers.openrouter import OpenRouterImages, OpenRouterProvider, OpenRouterVideo
from openmirror.providers.registry import ProviderRegistry

KEY = 'sk-or-v1-' + 'a' * 64
PNG = b'\x89PNG\r\n\x1a\n' + b'\x00' * 16
MP4 = b'\x00\x00\x00\x20ftypisom' + b'\x00' * 16

IMAGE_MODELS = [{
    'id': 'black-forest-labs/flux.2-klein-4b',
    'name': 'FLUX.2 [klein] 4B',
    'description': 'Fast and cheap. More marketing follows.',
    'supported_parameters': {
        'aspect_ratio': {'type': 'enum', 'values': ['auto', '1:1', '16:9']},
        'resolution': {'type': 'enum', 'values': ['2K', '1K', '4K']},
        'n': {'type': 'range', 'min': 1, 'max': 4},
        'seed': {'type': 'boolean'},
        'input_references': {'type': 'range', 'min': 0, 'max': 3},
    },
}]

VIDEO_MODELS = [{
    'id': 'google/veo-3.1-lite',
    'name': 'Google: Veo 3.1 Lite',
    'description': 'A video model.',
    # In the order the real listing gives them, which is not ascending.
    'supported_durations': [8, 4, 6],
    'supported_resolutions': ['1080p', '720p'],
    'supported_aspect_ratios': ['16:9', '9:16'],
    'supported_frame_images': ['first_frame', 'last_frame'],
    'generate_audio': True,
    'seed': True,
}]

LISTINGS = {
    '/api/v1/models': [{'id': 'deepseek/deepseek-v4-flash'}],
    '/api/v1/embeddings/models': [{'id': 'qwen/qwen3-embedding-8b'}],
    '/api/v1/models?output_modalities=transcription': [{'id': 'openai/whisper-large-v3-turbo'}],
    '/api/v1/models?output_modalities=speech': [
        {'id': 'hexgrad/kokoro-82m', 'supported_voices': ['af_heart', 'am_adam']},
    ],
    '/api/v1/images/models': IMAGE_MODELS,
    '/api/v1/videos/models': VIDEO_MODELS,
}


class Fake:
    """What the fake was asked, and what it should say next."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict]] = []
        self.polls = ['pending', 'in_progress', 'completed']
        self.video_error = ''
        self.speech_rate = 24_000
        self.base = ''

    def bodies(self, method: str, path: str) -> list[dict]:
        return [body for m, p, body in self.requests if (m, p) == (method, path)]


def _handler(fake: Fake):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body, content_type: str = 'application/json') -> None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _record(self) -> dict:
            length = int(self.headers.get('Content-Length') or 0)
            raw = self.rfile.read(length) if length else b''
            try:
                body = json.loads(raw) if raw else {}
            except ValueError:
                body = {}
            fake.requests.append((self.command, self.path, body))
            return body

        def _authorised(self) -> bool:
            return self.headers.get('Authorization') == f'Bearer {KEY}'

        def _refuse(self) -> None:
            self._send(401, {'error': {'message': 'No auth credentials found', 'code': 401}})

        def do_GET(self):  # noqa: N802 - the stdlib's name
            self._record()
            if self.path in LISTINGS:
                # Answered whoever asks, as the real listings are.
                return self._send(200, {'data': LISTINGS[self.path]})
            if not self._authorised():
                return self._refuse()
            if self.path == '/api/v1/key':
                return self._send(200, {'data': {'limit': 1}})
            if self.path == '/api/v1/videos/job1':
                status = fake.polls.pop(0) if fake.polls else 'completed'
                body = {'id': 'job1', 'status': status}
                if status == 'completed':
                    body['unsigned_urls'] = [f'{fake.base}/videos/job1/content?index=0']
                if status == 'failed':
                    body['error'] = fake.video_error
                return self._send(200, body)
            if self.path == '/api/v1/videos/job1/content?index=0':
                return self._send(200, MP4, 'video/mp4')
            return self._send(404, {'error': 'not found'})

        def do_POST(self):  # noqa: N802 - the stdlib's name
            self._record()
            if not self._authorised():
                return self._refuse()
            if self.path == '/api/v1/images':
                return self._send(200, {'data': [{'b64_json': base64.b64encode(PNG).decode(),
                                                  'media_type': 'image/png'}]})
            if self.path == '/api/v1/videos':
                # A polling URL on another host, which must not be followed
                # with the key attached.
                return self._send(202, {'id': 'job1', 'status': 'pending',
                                        'polling_url': 'https://elsewhere.invalid/videos/job1'})
            if self.path == '/api/v1/audio/speech':
                # A tenth of a second, at whatever rate this voice is made at.
                pcm = array('h', [i % 1000 for i in range(fake.speech_rate // 10)]).tobytes()
                return self._send(200, pcm, f'audio/pcm;rate={fake.speech_rate};channels=1')
            return self._send(404, {'error': 'not found'})

        def log_message(self, *args):
            return

    return Handler


@pytest.fixture
def fake():
    state = Fake()
    server = ThreadingHTTPServer(('127.0.0.1', 0), _handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.base = f'http://127.0.0.1:{server.server_port}/api/v1'
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _fast_polling(monkeypatch):
    monkeypatch.setattr('openmirror.providers.hosted_media.POLL_START', 0.001)
    monkeypatch.setattr('openmirror.providers.hosted_media.POLL_MAX', 0.001)


# -- one key, six kinds ------------------------------------------------------


def test_one_key_builds_all_six():
    conn = connections.from_host('openrouter', api_key=KEY)
    assert conn.adapter == 'openrouter'
    _, impls = connections.build(conn)

    assert set(impls) == set(Modality)
    for modality in (Modality.CHAT, Modality.EMBEDDING, Modality.STT, Modality.TTS):
        assert isinstance(impls[modality], OpenRouterProvider)
        assert impls[modality].kind is modality
    assert isinstance(impls[Modality.IMAGE], OpenRouterImages)
    assert isinstance(impls[Modality.VIDEO], OpenRouterVideo)
    # One transport under all six, which is what lets one Tor toggle cover them.
    assert len({id(impl.transport) for impl in impls.values()}) == 1


async def test_each_picker_is_asked_for_its_own_list(fake):
    _, impls = connections.build(connections.from_host('openrouter', api_key=KEY, base_url=fake.base))

    async def ids(modality, *kind):
        return [m['id'] for m in await impls[modality].models(*kind)]

    assert await ids(Modality.CHAT) == ['deepseek/deepseek-v4-flash']
    assert await ids(Modality.EMBEDDING) == ['qwen/qwen3-embedding-8b']
    assert await ids(Modality.STT) == ['openai/whisper-large-v3-turbo']
    assert await ids(Modality.TTS) == ['hexgrad/kokoro-82m']
    assert await ids(Modality.IMAGE, 'image') == ['black-forest-labs/flux.2-klein-4b']
    assert await ids(Modality.VIDEO, 'video') == ['google/veo-3.1-lite']
    # Neither media list leaks into the other kind's picker.
    assert await ids(Modality.IMAGE, 'video') == []
    assert await ids(Modality.VIDEO, 'image') == []


# -- the key -----------------------------------------------------------------


async def test_a_wrong_key_is_caught_although_the_model_list_answers_it(fake):
    wrong = OpenRouterProvider(fake.base, 'sk-or-v1-wrong')
    assert await wrong.models(), 'the listing is public, which is the whole problem'
    with pytest.raises(ProviderRefused, match='401'):
        await wrong.check()
    await OpenRouterProvider(fake.base, KEY).check()


def test_the_test_button_fails_a_wrong_key_and_passes_a_right_one(fake, tmp_path, monkeypatch):
    from openmirror.routers import providers as providers_router

    monkeypatch.setattr(providers_router, 'registry', ProviderRegistry())
    monkeypatch.setattr(providers_router, 'store', ConnectionStore(tmp_path / 'connections.json'))
    app = FastAPI()
    app.include_router(providers_router.router)

    with TestClient(app) as client:
        def save(key):
            body = {'host_id': 'openrouter', 'base_url': fake.base, 'api_key': key}
            assert client.put('/api/providers/connections', json=body).status_code == 200
            return client.post('/api/providers/connections/openrouter/test').json()

        refused = save('sk-or-v1-wrong')
        assert refused['ok'] is False
        assert '401' in refused['error']

        accepted = save(KEY)
        assert accepted['ok'] is True
        assert accepted['modalities'] == ['chat', 'embedding', 'image', 'stt', 'tts', 'video']


# -- speech ------------------------------------------------------------------


async def test_speech_at_the_rate_the_call_expects_is_passed_through(fake):
    tts = OpenRouterProvider(fake.base, KEY, kind=Modality.TTS)
    audio = b''.join([c async for c in tts.synthesize('hi', model='hexgrad/kokoro-82m', voice='af_heart')])
    assert len(audio) == 2 * 2400
    assert fake.bodies('POST', '/api/v1/audio/speech')[-1] == {
        'model': 'hexgrad/kokoro-82m', 'input': 'hi', 'response_format': 'pcm', 'voice': 'af_heart',
    }


async def test_speech_made_at_another_rate_is_resampled_rather_than_played_slow(fake):
    fake.speech_rate = 44_100
    tts = OpenRouterProvider(fake.base, KEY, kind=Modality.TTS)
    audio = b''.join([c async for c in tts.synthesize('hi', model='fish-audio/s2.1-pro', voice='')])
    # A tenth of a second in, a tenth of a second out, at the rate the call was told.
    assert len(audio) == 2 * 2400
    # And no voice sent when none was chosen, for the models with a default one.
    assert 'voice' not in fake.bodies('POST', '/api/v1/audio/speech')[-1]


async def test_voices_are_listed_with_the_model_they_belong_to(fake):
    voices = await OpenRouterProvider(fake.base, KEY, kind=Modality.TTS).voices()
    assert {'id': 'af_heart', 'model': 'hexgrad/kokoro-82m'} in voices


# -- images ------------------------------------------------------------------


async def test_a_picture_is_asked_for_by_ratio_and_never_by_a_made_up_size(fake):
    out = await OpenRouterImages(fake.base, KEY).generate(
        'a heron', model='black-forest-labs/flux.2-klein-4b', n=1, size='1024x1024',
        aspect_ratio='16:9', seed=-1, quality='auto', image=[PNG],
    )
    assert [(m.data, m.media_type) for m in out] == [(PNG, 'image/png')]

    body = fake.bodies('POST', '/api/v1/images')[-1]
    assert body['aspect_ratio'] == '16:9'
    # The size the media service synthesised, one image, a random seed and
    # "auto" all mean "not asked for", and several models refuse them.
    for absent in ('size', 'n', 'seed', 'quality'):
        assert absent not in body, absent
    assert body['input_references'][0]['image_url']['url'].startswith('data:image/png;base64,')


async def test_the_image_form_is_what_that_model_takes(fake):
    params = {p.name: p for p in await OpenRouterImages(fake.base, KEY).describe('black-forest-labs/flux.2-klein-4b')}
    assert set(params) == {'prompt', 'aspect_ratio', 'resolution', 'n', 'seed', 'image'}
    # The cheapest tier it offers, not whichever the listing happened to put first.
    assert params['resolution'].default == '1K'
    assert params['aspect_ratio'].default == '1:1'
    assert params['n'].maximum == 4


async def test_an_image_form_is_drawn_even_when_the_listing_cannot_be_reached():
    params = await OpenRouterImages('http://127.0.0.1:9/api/v1', KEY).describe('anything')
    assert params[0].name == 'prompt'


# -- video -------------------------------------------------------------------


async def test_a_video_is_submitted_polled_and_fetched_with_the_key(fake):
    notes: list[str] = []
    out = await OpenRouterVideo(fake.base, KEY).generate(
        'a heron takes off', model='google/veo-3.1-lite', duration='4', resolution='720p',
        aspect_ratio='16:9', generate_audio=False, first_frame=PNG,
        progress=lambda _fraction, note: notes.append(note),
    )
    assert [(m.data, m.media_type) for m in out] == [(MP4, 'video/mp4')]

    submitted = fake.bodies('POST', '/api/v1/videos')[-1]
    assert submitted['duration'] == 4, 'an integer on the wire, though the form offers a choice'
    assert submitted['generate_audio'] is False
    assert submitted['frame_images'][0]['frame_type'] == 'first_frame'
    # Polled on the host it was configured for, not the one the response named.
    assert len(fake.bodies('GET', '/api/v1/videos/job1')) == 3
    assert 'queued at OpenRouter' in notes and 'generating at OpenRouter' in notes


async def test_a_failed_video_says_what_openrouter_said(fake):
    fake.polls = ['failed']
    fake.video_error = 'Content policy violation'
    with pytest.raises(RuntimeError, match='Content policy violation'):
        await OpenRouterVideo(fake.base, KEY).generate('x', model='google/veo-3.1-lite')


async def test_the_video_form_defaults_to_the_cheapest_clip(fake):
    params = {p.name: p for p in await OpenRouterVideo(fake.base, KEY).describe('google/veo-3.1-lite')}
    assert params['duration'].options == ['4', '6', '8']
    assert params['duration'].default == '4'
    assert params['resolution'].default == '720p'
    assert params['generate_audio'].default is False
    assert {'first_frame', 'last_frame', 'seed'} <= set(params)


# -- routes ------------------------------------------------------------------


def test_the_other_four_are_pinned_only_when_named():
    from openmirror.config import Config
    from openmirror.providers.bootstrap import _default_routes

    cfg = Config()
    for name in ('chat_provider', 'embed_provider', 'stt_provider', 'tts_provider', 'image_provider', 'video_provider'):
        setattr(cfg, name, '')
    assert _default_routes(cfg).routes == {}

    cfg.stt_provider = cfg.tts_provider = cfg.image_provider = cfg.video_provider = 'openrouter'
    cfg.default_stt_model = 'openai/whisper-large-v3-turbo'
    cfg.default_tts_model, cfg.default_tts_voice = 'hexgrad/kokoro-82m', 'af_heart'
    cfg.image_model, cfg.video_model = 'black-forest-labs/flux.2-klein-4b', 'google/veo-3.1-lite'

    routes = _default_routes(cfg).routes
    assert routes[Modality.STT].model == 'openai/whisper-large-v3-turbo'
    assert routes[Modality.TTS].model == 'hexgrad/kokoro-82m'
    assert routes[Modality.TTS].options == {'voice': 'af_heart'}
    assert routes[Modality.IMAGE].model == 'black-forest-labs/flux.2-klein-4b'
    assert (routes[Modality.VIDEO].provider, routes[Modality.VIDEO].model) == ('openrouter', 'google/veo-3.1-lite')


def test_naming_the_routed_provider_keeps_the_routed_model():
    """Found in the studio, one click before spending money: its video tab names
    the provider its picker shows, and was then offered the first model
    OpenRouter lists — a video *editor* that needs a source clip — instead of
    the one configured."""
    from openmirror.providers.base import ProviderInfo
    from openmirror.providers.registry import Route, RouteSet

    registry = ProviderRegistry()
    for ident in ('openrouter', 'other'):
        registry.register(ProviderInfo(id=ident, label=ident, modalities={Modality.VIDEO}), {Modality.VIDEO: object()})
    registry.set_defaults(RouteSet(routes={Modality.VIDEO: Route(provider='openrouter', model='google/veo-3.1-lite')}))

    def model_for(provider: str, model: str = '') -> str:
        named = RouteSet(routes={Modality.VIDEO: Route(provider=provider, model=model)})
        return registry.resolve(Modality.VIDEO, named)[1].model

    assert model_for('openrouter') == 'google/veo-3.1-lite'
    # A model named outright still wins, and another provider is not handed this one's.
    assert model_for('openrouter', 'minimax/hailuo-3') == 'minimax/hailuo-3'
    assert model_for('other') == ''
