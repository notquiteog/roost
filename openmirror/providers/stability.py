"""Stability AI — Stable Image, and the video model that started the field.

Two shapes in one adapter, which is unusual here and follows the API: Stable
Image answers **synchronously** with the picture in the response, and
image-to-video is a job. Pretending both were jobs would mean polling for
something that was already in hand.

Three details, each of which produces a confusing failure if guessed:

**Ask for JSON, not for the image.** With `Accept: image/*` the body is the
PNG, which is convenient and throws away the seed — and the seed is the whole
of what makes a generation repeatable. `Accept: application/json` returns
base64 plus the seed that was actually used, which is worth the decode.

**Video needs a picture, not a prompt.** Stable Video Diffusion is
image-to-video only; there is no text path. The form says so rather than
offering a prompt box that would be ignored, and the image control is marked
required so it cannot be folded away behind the disclosure.

**Its input sizes are a closed set.** 1024×576, 576×1024 or 768×768, and
nothing else — a 1920×1080 still is a 400 that names dimensions rather than
explaining them.
"""

from __future__ import annotations

import base64
from typing import Any

import aiohttp

from openmirror.media.params import Param, common_image
from openmirror.providers.hosted_media import HostedMedia, Poll, Submitted

API = 'https://api.stability.ai'

IMAGE_MODELS: tuple[tuple[str, str], ...] = (
    ('ultra', 'Stable Image Ultra — the flagship, and the one with the best prompt adherence.'),
    ('core', 'Stable Image Core — fast and cheap, for working out a composition.'),
    ('sd3', 'Stable Diffusion 3.5 — takes a negative prompt and an init image, which Ultra and Core do not.'),
)

VIDEO_MODELS: tuple[tuple[str, str], ...] = (
    ('image-to-video', 'Stable Video Diffusion — animates a still. There is no text-to-video here.'),
)

#: The only three the video endpoint accepts.
VIDEO_SIZES = ('1024x576', '576x1024', '768x768')


class StabilityProvider(HostedMedia):
    provider_id = 'stability'

    def __init__(self, base_url: str = API, api_key: str = '', **kw: Any) -> None:
        super().__init__(base_url or API, api_key, **kw)

    def headers(self) -> dict[str, str]:
        h = {'Accept': 'application/json'}
        if self.api_key:
            h['Authorization'] = f'Bearer {self.api_key}'
        return h

    # -- video, which is a job ----------------------------------------------

    async def submit(self, prompt: str, *, model: str, **kw: Any) -> Submitted:
        image = kw.get('image')
        if not isinstance(image, bytes | bytearray):
            raise RuntimeError(
                'stability: image-to-video needs a starting image — there is no text-to-video on '
                f'this API. It must be one of {", ".join(VIDEO_SIZES)}.'
            )

        form = aiohttp.FormData()
        form.add_field('image', bytes(image), filename='start.png', content_type='image/png')
        for key in ('cfg_scale', 'motion_bucket_id'):
            if kw.get(key) not in (None, ''):
                form.add_field(key, str(kw[key]))
        if kw.get('seed') not in (None, '', -1, '-1'):
            form.add_field('seed', str(int(kw['seed'])))

        headers = self.headers()
        headers.pop('Content-Type', None)
        async with self.transport.session(120) as session:
            async with session.post(f'{self.base_url}/v2beta/image-to-video', data=form, headers=headers) as resp:
                body = await resp.json() if resp.status < 300 else {}
                if resp.status >= 300:
                    raise RuntimeError(f'stability: HTTP {resp.status}: {(await resp.text())[:400]}')
        return Submitted(id=str(body.get('id', '')),
                         poll_url=f'{self.base_url}/v2beta/image-to-video/result/{body.get("id")}')

    async def check(self, job: Submitted) -> Poll:
        # A 202 means "still rendering" here, so the status code is the state
        # and a non-200 is not automatically a failure.
        async with self.transport.session(60) as session:
            async with session.get(job.poll_url, headers=self.headers()) as resp:
                if resp.status == 202:
                    return Poll(state='running', note='rendering at Stability')
                text = await resp.text()
                if resp.status != 200:
                    return Poll(state='failed', error=f'stability: HTTP {resp.status}: {text[:300]}')
                import json

                body = json.loads(text or '{}')

        if body.get('finish_reason') == 'CONTENT_FILTERED':
            return Poll(state='failed', error='Stability filtered the result. The job ran and was not returned.')
        video = body.get('video')
        if not video:
            return Poll(state='failed', error='stability: the job finished with no video in the response')
        return Poll(state='done', progress=1.0, seed=body.get('seed'),
                    inline=[(base64.b64decode(video), 'video/mp4')])

    # -- images, which are not ----------------------------------------------

    async def generate(self, prompt: str, *, model: str = 'ultra', n: int = 1, size: str = '',
                       progress: Any = None, **kw: Any) -> list[Any]:
        model = (model or 'ultra').strip()
        if model.startswith('image-to-video') or model == 'video':
            return await self.run(prompt, model=model, progress=progress, **kw)

        out: list[Any] = []
        for index in range(max(1, int(n or 1))):
            if progress:
                progress(index / max(1, n), f'generating at Stability ({index + 1} of {n})')
            out.append(await self._one(prompt, model, kw))
        return out

    async def _one(self, prompt: str, model: str, kw: dict[str, Any]) -> Any:
        from openmirror.providers.base import GeneratedMedia

        form = aiohttp.FormData()
        form.add_field('prompt', prompt)
        for key in ('aspect_ratio', 'style_preset', 'output_format', 'negative_prompt', 'strength'):
            value = kw.get(key)
            if value not in (None, '', 'auto', 'none'):
                form.add_field(key, str(value))
        if kw.get('seed') not in (None, '', -1, '-1'):
            form.add_field('seed', str(int(kw['seed'])))
        if model == 'sd3' and kw.get('model'):
            form.add_field('model', str(kw['model']))
        if isinstance(kw.get('image'), bytes | bytearray):
            form.add_field('image', bytes(kw['image']), filename='init.png', content_type='image/png')
            # `mode` is not offered as a control: it is implied by whether an
            # image was given, and a form where the two can disagree is a form
            # where "image-to-image with no image" is a 400 somebody has to
            # work out for themselves.
            form.add_field('mode', 'image-to-image')
            if kw.get('strength') in (None, ''):
                form.add_field('strength', '0.6')

        headers = self.headers()
        headers.pop('Content-Type', None)
        async with self.transport.session(self.timeout) as session:
            async with session.post(
                f'{self.base_url}/v2beta/stable-image/generate/{model}', data=form, headers=headers
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f'stability: HTTP {resp.status}: {(await resp.text())[:400]}')
                body = await resp.json()

        if body.get('finish_reason') == 'CONTENT_FILTERED':
            raise RuntimeError('Stability filtered this result. It was generated and not returned.')
        fmt = kw.get('output_format') or 'png'
        return GeneratedMedia(
            data=base64.b64decode(body['image']),
            media_type=f'image/{"jpeg" if fmt == "jpeg" else fmt}',
            seed=body.get('seed'),
        )

    # -- what it can do -----------------------------------------------------

    async def models(self, kind: str = '') -> list[dict[str, Any]]:
        rows = VIDEO_MODELS if kind == 'video' else IMAGE_MODELS
        return [{'id': i, 'provider': self.provider_id, 'label': i, 'note': n} for i, n in rows]

    async def describe(self, model: str = '') -> list[Param]:
        model = (model or 'ultra').strip()

        if model.startswith('image-to-video') or model == 'video':
            return [
                Param('image', 'Starting image', 'image', group='prompt',
                      help='Required, and it must be exactly 1024×576, 576×1024 or 768×768. '
                           'There is no text-to-video on this API — the picture is the prompt.'),
                Param('seed', 'Seed', 'seed', default=-1, group='sampling', advanced=True),
                Param('cfg_scale', 'Adherence to the still', 'float', default=1.8, minimum=0.0, maximum=10.0,
                      step=0.1, group='sampling',
                      help='How closely the video sticks to the image it started from.'),
                Param('motion_bucket_id', 'Motion', 'int', default=127, minimum=1, maximum=255, step=1,
                      group='motion',
                      help='How much movement. Low is a slow drift; high is a camera in motion, '
                           'and above about 200 it starts to come apart.'),
            ]

        params = common_image()
        if model != 'sd3':
            # Neither Ultra nor Core has negative conditioning. Removed rather
            # than shown and dropped: a field that does nothing is worse than
            # no field at all.
            params = [p for p in params if p.name != 'negative_prompt']
        params += [
            Param('aspect_ratio', 'Aspect ratio', 'enum', default='1:1',
                  options=['21:9', '16:9', '3:2', '5:4', '1:1', '4:5', '2:3', '9:16', '9:21'], group='shape'),
            Param('style_preset', 'Style preset', 'enum', default='none',
                  options=['none', 'photographic', 'cinematic', 'digital-art', 'anime', 'comic-book',
                           'fantasy-art', 'line-art', 'analog-film', 'neon-punk', 'isometric',
                           'low-poly', 'origami', 'modeling-compound', 'pixel-art', '3d-model',
                           'tile-texture', 'enhance'],
                  group='quality', help='A house style applied on top of the prompt.'),
            Param('output_format', 'File format', 'enum', default='png', options=['png', 'jpeg', 'webp'],
                  group='output', advanced=True),
            Param('image', 'Starting image', 'image', group='prompt', advanced=True,
                  help='Supplying one switches this to image-to-image.'),
            Param('strength', 'How much to change it', 'float', default=0.6, minimum=0.0, maximum=1.0,
                  step=0.05, group='sampling', advanced=True,
                  help='Only used with a starting image. 0 returns it unchanged, 1 ignores it.'),
        ]
        return params
