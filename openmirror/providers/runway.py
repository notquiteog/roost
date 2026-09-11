"""Runway — Gen-4, and the reference-image mechanism nobody else has.

Runway earns a direct adapter for one feature: **tagged reference images.**
You hand it up to three pictures, give each a name, and then write
`@sarah standing in @warehouse` — and it keeps that person and that place
across every shot. Every other service in this build takes a reference image
as an unnamed influence, which is a different and much weaker thing. Reaching
Runway through an aggregator loses the tags, so it is worth the file.

It is also one of the few that reports a real `progress` fraction, so the bar
for a Runway job is the model's own number rather than elapsed time.

Two shapes, chosen by whether a starting image was given: `text_to_video` and
`image_to_video` are separate endpoints, and Gen-4 Turbo is image-to-video
only — so a prompt with no picture on that model is a 400 about a missing
field rather than an explanation. The form says which is which.
"""

from __future__ import annotations

import logging
from typing import Any

from openmirror.media.params import Param, common_image
from openmirror.providers.hosted_media import HostedMedia, Poll, Submitted, data_uri, urls_in

log = logging.getLogger(__name__)

API = 'https://api.dev.runwayml.com/v1'

#: Pinned rather than latest. Runway dates its API and an undated request is
#: refused; a floating version would mean a breaking change arriving on a day
#: nobody deployed anything.
VERSION = '2024-11-06'

IMAGE_MODELS: tuple[tuple[str, str], ...] = (
    ('gen4_image', 'Gen-4 Image — takes up to three tagged reference images and keeps them consistent.'),
    ('gen4_image_turbo', 'The same, faster and cheaper. Needs at least one reference image.'),
)

VIDEO_MODELS: tuple[tuple[str, str], ...] = (
    ('gen4_turbo', 'Gen-4 Turbo — image-to-video, and the one to reach for by default.'),
    ('gen4_aleph', 'Gen-4 Aleph — edits a video you already have rather than making one.'),
    ('gen3a_turbo', 'Gen-3 Alpha Turbo — the previous generation, cheaper.'),
    ('veo3', 'Veo 3 through Runway — makes its own audio.'),
)

IMAGE_RATIOS = ('1920:1080', '1080:1920', '1024:1024', '1360:768', '1080:1080',
                '1168:880', '1440:1080', '1080:1440', '1808:768', '2112:912')
VIDEO_RATIOS = ('1280:720', '720:1280', '1104:832', '832:1104', '960:960', '1584:672', '1280:768', '768:1280')


class RunwayProvider(HostedMedia):
    provider_id = 'runway'

    def __init__(self, base_url: str = API, api_key: str = '', **kw: Any) -> None:
        super().__init__(base_url or API, api_key, **kw)

    def headers(self) -> dict[str, str]:
        h = {'Content-Type': 'application/json', 'X-Runway-Version': VERSION}
        if self.api_key:
            h['Authorization'] = f'Bearer {self.api_key}'
        return h

    async def submit(self, prompt: str, *, model: str, **kw: Any) -> Submitted:
        model = (model or 'gen4_turbo').strip()
        image = kw.get('promptImage') or kw.get('image')
        is_image_model = model.startswith('gen4_image')

        payload: dict[str, Any] = {'model': model, 'promptText': prompt}
        if kw.get('ratio'):
            payload['ratio'] = kw['ratio']
        if kw.get('seed') not in (None, '', -1, '-1'):
            payload['seed'] = int(kw['seed'])

        if is_image_model:
            path = '/text_to_image'
            tags = self._references(kw)
            if tags:
                payload['referenceImages'] = tags
        elif isinstance(image, bytes | bytearray):
            path = '/image_to_video'
            payload['promptImage'] = data_uri(bytes(image))
            for key in ('duration', 'ratio'):
                if kw.get(key) not in (None, ''):
                    payload[key] = kw[key]
        else:
            path = '/text_to_video'
            if kw.get('duration') not in (None, ''):
                payload['duration'] = kw['duration']

        if payload.get('duration') is not None:
            payload['duration'] = int(payload['duration'])

        body = await self.post_json(f'{self.base_url}{path}', payload)
        task = str(body.get('id', ''))
        return Submitted(id=task, poll_url=f'{self.base_url}/tasks/{task}')

    @staticmethod
    def _references(kw: dict[str, Any]) -> list[dict[str, str]]:
        """Up to three pictures, each with a name you can write into the prompt.

        The tag is the whole point and it is easy to lose: without one the
        image is a vague influence, and with one it is `@sarah`, referable by
        name in this and every later prompt. Tags are taken from the
        `reference_tags` field when it is filled in and named `ref1`, `ref2`,
        `ref3` when it is not, because an unnamed reference is still better
        than a dropped one.
        """
        blobs = kw.get('referenceImages') or kw.get('reference_images') or []
        if isinstance(blobs, bytes | bytearray):
            blobs = [blobs]
        names = [n.strip() for n in str(kw.get('reference_tags') or '').split(',') if n.strip()]

        out: list[dict[str, str]] = []
        for index, blob in enumerate(blobs[:3]):
            if not isinstance(blob, bytes | bytearray):
                continue
            tag = names[index] if index < len(names) else f'ref{index + 1}'
            # Runway requires the tag to be alphanumeric and to start with a
            # letter, and rejects the whole request over a stray space.
            tag = ''.join(c for c in tag if c.isalnum()) or f'ref{index + 1}'
            out.append({'uri': data_uri(bytes(blob)), 'tag': tag})
        return out

    async def check(self, job: Submitted) -> Poll:
        body = await self.get_json(job.poll_url)
        status = str(body.get('status', '')).upper()

        if status == 'SUCCEEDED':
            return Poll(state='done', progress=1.0, outputs=urls_in(body.get('output')))
        if status in ('FAILED', 'CANCELLED'):
            reason = body.get('failure') or body.get('failureCode') or f'the task {status.lower()}'
            return Poll(state='failed', error=f'runway: {reason}')

        fraction = body.get('progress')
        return Poll(
            state='running' if status == 'RUNNING' else 'queued',
            progress=float(fraction) if isinstance(fraction, int | float) else None,
            note='generating at Runway' if status == 'RUNNING' else 'queued at Runway',
        )

    async def cancel(self, job: Submitted) -> None:
        """A DELETE on the task, which is also how Runway deletes a finished one.

        Worth doing rather than leaving: a Runway task that is abandoned rather
        than cancelled runs to completion and is charged for.
        """
        try:
            async with self.transport.session(30) as session:
                async with session.delete(f'{self.base_url}/tasks/{job.id}', headers=self.headers()) as resp:
                    log.info('runway: cancelled %s upstream (HTTP %s)', job.id, resp.status)
        except Exception as exc:  # noqa: BLE001 - best effort by definition
            log.warning('runway: could not cancel %s upstream: %s', job.id, exc)

    async def generate(self, prompt: str, *, model: str = '', n: int = 1, size: str = '',
                       progress: Any = None, **kw: Any) -> list[Any]:
        return await self.run(prompt, model=model or 'gen4_turbo', progress=progress, **kw)

    async def models(self, kind: str = '') -> list[dict[str, Any]]:
        rows = VIDEO_MODELS if kind == 'video' else IMAGE_MODELS
        return [{'id': i, 'provider': self.provider_id, 'label': i, 'note': n} for i, n in rows]

    async def describe(self, model: str = '') -> list[Param]:
        model = (model or 'gen4_turbo').strip()

        if model.startswith('gen4_image'):
            params = [p for p in common_image() if p.name != 'negative_prompt']
            return params + [
                Param('ratio', 'Size', 'enum', default='1920:1080', options=list(IMAGE_RATIOS), group='shape',
                      help='Runway names these by pixels rather than by ratio.'),
                Param('reference_images', 'Reference images', 'image', group='prompt',
                      help='Up to three. A person, a place, a product — whatever has to stay the same.'),
                Param('reference_tags', 'Names for them', 'string', group='prompt',
                      help='Comma separated, in the same order. Then write them into the prompt with an '
                           '@ — "@sarah in @warehouse". This is what makes the reference specific rather '
                           'than merely an influence, and it is the reason to use Runway for this.'),
            ]

        return [
            Param('prompt', 'Prompt', 'text', group='prompt',
                  help='What should happen. Describe the motion, not only the scene — a still '
                       'description tends to produce a still.'),
            Param('image', 'Starting image', 'image', group='prompt',
                  help='Gen-4 Turbo is image-to-video and needs one. Without it this falls back to '
                       'text-to-video, which only some of these models support.'),
            Param('ratio', 'Size', 'enum', default='1280:720', options=list(VIDEO_RATIOS), group='shape'),
            Param('duration', 'Duration', 'enum', default='5', options=['5', '10'], group='motion',
                  help='Seconds. Runway takes these two and no others.'),
            Param('seed', 'Seed', 'seed', default=-1, group='sampling', advanced=True),
        ]
