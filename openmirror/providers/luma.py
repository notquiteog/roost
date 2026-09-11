"""Luma Dream Machine — Ray for video, Photon for stills.

Luma is here for camera movement. Its models were trained in a way that makes
"slow dolly in", "orbit left" and "crane up" mean something specific, and it
takes keyframes: a first frame, a last frame, or both, so a shot can be made
to start and end somewhere decided in advance. Nothing else in this build
does the end-frame part.

One constraint shapes this adapter and is worth stating plainly rather than
working around: **Luma takes reference images as public URLs only.** There is
no upload endpoint and no data URI. Every other provider here accepts bytes,
so the honest thing is a URL field that says so, rather than an image picker
that silently does nothing with the file it was given — which is what
pretending otherwise would produce.
"""

from __future__ import annotations

import logging
from typing import Any

from openmirror.media.params import Param, common_image
from openmirror.providers.hosted_media import HostedMedia, Poll, Submitted, urls_in

log = logging.getLogger(__name__)

API = 'https://api.lumalabs.ai/dream-machine/v1'

IMAGE_MODELS: tuple[tuple[str, str], ...] = (
    ('photon-1', 'Photon — Luma\'s image model. Strong at style, and cheap enough to iterate with.'),
    ('photon-flash-1', 'Photon Flash — faster and cheaper again.'),
)

VIDEO_MODELS: tuple[tuple[str, str], ...] = (
    ('ray-2', 'Ray 2 — the current one. The best camera language of anything here.'),
    ('ray-flash-2', 'Ray Flash 2 — quicker and cheaper, for finding the shot.'),
    ('ray-1-6', 'Ray 1.6 — the previous generation.'),
)


class LumaProvider(HostedMedia):
    provider_id = 'luma'

    def __init__(self, base_url: str = API, api_key: str = '', **kw: Any) -> None:
        super().__init__(base_url or API, api_key, **kw)

    async def submit(self, prompt: str, *, model: str, **kw: Any) -> Submitted:
        model = (model or 'ray-2').strip()
        is_image = model.startswith('photon')
        payload: dict[str, Any] = {'prompt': prompt, 'model': model}

        for key in ('aspect_ratio', 'resolution', 'duration', 'loop'):
            if kw.get(key) not in (None, ''):
                payload[key] = kw[key]

        if is_image:
            for field, target in (('image_ref', 'image_ref'), ('style_ref', 'style_ref'),
                                  ('character_ref', 'character_ref')):
                url = kw.get(field)
                if isinstance(url, str) and url.startswith('http'):
                    if target == 'character_ref':
                        payload[target] = {'identity0': {'images': [url]}}
                    else:
                        payload[target] = [{'url': url, 'weight': 0.85}]
            path = '/generations/image'
        else:
            # Keyframes: frame0 is where the shot starts and frame1 is where it
            # ends. Giving only frame1 is a legitimate and under-used request —
            # "arrive at this image" — so both are sent independently.
            frames: dict[str, Any] = {}
            for field, slot in (('start_image_url', 'frame0'), ('end_image_url', 'frame1')):
                url = kw.get(field)
                if isinstance(url, str) and url.startswith('http'):
                    frames[slot] = {'type': 'image', 'url': url}
            if frames:
                payload['keyframes'] = frames
            path = '/generations'

        body = await self.post_json(f'{self.base_url}{path}', payload)
        ident = str(body.get('id', ''))
        return Submitted(id=ident, poll_url=f'{self.base_url}/generations/{ident}')

    async def check(self, job: Submitted) -> Poll:
        body = await self.get_json(job.poll_url)
        state = str(body.get('state', '')).lower()

        if state == 'completed':
            return Poll(state='done', progress=1.0, outputs=urls_in(body.get('assets')))
        if state == 'failed':
            return Poll(state='failed', error=f'luma: {body.get("failure_reason") or "no reason given"}')
        return Poll(
            state='running' if state == 'dreaming' else 'queued',
            note='dreaming at Luma' if state == 'dreaming' else 'queued at Luma',
        )

    async def cancel(self, job: Submitted) -> None:
        try:
            async with self.transport.session(30) as session:
                async with session.post(
                    f'{self.base_url}/generations/{job.id}/cancel', headers=self.headers()
                ) as resp:
                    log.info('luma: cancelled %s upstream (HTTP %s)', job.id, resp.status)
        except Exception as exc:  # noqa: BLE001 - best effort by definition
            log.warning('luma: could not cancel %s upstream: %s', job.id, exc)

    async def generate(self, prompt: str, *, model: str = '', n: int = 1, size: str = '',
                       progress: Any = None, **kw: Any) -> list[Any]:
        return await self.run(prompt, model=model or 'ray-2', progress=progress, **kw)

    async def models(self, kind: str = '') -> list[dict[str, Any]]:
        rows = VIDEO_MODELS if kind == 'video' else IMAGE_MODELS
        return [{'id': i, 'provider': self.provider_id, 'label': i, 'note': n} for i, n in rows]

    async def describe(self, model: str = '') -> list[Param]:
        model = (model or 'ray-2').strip()
        ratios = ['1:1', '16:9', '9:16', '4:3', '3:4', '21:9', '9:21']

        if model.startswith('photon'):
            params = [p for p in common_image() if p.name not in ('negative_prompt', 'seed')]
            return params + [
                Param('aspect_ratio', 'Aspect ratio', 'enum', default='16:9', options=ratios, group='shape'),
                Param('image_ref', 'Reference image URL', 'string', group='prompt', advanced=True,
                      help='A publicly reachable URL. Luma has no upload endpoint and does not take '
                           'a file or a data URI, which is why this is a URL box and not a picker.'),
                Param('style_ref', 'Style reference URL', 'string', group='prompt', advanced=True,
                      help='Copies the look rather than the content. Also a public URL.'),
                Param('character_ref', 'Character reference URL', 'string', group='prompt', advanced=True,
                      help='A face to keep consistent across generations. Also a public URL.'),
            ]

        return [
            Param('prompt', 'Prompt', 'text', group='prompt',
                  help='Luma understands camera language — "slow dolly in", "orbit left", "crane up", '
                       '"handheld". Say what the camera does as well as what is in front of it.'),
            Param('aspect_ratio', 'Aspect ratio', 'enum', default='16:9', options=ratios, group='shape'),
            Param('resolution', 'Resolution', 'enum', default='720p',
                  options=['540p', '720p', '1080p', '4k'], group='shape'),
            Param('duration', 'Duration', 'enum', default='5s', options=['5s', '9s'], group='motion'),
            Param('loop', 'Loop', 'bool', default=False, group='motion',
                  help='Ends where it began, so it can play round without a cut.'),
            Param('start_image_url', 'Start frame URL', 'string', group='prompt', advanced=True,
                  help='A public URL for the first frame.'),
            Param('end_image_url', 'End frame URL', 'string', group='prompt', advanced=True,
                  help='A public URL for the last frame. Giving only this one is the under-used '
                       'request: "however it starts, arrive here".'),
        ]
