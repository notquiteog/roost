"""MiniMax — Hailuo video, and the Director models.

Hailuo is here for two things the rest of the field does less well: prompt
adherence on a complicated instruction, and the **Director** variants, which
take shot instructions inline — `[Push in]`, `[Pan left]`, `[Tracking shot]`
written into the prompt itself and executed as camera moves rather than
described as scenery.

The one structural quirk is that a finished job is not a URL. MiniMax hands
back a `file_id`, and the download address is a second call to the files
endpoint. Doing that lookup here, inside the poll, rather than leaving it to
the caller, is what keeps every provider in this build returning the same
thing: bytes, and nothing about which company they came from.
"""

from __future__ import annotations

import base64
from typing import Any

from openmirror.media.params import Param, common_image
from openmirror.providers.hosted_media import HostedMedia, Poll, Submitted

API = 'https://api.minimax.io/v1'

VIDEO_MODELS: tuple[tuple[str, str], ...] = (
    ('MiniMax-Hailuo-02', 'Hailuo 02 — the current one. 1080p, and the best of these at following '
                          'a long, specific prompt.'),
    ('T2V-01-Director', 'Director, text to video — takes [Push in], [Pan left], [Truck right] and '
                        'similar shot instructions written into the prompt.'),
    ('I2V-01-Director', 'Director, image to video — the same shot instructions, starting from a still.'),
    ('I2V-01-live', 'Tuned for illustration and anime rather than for photographic motion.'),
    ('S2V-01', 'Subject reference: give it a face and it keeps that person across the shot.'),
    ('T2V-01', 'The previous general-purpose text-to-video model.'),
)

IMAGE_MODELS: tuple[tuple[str, str], ...] = (
    ('image-01', 'MiniMax image-01.'),
)


class MiniMaxProvider(HostedMedia):
    provider_id = 'minimax'

    def __init__(self, base_url: str = API, api_key: str = '', *, kind: str = 'video', **kw: Any) -> None:
        super().__init__(base_url or API, api_key, **kw)
        self.kind = kind

    async def submit(self, prompt: str, *, model: str, **kw: Any) -> Submitted:
        payload: dict[str, Any] = {'model': model or 'MiniMax-Hailuo-02', 'prompt': prompt}
        for key in ('duration', 'resolution', 'prompt_optimizer'):
            if kw.get(key) not in (None, ''):
                payload[key] = kw[key]
        if payload.get('duration') is not None:
            payload['duration'] = int(payload['duration'])

        image = kw.get('first_frame_image')
        if isinstance(image, bytes | bytearray):
            # A data URI here, which MiniMax does take — unlike Kling, which
            # wants bare base64. The two are a byte apart and neither says so.
            from openmirror.providers.hosted_media import data_uri

            payload['first_frame_image'] = data_uri(bytes(image))

        body = await self.post_json(f'{self.base_url}/video_generation', payload)
        self._raise_for_base_resp(body)
        return Submitted(id=str(body.get('task_id', '')),
                         poll_url=f'{self.base_url}/query/video_generation?task_id={body.get("task_id")}')

    def _raise_for_base_resp(self, body: dict[str, Any]) -> None:
        """MiniMax returns failures as a 200 with a status code inside.

        So the HTTP status says nothing, and a request that failed for lack of
        credit looks exactly like one that worked until this is read.
        """
        base = body.get('base_resp') or {}
        code = base.get('status_code')
        if code not in (0, None):
            raise RuntimeError(f'minimax: {base.get("status_msg") or code}')

    async def check(self, job: Submitted) -> Poll:
        body = await self.get_json(job.poll_url)
        status = str(body.get('status', ''))

        if status == 'Success':
            file_id = body.get('file_id')
            if not file_id:
                return Poll(state='failed', error='minimax: the task succeeded with no file attached')
            listing = await self.get_json(f'{self.base_url}/files/retrieve?file_id={file_id}')
            url = (listing.get('file') or {}).get('download_url')
            if not url:
                return Poll(state='failed', error='minimax: the file has no download URL')
            return Poll(state='done', progress=1.0, outputs=[url])

        if status in ('Fail', 'Failed'):
            base = body.get('base_resp') or {}
            return Poll(state='failed', error=f'minimax: {base.get("status_msg") or "the task failed"}')

        note = {'Queueing': 'queued at MiniMax', 'Preparing': 'preparing at MiniMax'}.get(
            status, 'generating at MiniMax')
        return Poll(state='queued' if status in ('Queueing', 'Preparing') else 'running', note=note)

    # -- images are synchronous ---------------------------------------------

    async def generate(self, prompt: str, *, model: str = '', n: int = 1, size: str = '',
                       progress: Any = None, **kw: Any) -> list[Any]:
        if self.kind == 'image':
            return await self._images(prompt, model or 'image-01', n, progress, kw)
        return await self.run(prompt, model=model or 'MiniMax-Hailuo-02', progress=progress, **kw)

    async def _images(self, prompt: str, model: str, n: int, progress: Any, kw: dict[str, Any]) -> list[Any]:
        from openmirror.providers.base import GeneratedMedia

        payload: dict[str, Any] = {
            'model': model,
            'prompt': prompt,
            'n': max(1, min(9, int(n or 1))),
            'response_format': 'base64',
        }
        for key in ('aspect_ratio', 'prompt_optimizer', 'width', 'height'):
            if kw.get(key) not in (None, ''):
                payload[key] = kw[key]

        if progress:
            progress(None, 'generating at MiniMax')
        body = await self.post_json(f'{self.base_url}/image_generation', payload, seconds=self.timeout)
        self._raise_for_base_resp(body)

        data = body.get('data') or {}
        out: list[Any] = []
        for blob in data.get('image_base64') or []:
            out.append(GeneratedMedia(data=base64.b64decode(blob), media_type='image/jpeg'))
        for url in data.get('image_urls') or []:
            raw, media_type = await self.download(url)
            out.append(GeneratedMedia(data=raw, media_type=media_type))
        return out

    async def models(self, kind: str = '') -> list[dict[str, Any]]:
        rows = IMAGE_MODELS if (kind or self.kind) == 'image' else VIDEO_MODELS
        return [{'id': i, 'provider': self.provider_id, 'label': i, 'note': n} for i, n in rows]

    async def describe(self, model: str = '') -> list[Param]:
        if self.kind == 'image':
            return [p for p in common_image() if p.name not in ('negative_prompt', 'seed')] + [
                Param('aspect_ratio', 'Aspect ratio', 'enum', default='16:9',
                      options=['1:1', '16:9', '4:3', '3:2', '2:3', '3:4', '9:16', '21:9'], group='shape'),
                Param('prompt_optimizer', 'Expand the prompt', 'bool', default=True, group='prompt',
                      advanced=True,
                      help='MiniMax rewrites a short prompt into a longer one. Turn it off when the '
                           'wording is deliberate.'),
            ]

        return [
            Param('prompt', 'Prompt', 'text', group='prompt',
                  help='On a Director model, shot instructions in square brackets are executed rather '
                       'than described: [Push in], [Pan left], [Tracking shot], [Static shot]. '
                       'Up to three per prompt.'),
            Param('first_frame_image', 'Starting frame', 'image', group='prompt',
                  help='Required by the I2V models, optional elsewhere.'),
            Param('duration', 'Duration', 'enum', default='6', options=['6', '10'], group='motion',
                  help='Seconds. 10 is only available at 768p on Hailuo 02.'),
            Param('resolution', 'Resolution', 'enum', default='1080P', options=['512P', '768P', '1080P'],
                  group='shape'),
            Param('prompt_optimizer', 'Expand the prompt', 'bool', default=True, group='prompt',
                  advanced=True,
                  help='Off when you have written the shot instructions yourself — it rewrites them.'),
        ]
