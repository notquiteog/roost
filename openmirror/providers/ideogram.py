"""Ideogram, which is the one to reach for when the picture has words in it.

Every other image model treats text as texture. Ideogram treats it as text,
and that single difference is why it is worth its own adapter rather than
being left to an aggregator: a poster, a logo, a sign, a book cover or a
diagram with legible labels is a request the rest of the field quietly fails.

It answers synchronously — no job, no polling — which is why this is a plain
provider and not a `HostedMedia`. The request is multipart rather than JSON,
which is Ideogram's choice and the only awkward thing here: a JSON body gets a
415 that does not explain itself.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from openmirror.media.params import Param, common_image
from openmirror.net.transport import Transport
from openmirror.providers.base import GeneratedMedia, ImageProvider
from openmirror.providers.hosted_media import sniff

log = logging.getLogger(__name__)

API = 'https://api.ideogram.ai'

MODELS: tuple[tuple[str, str], ...] = (
    ('v3', 'Ideogram 3.0 — the current one, and the best of the field at text in an image.'),
    ('v2', 'The previous generation, kept because prompts tuned against it behave differently on v3.'),
    ('v2-turbo', 'Faster and cheaper v2, for iterating on a layout before committing.'),
)

#: Ideogram names resolutions by ratio. These are the ones its v3 endpoint
#: documents; an unknown one is a 422 rather than a nearest match.
RATIOS = ('1x1', '16x9', '9x16', '4x3', '3x4', '3x2', '2x3', '16x10', '10x16', '1x3', '3x1')


class IdeogramProvider(ImageProvider):
    def __init__(
        self,
        base_url: str = API,
        api_key: str = '',
        *,
        provider_id: str = 'ideogram',
        timeout: int = 300,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = (base_url or API).rstrip('/')
        self.api_key = api_key
        self.provider_id = provider_id
        self.timeout = timeout
        self.transport = transport or Transport(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        # No Content-Type: aiohttp sets the multipart boundary, and setting it
        # here would overwrite the boundary with one that is not in the body.
        return {'Api-Key': self.api_key} if self.api_key else {}

    async def generate(
        self, prompt: str, *, model: str = 'v3', n: int = 1, size: str = '', progress: Any = None, **kw: Any
    ) -> list[GeneratedMedia]:
        model = (model or 'v3').strip().lower().lstrip('v') or '3'
        path = f'/v1/ideogram-v{model[0]}/generate' if model[0] in '23' else '/v1/ideogram-v3/generate'

        form = aiohttp.FormData()
        form.add_field('prompt', prompt)
        form.add_field('num_images', str(max(1, min(8, int(n or 1)))))
        for key in ('aspect_ratio', 'rendering_speed', 'style_type', 'negative_prompt', 'magic_prompt'):
            value = kw.get(key)
            if value not in (None, '', 'auto'):
                form.add_field(key, str(value))
        seed = kw.get('seed')
        if seed not in (None, '', -1, '-1'):
            form.add_field('seed', str(int(seed)))
        # A style reference is sent as the file itself; several may be given
        # and Ideogram averages them.
        reference = kw.get('style_reference_images')
        for index, blob in enumerate(reference if isinstance(reference, list) else [reference]):
            if isinstance(blob, bytes | bytearray):
                form.add_field('style_reference_images', bytes(blob),
                               filename=f'reference-{index}.png', content_type=sniff(bytes(blob)))

        if progress:
            progress(None, 'generating at Ideogram')

        async with self.transport.session(self.timeout) as session:
            async with session.post(f'{self.base_url}{path}', data=form, headers=self._headers()) as resp:
                if resp.status != 200:
                    raise RuntimeError(f'ideogram: HTTP {resp.status}: {(await resp.text())[:400]}')
                body = await resp.json()

        out: list[GeneratedMedia] = []
        for row in body.get('data', []):
            url = row.get('url')
            if not url:
                continue
            async with self.transport.session(120) as session:
                async with session.get(url) as media:
                    if media.status != 200:
                        log.warning('ideogram: a result URL would not download: HTTP %s', media.status)
                        continue
                    data = await media.read()
            out.append(GeneratedMedia(
                data=data,
                media_type=sniff(data, url=url),
                seed=row.get('seed'),
                meta={'revised_prompt': row.get('prompt', '')},
            ))
        return out

    async def models(self, kind: str = '') -> list[dict[str, Any]]:
        if kind == 'video':
            return []
        return [{'id': i, 'provider': self.provider_id, 'label': f'Ideogram {i}', 'note': n} for i, n in MODELS]

    async def describe(self, model: str = '') -> list[Param]:
        params = common_image()
        return params + [
            Param('aspect_ratio', 'Aspect ratio', 'enum', default='1x1', options=list(RATIOS), group='shape',
                  help='Ideogram spells these with an x rather than a colon.'),
            Param('rendering_speed', 'Rendering speed', 'enum', default='DEFAULT',
                  options=['TURBO', 'DEFAULT', 'QUALITY'], group='quality',
                  help='TURBO for trying a layout, QUALITY for the one you keep.'),
            Param('style_type', 'Style', 'enum', default='AUTO',
                  options=['AUTO', 'GENERAL', 'REALISTIC', 'DESIGN'], group='quality',
                  help='DESIGN is the one for logos, posters and anything with a typeface in it.'),
            Param('magic_prompt', 'Magic prompt', 'enum', default='AUTO', options=['AUTO', 'ON', 'OFF'],
                  group='prompt', advanced=True,
                  help='Ideogram rewriting your prompt before generating. Turn it OFF when the '
                       'wording matters — including when the wording is what you want rendered.'),
            Param('style_reference_images', 'Style reference', 'image', group='prompt', advanced=True,
                  help='Pictures to copy the look of, not the content.'),
        ]
