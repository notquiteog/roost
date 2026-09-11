"""Sora, on the same key as everything else OpenAI.

Separate from `openai_compat` rather than folded into it, because this is not
an OpenAI-*shaped* API that every gateway copied — it is OpenAI's own video
endpoint, and nothing else serves it. Putting it in the adapter that Groq,
Together, LM Studio, vLLM and llama.cpp all share would mean five services
advertising a capability none of them has.

It is the best-behaved of the hosted video APIs, and the reason is one field:
`progress`, an honest percentage, on every poll. Most of the others report a
queue position or nothing at all.

Two details worth knowing before they surprise someone:

**The video is not in the status response.** The job says `completed` and the
bytes come from a separate `/content` request, which is also the only one of
these that needs the API key to download — the rest hand back a signed URL on
a CDN.

**`size` and `seconds` are closed sets.** 1280×720, 720×1280, 1024×1808 or
1808×1024, and 4, 8 or 12 seconds. Anything else is a 400 listing the
permitted values, which the form offers instead.
"""

from __future__ import annotations

from typing import Any

import aiohttp

from openmirror.media.params import Param
from openmirror.providers.hosted_media import HostedMedia, Poll, Submitted, sniff

API = 'https://api.openai.com/v1'

MODELS: tuple[tuple[str, str], ...] = (
    ('sora-2', 'Sora 2 — synchronised dialogue and sound effects, and physics that mostly hold up.'),
    ('sora-2-pro', 'Sora 2 Pro — slower, more expensive, and the one for anything final.'),
)

SIZES = ('1280x720', '720x1280', '1024x1808', '1808x1024')


class SoraProvider(HostedMedia):
    provider_id = 'sora'

    def __init__(self, base_url: str = API, api_key: str = '', **kw: Any) -> None:
        super().__init__(base_url or API, api_key, **kw)

    async def submit(self, prompt: str, *, model: str, **kw: Any) -> Submitted:
        model = (model or 'sora-2').strip()
        reference = kw.get('input_reference')

        if isinstance(reference, bytes | bytearray):
            # With a reference the request is multipart, because the reference
            # is a file rather than a field. Without one it is plain JSON, and
            # sending multipart anyway works but makes every error message
            # about form parsing rather than about the request.
            form = aiohttp.FormData()
            form.add_field('model', model)
            form.add_field('prompt', prompt)
            for key in ('seconds', 'size'):
                if kw.get(key) not in (None, ''):
                    form.add_field(key, str(kw[key]))
            form.add_field('input_reference', bytes(reference),
                           filename='reference.png', content_type=sniff(bytes(reference)))

            headers = {'Authorization': f'Bearer {self.api_key}'} if self.api_key else {}
            async with self.transport.session(300) as session:
                async with session.post(f'{self.base_url}/videos', data=form, headers=headers) as resp:
                    if resp.status >= 300:
                        raise RuntimeError(f'sora: HTTP {resp.status}: {(await resp.text())[:400]}')
                    body = await resp.json()
        else:
            payload: dict[str, Any] = {'model': model, 'prompt': prompt}
            for key in ('seconds', 'size'):
                if kw.get(key) not in (None, ''):
                    payload[key] = str(kw[key])
            body = await self.post_json(f'{self.base_url}/videos', payload)

        ident = str(body.get('id', ''))
        return Submitted(id=ident, poll_url=f'{self.base_url}/videos/{ident}',
                         result_url=f'{self.base_url}/videos/{ident}/content')

    async def check(self, job: Submitted) -> Poll:
        body = await self.get_json(job.poll_url)
        status = str(body.get('status', ''))

        if status == 'completed':
            return Poll(state='done', progress=1.0, outputs=[job.result_url])
        if status == 'failed':
            error = body.get('error') or {}
            return Poll(state='failed', error=f'sora: {error.get("message") or "the job failed"}')

        percent = body.get('progress')
        return Poll(
            state='running' if status == 'in_progress' else 'queued',
            progress=(float(percent) / 100) if isinstance(percent, int | float) else None,
            note='generating at OpenAI' if status == 'in_progress' else 'queued at OpenAI',
        )

    async def cancel(self, job: Submitted) -> None:
        """There is no cancel endpoint for a video job.

        Said here rather than left to the generic implementation, because the
        consequence is real: stopping a Sora job in this UI stops the waiting,
        not the billing. The log line is what makes that visible afterwards.
        """
        import logging

        logging.getLogger(__name__).info(
            'sora: %s was cancelled locally. OpenAI has no cancel endpoint for video, so it will '
            'run to completion and be charged for.', job.id
        )

    async def generate(self, prompt: str, *, model: str = '', n: int = 1, size: str = '',
                       progress: Any = None, **kw: Any) -> list[Any]:
        return await self.run(prompt, model=model or 'sora-2', progress=progress, **kw)

    async def models(self, kind: str = '') -> list[dict[str, Any]]:
        if kind == 'image':
            return []
        return [{'id': i, 'provider': self.provider_id, 'label': i, 'note': n} for i, n in MODELS]

    async def describe(self, model: str = '') -> list[Param]:
        return [
            Param('prompt', 'Prompt', 'text', group='prompt',
                  help='Sora makes the audio too. Dialogue in quotes is spoken; described sound — '
                       'rain on a window, a door closing — is produced.'),
            Param('input_reference', 'Reference image', 'image', group='prompt',
                  help='A first frame, or an image to match the look of. It must be exactly the '
                       'size chosen below.'),
            Param('size', 'Size', 'enum', default='1280x720', options=list(SIZES), group='shape'),
            Param('seconds', 'Duration', 'enum', default='4', options=['4', '8', '12'], group='motion'),
        ]
