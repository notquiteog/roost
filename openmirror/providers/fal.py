"""fal, the other way to reach most of the field.

fal and Replicate overlap heavily and are worth having both, for a reason that
is about failure rather than features: they are two independent routes to the
same open-weight models, so a model being down, rate-limited or withdrawn on
one is not the end of the road. Between them they carry FLUX, Wan, Kling, Veo,
Hailuo, LTX, Hunyuan, Seedream, Qwen-Image, Imagen, Recraft, Ideogram, Pika,
Luma, Mochi and CogVideoX, which is most of what anyone is actually running.

fal publishes OpenAPI per endpoint, as Replicate does, so the form comes from
the model — see `openmirror.providers.openapi_form`, which is shared between them
precisely so the two do not drift.

Three places fal differs from Replicate, all of them shallow and all of them
worth getting right rather than approximating:

**The input is the body.** Replicate wraps it in `{"input": …}`; fal posts the
object itself. Wrapping it anyway produces a 422 that names none of the real
fields.

**Cancel is a PUT.** Not a POST. The generic cancellation in `HostedMedia`
posts, so this overrides it — and it matters more here than almost anywhere,
because a queued fal job that is never cancelled still runs and still bills.

**The queue position is the honest progress.** fal reports where you are in
the line rather than how far along the model is, and "third in the queue" is
genuinely more useful to someone waiting than a bar that cannot move until the
job starts.

The model list is from fal's documentation rather than from fal: there is no
public endpoint that enumerates what it hosts. It is therefore a starting
point and clearly marked as one — anything fal serves can be typed in, because
the id in the box is the endpoint path and nothing here validates it.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from openmirror.media.params import Param
from openmirror.providers.base import GeneratedMedia
from openmirror.providers.hosted_media import HostedMedia, Poll, Submitted, data_uri, urls_in
from openmirror.providers.openapi_form import params_from_openapi

log = logging.getLogger(__name__)

QUEUE = 'https://queue.fal.run'
SCHEMA = 'https://fal.ai/api/openapi/queue/openapi.json'

#: From fal's documentation, not from fal — it publishes no listing endpoint.
#: A starting point for the picker and never a limit: the model box takes any
#: endpoint path, which is how something released this morning gets used today.
KNOWN: dict[str, tuple[tuple[str, str], ...]] = {
    'image': (
        ('fal-ai/flux-pro/v1.1-ultra', 'FLUX1.1 [pro] ultra — up to 4MP, the sharpest of the FLUX line'),
        ('fal-ai/flux/dev', 'FLUX.1 [dev] — open weights, 12B, the common default'),
        ('fal-ai/flux/schnell', 'FLUX.1 [schnell] — four steps, for iterating on a composition'),
        ('fal-ai/flux-kontext/dev', 'FLUX.1 Kontext — edits an image you give it, in place'),
        ('fal-ai/nano-banana', 'Gemini 2.5 Flash Image — conversational editing, strong at likeness'),
        ('fal-ai/bytedance/seedream/v4/text-to-image', 'Seedream 4 — 4K, unusually good with text in-image'),
        ('fal-ai/qwen-image', 'Qwen-Image — open weights, best-in-class at rendering text'),
        ('fal-ai/imagen4/preview', 'Imagen 4 — Google, photographic'),
        ('fal-ai/recraft-v3', 'Recraft V3 — vector and brand styles, real SVG output'),
        ('fal-ai/ideogram/v3', 'Ideogram V3 — typography and posters'),
        ('fal-ai/stable-diffusion-v35-large', 'Stable Diffusion 3.5 Large — open weights'),
        ('fal-ai/hidream-i1-full', 'HiDream-I1 — open weights, 17B'),
        ('fal-ai/luma-photon', 'Luma Photon — cheap and fast, good at style'),
    ),
    'video': (
        ('fal-ai/veo3', 'Veo 3 — Google, and the only one of these that makes its own audio'),
        ('fal-ai/kling-video/v2/master/text-to-video', 'Kling 2 Master — the strongest at physical motion'),
        ('fal-ai/minimax/hailuo-02/standard/text-to-video', 'Hailuo 02 — MiniMax, strong prompt adherence'),
        ('fal-ai/bytedance/seedance/v1/pro/text-to-video', 'Seedance 1 Pro — ByteDance, multi-shot'),
        ('fal-ai/wan-t2v', 'Wan 2.x — open weights, the one you can also run locally'),
        ('fal-ai/wan-i2v', 'Wan image-to-video — animates a still you give it'),
        ('fal-ai/ltx-video', 'LTX Video — real-time class, open weights'),
        ('fal-ai/hunyuan-video', 'HunyuanVideo — open weights, 13B'),
        ('fal-ai/pika/v2.2/text-to-video', 'Pika 2.2 — stylised, quick'),
        ('fal-ai/luma-dream-machine', 'Luma Ray — cheap, good camera movement'),
        ('fal-ai/mochi-v1', 'Mochi 1 — open weights, Apache licensed'),
    ),
}


class FalProvider(HostedMedia):
    provider_id = 'fal'

    def __init__(self, base_url: str = QUEUE, api_key: str = '', **kw: Any) -> None:
        super().__init__(base_url or QUEUE, api_key, **kw)
        self._schemas: dict[str, tuple[float, list[Param]]] = {}
        self._fields: dict[str, set[str]] = {}

    def headers(self) -> dict[str, str]:
        # `Key`, not `Bearer`. fal answers a Bearer token with a 401 that says
        # nothing about the scheme, which is a long way to go to find a typo.
        h = {'Content-Type': 'application/json'}
        if self.api_key:
            h['Authorization'] = f'Key {self.api_key}'
        return h

    # -- submitting ---------------------------------------------------------

    async def submit(self, prompt: str, *, model: str, **kw: Any) -> Submitted:
        if not model:
            raise RuntimeError('fal: no model. Give it an endpoint path, such as fal-ai/flux/dev.')

        body = await self.post_json(f'{self.base_url}/{model.strip("/")}', await self._input(prompt, model, kw))
        request_id = str(body.get('request_id') or body.get('id') or '')
        if not request_id:
            raise RuntimeError(f'fal: the queue accepted nothing it would give an id for: {body}')
        return Submitted(
            id=request_id,
            poll_url=body.get('status_url') or '',
            result_url=body.get('response_url') or '',
            cancel_url=body.get('cancel_url') or '',
        )

    async def _input(self, prompt: str, model: str, kw: dict[str, Any]) -> dict[str, Any]:
        fields = await self._field_names(model)
        out: dict[str, Any] = {}
        for key, value in kw.items():
            if value is None or value == '':
                continue
            if fields and key not in fields:
                continue
            out[key] = data_uri(value) if isinstance(value, bytes | bytearray) else value
        if not fields or 'prompt' in fields:
            out['prompt'] = prompt
        if out.get('seed') in (-1, '-1'):
            out.pop('seed')
        return out

    async def check(self, job: Submitted) -> Poll:
        body = await self.get_json(f'{job.poll_url}?logs=0')
        status = str(body.get('status', '')).upper()

        if status == 'COMPLETED':
            # The status response says it finished; the payload lives at the
            # response URL, which is a second request fal requires.
            result = await self.get_json(job.result_url or job.poll_url.removesuffix('/status'))
            seed = result.get('seed')
            return Poll(
                state='done',
                progress=1.0,
                outputs=urls_in(result),
                seed=int(seed) if isinstance(seed, int | float) else None,
            )

        if status in ('FAILED', 'ERROR', 'CANCELLED'):
            detail = body.get('error') or body.get('detail') or f'the request {status.lower()}'
            return Poll(state='failed', error=str(detail)[:400])

        position = body.get('queue_position')
        if status == 'IN_QUEUE' and isinstance(position, int):
            # A position, not a fraction. There is no honest percentage for a
            # job that has not started, and `1 - 1/position` would be one.
            place = 'next' if position == 0 else f'number {position + 1} in the queue'
            return Poll(state='queued', note=f'waiting at fal — {place}')
        return Poll(state='running' if status == 'IN_PROGRESS' else 'queued', note='generating at fal')

    async def cancel(self, job: Submitted) -> None:
        """A PUT, which is the one thing fal spells differently from everyone.

        Worth the override rather than a shrug: a fal job left in the queue
        runs and bills whether or not anyone is still waiting for it.
        """
        if not job.cancel_url:
            return
        try:
            async with self.transport.session(30) as session:
                async with session.put(job.cancel_url, headers=self.headers()) as resp:
                    log.info('fal: cancelled %s upstream (HTTP %s)', job.id, resp.status)
        except Exception as exc:  # noqa: BLE001 - best effort by definition
            log.warning('fal: could not cancel %s upstream: %s', job.id, exc)

    # -- generation ---------------------------------------------------------

    async def generate(
        self, prompt: str, *, model: str = '', n: int = 1, size: str = '', progress: Any = None, **kw: Any
    ) -> list[GeneratedMedia]:
        fields = await self._field_names(model)
        if n and n > 1 and 'num_images' in fields and 'num_images' not in kw:
            kw['num_images'] = n
        return await self.run(prompt, model=model, progress=progress, **kw)

    # -- what it can do -----------------------------------------------------

    async def models(self, kind: str = '') -> list[dict[str, Any]]:
        return [
            {'id': ident, 'provider': self.provider_id, 'label': ident, 'note': note}
            for ident, note in KNOWN.get(kind or 'image', KNOWN['image'])
        ]

    async def _field_names(self, model: str) -> set[str]:
        if model not in self._fields:
            await self.describe(model)
        return self._fields.get(model, set())

    async def describe(self, model: str = '') -> list[Param]:
        if not model:
            return [Param('prompt', 'Prompt', 'text', group='prompt',
                          help='Pick a model first — the controls come from the model itself.')]

        cached = self._schemas.get(model)
        if cached and time.monotonic() - cached[0] < 300:
            return cached[1]

        try:
            doc = await self.get_json(f'{SCHEMA}?endpoint_id={model.strip("/")}', seconds=30)
            params = params_from_openapi(doc)
        except Exception as exc:  # noqa: BLE001
            log.info('fal: no schema for %s: %s', model, exc)
            params = []

        if not params:
            # A schema that could not be read is not a reason to refuse the
            # model: fal will still take a prompt, and several endpoints want
            # nothing else.
            params = [Param('prompt', 'Prompt', 'text', group='prompt',
                            help=f'fal published no readable schema for {model}, so only the prompt is offered. '
                                 'Anything else this endpoint takes can still be sent through the API.')]

        self._schemas[model] = (time.monotonic(), params)
        self._fields[model] = {p.name for p in params}
        return params
