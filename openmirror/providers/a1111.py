"""Stable Diffusion behind the AUTOMATIC1111 API.

The A1111 API is large and mostly about reconfiguring the server. Only
generation is used here, which also happens to be all Perch exposes — its
image service allows txt2img, img2img and four read-only queries and refuses
the rest, so anything more ambitious would 404 there anyway.

What this file does spend effort on is the *knobs*. Diffusion is the one
generator where the difference between a default and a considered setting is
visible in every image, and hiding that behind three sliders would be the
wrong trade for the people who run their own hardware precisely so they can
turn things up. So `describe()` asks the server what samplers, schedulers,
upscalers and checkpoints it actually has and builds the form from that. A
list typed into this file would offer a sampler that was removed two releases
ago and omit the one that was added last week.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

from openmirror.media.params import Param, common_image
from openmirror.net.transport import Transport
from openmirror.providers.base import GeneratedMedia, ImageProvider

log = logging.getLogger(__name__)


class A1111Provider(ImageProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str = '',
        *,
        provider_id: str = 'automatic1111',
        timeout: int = 600,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.provider_id = provider_id
        self.timeout = timeout
        self.transport = transport or Transport(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        h = {'Content-Type': 'application/json'}
        if self.api_key:
            h['Authorization'] = f'Bearer {self.api_key}'
        return h

    # -- generation ---------------------------------------------------------

    async def generate(
        self, prompt: str, *, model: str = '', n: int = 1, size: str = '1024x1024',
        progress: Any = None, **kw: Any
    ) -> list[GeneratedMedia]:
        # Taken as a named argument rather than left in `kw`: everything still
        # in `kw` at the end is forwarded to A1111 verbatim, and `progress`
        # arriving there would be an unknown field on every request.
        try:
            width, height = (int(v) for v in size.lower().split('x', 1))
        except ValueError:
            width = height = 1024

        seed = kw.pop('seed', None)
        if seed in (-1, '', None):
            seed = None

        payload: dict[str, Any] = {
            'prompt': prompt,
            'negative_prompt': kw.pop('negative_prompt', ''),
            'width': int(kw.pop('width', width)),
            'height': int(kw.pop('height', height)),
            'batch_size': n,
            'steps': kw.pop('steps', 25),
            'cfg_scale': kw.pop('cfg_scale', 7.0),
            'sampler_name': kw.pop('sampler', 'DPM++ 2M'),
        }
        if seed is not None:
            payload['seed'] = int(seed)

        # Everything below is optional and only sent when set, because A1111
        # treats a present-but-empty field differently from an absent one for
        # several of them — an empty `scheduler` selects nothing rather than
        # the default, and the job fails with a KeyError from inside the queue.
        for key, target in (
            ('scheduler', 'scheduler'),
            ('clip_skip', 'override_settings'),
            ('subseed', 'subseed'),
            ('subseed_strength', 'subseed_strength'),
            ('tiling', 'tiling'),
            ('restore_faces', 'restore_faces'),
        ):
            value = kw.pop(key, None)
            if value in (None, ''):
                continue
            if target == 'override_settings':
                payload.setdefault('override_settings', {})['CLIP_stop_at_last_layers'] = int(value)
            else:
                payload[target] = value

        # Hires fix is four fields that only mean anything together, so it is
        # one switch here and expanded on the way out.
        if kw.pop('hires', False):
            payload['enable_hr'] = True
            payload['hr_scale'] = float(kw.pop('hires_scale', 2.0))
            payload['hr_upscaler'] = kw.pop('hires_upscaler', 'Latent')
            payload['hr_second_pass_steps'] = int(kw.pop('hires_steps', 0))
            payload['denoising_strength'] = float(kw.pop('denoising_strength', 0.7))
        else:
            for stale in ('hires_scale', 'hires_upscaler', 'hires_steps'):
                kw.pop(stale, None)

        # A starting image switches endpoint. img2img is the *same* request
        # with two extra fields, which is why it was worth wiring rather than
        # leaving to a second provider: Perch's allowlist already carries it,
        # and without it every local install here is text-to-image only.
        init = kw.pop('image', None)
        path = '/sdapi/v1/txt2img'
        if isinstance(init, bytes | bytearray):
            path = '/sdapi/v1/img2img'
            payload['init_images'] = [base64.b64encode(bytes(init)).decode()]
            # Only meaningful with an init image, and A1111's own default of
            # 0.75 changes more than most people expect from a slider they did
            # not touch.
            payload['denoising_strength'] = float(kw.pop('denoising_strength', 0.6))
            mask = kw.pop('mask', None)
            if isinstance(mask, bytes | bytearray):
                payload['mask'] = base64.b64encode(bytes(mask)).decode()
                payload['inpainting_fill'] = int(kw.pop('inpainting_fill', 1))
                payload['inpaint_full_res'] = bool(kw.pop('inpaint_full_res', True))
        else:
            # Without one these mean nothing, and `mask` in particular is a
            # 400 rather than a no-op.
            for stale in ('mask', 'inpainting_fill', 'inpaint_full_res'):
                kw.pop(stale, None)

        # A checkpoint is selected per request rather than by changing the
        # server's own settings, which Perch's allowlist forbids and which
        # would in any case race with anyone else generating at the time.
        if model:
            payload.setdefault('override_settings', {})['sd_model_checkpoint'] = model
            payload['override_settings_restore_afterwards'] = True
        payload.update(kw)

        if progress:
            # A1111 is synchronous over HTTP: the request returns when the
            # image does. There is a /progress endpoint, but polling it needs a
            # second connection to a server that is busy with this one, and
            # Perch's allowlist does not carry it — so what can honestly be
            # said is that it has started.
            progress(None, f'generating on {self.provider_id}')

        async with self.transport.session() as session:
            async with session.post(
                f'{self.base_url}{path}', json=payload, headers=self._headers()
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f'{self.provider_id}: HTTP {resp.status}: {(await resp.text())[:300]}')
                body = await resp.json()

        # A1111 returns the grid image as an extra entry when batching, so only
        # the first n are the actual results.
        images = body.get('images', [])[:n]
        # The seed it actually used, which is the only way to reproduce a
        # generation that was asked for with seed -1.
        used_seed = seed
        try:
            import json

            info = json.loads(body.get('info') or '{}')
            used_seed = info.get('seed', seed)
        except (ValueError, TypeError):
            pass

        return [
            GeneratedMedia(
                data=base64.b64decode(b),
                media_type='image/png',
                seed=used_seed,
                meta={'steps': payload['steps'], 'cfg_scale': payload['cfg_scale'],
                      'sampler': payload['sampler_name'], 'size': f'{payload["width"]}x{payload["height"]}'},
            )
            for b in images
        ]

    # -- what it can do -----------------------------------------------------

    async def models(self) -> list[dict[str, Any]]:
        body = await self._get('/sdapi/v1/sd-models')
        return [
            {'id': m.get('title') or m.get('model_name'), 'provider': self.provider_id}
            for m in (body or [])
        ]

    async def _get(self, path: str) -> Any:
        try:
            async with self.transport.session(30) as session:
                async with session.get(f'{self.base_url}{path}', headers=self._headers()) as resp:
                    if resp.status != 200:
                        return None
                    return await resp.json()
        except Exception as exc:  # noqa: BLE001 - a missing endpoint is an answer
            log.debug('%s: %s did not answer: %s', self.provider_id, path, exc)
            return None

    async def describe(self, model: str = '') -> list[Param]:
        """The form for this install, built from what it has installed.

        The four lists are fetched together: they are independent, each is a
        round trip, and doing them in sequence makes opening the panel feel
        like the server is slow when it is only being asked four questions.
        """
        samplers, schedulers, upscalers, loras = await asyncio.gather(
            self._get('/sdapi/v1/samplers'),
            self._get('/sdapi/v1/schedulers'),
            self._get('/sdapi/v1/upscalers'),
            self._get('/sdapi/v1/loras'),
        )

        def names(rows: Any, key: str = 'name') -> list[str]:
            return [str(r.get(key)) for r in rows if r.get(key)] if isinstance(rows, list) else []

        sampler_names = names(samplers)
        params = common_image()
        params += [
            Param('width', 'Width', 'int', default=1024, minimum=64, maximum=4096, step=64, group='shape',
                  help='Multiples of 64. SDXL was trained at 1024; going far above it '
                       'produces repeated limbs rather than more detail.'),
            Param('height', 'Height', 'int', default=1024, minimum=64, maximum=4096, step=64, group='shape'),
            Param('steps', 'Steps', 'int', default=25, minimum=1, maximum=150, step=1, group='sampling',
                  help='How many denoising passes. Past about 40 the picture stops '
                       'improving and only costs more.'),
            Param('cfg_scale', 'CFG scale', 'float', default=7.0, minimum=1.0, maximum=30.0, step=0.5,
                  group='sampling',
                  help='How hard to push toward the prompt. Low is loose and creative, '
                       'high is literal and eventually burnt-looking. 5-8 is the usual range.'),
            Param('sampler', 'Sampler', 'enum', default=sampler_names[0] if sampler_names else 'DPM++ 2M',
                  options=sampler_names, group='sampling',
                  help='The solver. These are the ones this server has installed.'),
        ]
        if schedulers:
            params.append(
                Param('scheduler', 'Scheduler', 'enum', options=names(schedulers), group='sampling',
                      advanced=True, help='The noise schedule. Karras is a good default where it is offered.')
            )
        params += [
            Param('clip_skip', 'CLIP skip', 'int', minimum=1, maximum=12, step=1, group='sampling',
                  advanced=True,
                  help='Stop reading the text encoder this many layers early. 2 for most '
                       'anime-style checkpoints, 1 for everything else.'),
            Param('hires', 'High-res fix', 'bool', default=False, group='quality',
                  help='Generate small, then upscale and re-diffuse. The usual way to get '
                       'a large image that is not a large mess.'),
            Param('hires_scale', 'High-res scale', 'float', default=2.0, minimum=1.0, maximum=4.0,
                  step=0.05, group='quality', advanced=True),
            Param('hires_upscaler', 'Upscaler', 'enum', options=names(upscalers), group='quality',
                  advanced=True, help='These are the upscalers installed on this server.'),
            Param('hires_steps', 'High-res steps', 'int', default=0, minimum=0, maximum=150, step=1,
                  group='quality', advanced=True, help='0 reuses the step count above.'),
            Param('denoising_strength', 'How much it may change', 'float', default=0.6, minimum=0.0,
                  maximum=1.0, step=0.01, group='quality',
                  help='Used by both the high-res second pass and img2img. 0 returns what it was '
                       'given; above about 0.6 it starts inventing rather than redrawing.'),
            Param('image', 'Starting image', 'image', group='prompt',
                  help='Supplying one switches this to img2img: the picture is redrawn rather than '
                       'made from noise. How much it may change is the denoising strength below.'),
            Param('mask', 'Inpainting mask', 'image', group='prompt', advanced=True,
                  help='Only used with a starting image. White is repainted, black is kept.'),
            Param('restore_faces', 'Restore faces', 'bool', default=False, group='quality', advanced=True),
            Param('tiling', 'Seamless tiling', 'bool', default=False, group='quality', advanced=True,
                  help='Makes an image that repeats without a visible seam.'),
            Param('subseed', 'Variation seed', 'seed', group='sampling', advanced=True,
                  help='Blended with the seed to make near-variants of one image.'),
            Param('subseed_strength', 'Variation strength', 'float', default=0.0, minimum=0.0,
                  maximum=1.0, step=0.01, group='sampling', advanced=True),
        ]

        if loras:
            # Not a parameter — A1111 takes LoRAs as `<lora:name:weight>` inside
            # the prompt — so they are offered as something to insert rather
            # than as a field that would silently do nothing.
            params.append(
                Param(
                    'loras_available', 'LoRAs on this server', 'enum',
                    options=names(loras), group='prompt', advanced=True,
                    help='Not a setting: add one to the prompt as <lora:name:0.8>. '
                         'Listed here so you know which names this server will accept.',
                )
            )
        return params
