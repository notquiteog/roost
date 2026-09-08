"""Stable Diffusion behind the AUTOMATIC1111 API.

The A1111 API is large and mostly about reconfiguring the server. Only
generation is used here, which also happens to be all Perch exposes — its
image service allows txt2img, img2img and four read-only queries and refuses
the rest, so anything more ambitious would 404 there anyway.
"""

from __future__ import annotations

import base64
from typing import Any

import aiohttp

from roost.providers.base import GeneratedMedia, ImageProvider


class A1111Provider(ImageProvider):
    def __init__(
        self, base_url: str, api_key: str = '', *, provider_id: str = 'automatic1111', timeout: int = 600
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.provider_id = provider_id
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {'Content-Type': 'application/json'}
        if self.api_key:
            h['Authorization'] = f'Bearer {self.api_key}'
        return h

    async def generate(
        self, prompt: str, *, model: str = '', n: int = 1, size: str = '1024x1024', **kw: Any
    ) -> list[GeneratedMedia]:
        try:
            width, height = (int(v) for v in size.lower().split('x', 1))
        except ValueError:
            width = height = 1024

        payload: dict[str, Any] = {
            'prompt': prompt,
            'negative_prompt': kw.pop('negative_prompt', ''),
            'width': width,
            'height': height,
            'batch_size': n,
            'steps': kw.pop('steps', 25),
            'cfg_scale': kw.pop('cfg_scale', 7.0),
            'sampler_name': kw.pop('sampler', 'DPM++ 2M'),
        }
        seed = kw.pop('seed', None)
        if seed is not None:
            payload['seed'] = seed
        # A checkpoint is selected per request rather than by changing the
        # server's own settings, which Perch's allowlist forbids and which
        # would in any case race with anyone else generating at the time.
        if model:
            payload.setdefault('override_settings', {})['sd_model_checkpoint'] = model
            payload['override_settings_restore_afterwards'] = True
        payload.update(kw)

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f'{self.base_url}/sdapi/v1/txt2img', json=payload, headers=self._headers()
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f'{self.provider_id}: HTTP {resp.status}: {(await resp.text())[:300]}')
                body = await resp.json()

        # A1111 returns the grid image as an extra entry when batching, so only
        # the first n are the actual results.
        images = body.get('images', [])[:n]
        return [GeneratedMedia(data=base64.b64decode(b), media_type='image/png', seed=seed) for b in images]

    async def models(self) -> list[dict[str, Any]]:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f'{self.base_url}/sdapi/v1/sd-models', headers=self._headers()) as resp:
                if resp.status != 200:
                    return []
                body = await resp.json()
        return [{'id': m.get('title') or m.get('model_name'), 'provider': self.provider_id} for m in body]
