"""Replicate, which is most of the field behind one key.

This adapter matters out of proportion to its size, and the reason is not that
Replicate hosts a lot of models. It is that Replicate **publishes each model's
input schema**, as OpenAPI, at a stable address — so the form for a model
nobody has ever heard of can be built correctly, with its real ranges, its real
enums and its own descriptions, without anybody adding a line here.

That is the same bargain `describe()` already strikes with A1111, extended to
several thousand models: FLUX, SDXL, Imagen, Seedream, Qwen-Image, Ideogram,
Recraft, Wan, Kling, Hunyuan, LTX, Veo, Mochi, CogVideoX and whatever is
published next week. None of them are named in this file, and that is the
point — a list of model names here would be out of date the week it was
written.

Two things the schema translation gets right, because getting them wrong is
how a generated form becomes actively misleading:

**An enum is an enum.** Replicate expresses one as `allOf: [$ref]` pointing at
a component that carries the values, so a naive reader sees "a string" and
renders a text box — and a text box for `aspect_ratio` is a field where `16:9`
and `16x9` look equally plausible and one of them is a 422.

**A file input is a file input.** `format: uri` is how an image-to-image or
image-to-video model says it needs a picture. Rendering that as a text box is
the difference between an image-to-video model working and appearing broken.

`x-order` is honoured, so the fields appear in the order the model's author
put them in, which is nearly always more sensible than alphabetical.
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

API = 'https://api.replicate.com/v1'

#: Replicate's own curated collections, which is the only model listing it
#: offers that is scoped by what a model *does*. Asked per kind so the video
#: picker does not fill up with image models.
COLLECTIONS = {
    'image': ('text-to-image', 'image-editing'),
    'video': ('text-to-video', 'image-to-video'),
}


class ReplicateProvider(HostedMedia):
    provider_id = 'replicate'

    def __init__(self, base_url: str = API, api_key: str = '', **kw: Any) -> None:
        super().__init__(base_url or API, api_key, **kw)
        # Schemas are fetched per model and cached, because the form is
        # re-described on every provider or model change and a round trip to
        # Replicate on each of those makes the panel feel broken.
        self._schemas: dict[str, tuple[float, list[Param]]] = {}
        self._fields: dict[str, set[str]] = {}

    def headers(self) -> dict[str, str]:
        h = {'Content-Type': 'application/json'}
        if self.api_key:
            h['Authorization'] = f'Bearer {self.api_key}'
        return h

    # -- submitting ---------------------------------------------------------

    @staticmethod
    def _split(model: str) -> tuple[str, str]:
        """`owner/name`, `owner/name:version`, or a bare version hash.

        All three are in circulation — the first is what the website shows, the
        second is what a reproducible script pins, and the third is what the
        API returns — and a person pasting any of them should get a
        generation rather than a lecture.
        """
        model = (model or '').strip()
        if ':' in model:
            name, version = model.split(':', 1)
            return name.strip(), version.strip()
        return model, ''

    async def submit(self, prompt: str, *, model: str, **kw: Any) -> Submitted:
        name, version = self._split(model)
        if not name and not version:
            raise RuntimeError('replicate: no model. Give it an owner/name, as shown on replicate.com.')

        payload: dict[str, Any] = {'input': await self._input(prompt, model, kw)}
        if version and '/' not in version:
            payload['version'] = version
            url = f'{self.base_url}/predictions'
        else:
            url = f'{self.base_url}/models/{name}/predictions'

        body = await self.post_json(url, payload)
        urls = body.get('urls') or {}
        return Submitted(
            id=str(body.get('id', '')),
            poll_url=urls.get('get') or f'{self.base_url}/predictions/{body.get("id")}',
            cancel_url=urls.get('cancel') or '',
        )

    async def _input(self, prompt: str, model: str, kw: dict[str, Any]) -> dict[str, Any]:
        """The `input` object, narrowed to fields this model actually has.

        Narrowed rather than sent whole: Replicate rejects an unknown input
        with a 422 naming it, so a leftover `width` from the previous model
        would turn switching models into an error instead of a new form.
        """
        fields = await self._field_names(model)
        out: dict[str, Any] = {}

        for key, value in kw.items():
            if value is None or value == '':
                continue
            if fields and key not in fields:
                continue
            # A reference image arrives as bytes and leaves as a data URI,
            # which every file input on Replicate accepts.
            out[key] = data_uri(value) if isinstance(value, bytes | bytearray) else value

        if not fields or 'prompt' in fields:
            out['prompt'] = prompt
        # `seed: -1` means "random" in this build's vocabulary and means
        # "the literal seed minus one" to a model that takes an integer.
        if out.get('seed') in (-1, '-1'):
            out.pop('seed')
        return out

    async def check(self, job: Submitted) -> Poll:
        body = await self.get_json(job.poll_url)
        status = str(body.get('status', '')).lower()
        if status in ('succeeded',):
            return Poll(state='done', outputs=urls_in(body.get('output')), progress=1.0)
        if status in ('failed', 'canceled'):
            return Poll(state='failed', error=str(body.get('error') or f'the prediction was {status}'))

        # Replicate has no progress field. Several models print one into the
        # logs as a tqdm bar, and reading the last percentage out of that is
        # honest — it came from the model — where inventing a curve would not.
        fraction, note = _from_logs(str(body.get('logs') or ''))
        return Poll(
            state='queued' if status == 'starting' else 'running',
            progress=fraction,
            note=note or ('queued at replicate' if status == 'starting' else 'generating'),
        )

    # -- generation, for both modalities ------------------------------------

    async def generate(
        self, prompt: str, *, model: str = '', n: int = 1, size: str = '', progress: Any = None, **kw: Any
    ) -> list[GeneratedMedia]:
        """One entry point for images and video.

        Replicate does not distinguish them — a prediction is a prediction and
        what comes back is whatever the model makes — so neither does this,
        and the `kind` is only ever used to decide which collection to list.
        """
        fields = await self._field_names(model)
        if n and n > 1 and 'num_outputs' in fields and 'num_outputs' not in kw:
            kw['num_outputs'] = n
        if size and '_' not in size and 'width' in fields and 'height' in fields:
            if 'width' not in kw and 'height' not in kw and 'x' in size:
                try:
                    kw['width'], kw['height'] = (int(v) for v in size.lower().split('x', 1))
                except ValueError:
                    pass
        return await self.run(prompt, model=model, progress=progress, **kw)

    # -- what it can do -----------------------------------------------------

    async def models(self, kind: str = '') -> list[dict[str, Any]]:
        """The models Replicate itself files under this kind of generation.

        Its collections rather than a list kept here, for the reason this
        whole file exists: a hard-coded list of model names is wrong within a
        month, and the person who wants the one that came out yesterday can
        type `owner/name` into the box regardless.
        """
        wanted = COLLECTIONS.get(kind or 'image', COLLECTIONS['image'])
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for slug in wanted:
            try:
                body = await self.get_json(f'{self.base_url}/collections/{slug}', seconds=30)
            except Exception as exc:  # noqa: BLE001 - a collection that has been renamed is not fatal
                log.info('replicate: collection %s did not answer: %s', slug, exc)
                continue
            for model in body.get('models', []):
                ident = f'{model.get("owner")}/{model.get("name")}'
                if ident in seen or '/' not in ident:
                    continue
                seen.add(ident)
                out.append({
                    'id': ident,
                    'provider': self.provider_id,
                    'label': ident,
                    'note': (model.get('description') or '')[:200],
                })
        return out

    async def _field_names(self, model: str) -> set[str]:
        if model not in self._fields:
            await self.describe(model)
        return self._fields.get(model, set())

    async def describe(self, model: str = '') -> list[Param]:
        """The model's own input schema, rendered as controls.

        Cached for a few minutes. A schema does change — a model author can
        publish a new version — but not between two clicks, and re-fetching it
        on every keystroke in the model box would be a request per character.
        """
        if not model:
            return [Param('prompt', 'Prompt', 'text', group='prompt',
                          help='Pick a model first — the controls come from the model itself.')]

        cached = self._schemas.get(model)
        if cached and time.monotonic() - cached[0] < 300:
            return cached[1]

        name, version = self._split(model)
        try:
            if version and '/' not in version:
                body = await self.get_json(f'{self.base_url}/versions/{version}', seconds=30)
                schema = (body.get('openapi_schema') or {})
            else:
                body = await self.get_json(f'{self.base_url}/models/{name}', seconds=30)
                schema = ((body.get('latest_version') or {}).get('openapi_schema') or {})
        except Exception as exc:  # noqa: BLE001
            log.info('replicate: no schema for %s: %s', model, exc)
            return [Param('prompt', 'Prompt', 'text', group='prompt',
                          help=f'Replicate did not return a schema for {model}, so only the prompt is offered.')]

        params = params_from_openapi(schema)
        self._schemas[model] = (time.monotonic(), params)
        self._fields[model] = {p.name for p in params}
        return params


def _from_logs(logs: str) -> tuple[float | None, str]:
    """A percentage out of a model's own tqdm output, if it printed one.

    Deliberately narrow: it reads what the model said and reports nothing when
    the model said nothing, rather than filling the gap with a timer dressed up
    as progress.
    """
    if not logs:
        return None, ''
    import re

    matches = re.findall(r'(\d{1,3})%\|', logs)
    if not matches:
        return None, ''
    percent = min(100, int(matches[-1]))
    return percent / 100, f'{percent}% through the model’s own steps'
