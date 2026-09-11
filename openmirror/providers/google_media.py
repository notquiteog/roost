"""Imagen, Gemini's image model, and Veo — all on one key.

Kept separate from `openmirror.providers.google`, which is deliberately embeddings
only, because these are three different request shapes that happen to share a
host and a credential:

* **Imagen** is `:predict` — a batch of images, returned base64, synchronously.
* **Gemini image** is `:generateContent` — a conversation that happens to
  return a picture, which is what makes it the one to use for *editing*: you
  hand it an image and say what to change, in words, and it answers with the
  changed image. Nothing else here does iterative editing in a sentence.
* **Veo** is `:predictLongRunning` — a job, an operation name, and a poll.

The model list is asked of Google rather than written here, and filtered by
the generation method each model declares. That is the difference between a
picker that grows a new Imagen version by itself and one that has to be
edited: `supportedGenerationMethods` is exactly the fact needed to decide
whether something belongs in the image list or the video list, and it comes
from the service.

The key goes in `x-goog-api-key`, never in the query string — same as the
embeddings adapter, and for the same reason: a credential in a URL is a
credential in every access log between here and there. Veo's result URI is on
Google's own host and needs the key to download, which is the one case where
the header has to be sent to something that is not the API endpoint itself.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from openmirror.media.params import Param, common_image
from openmirror.providers.base import GeneratedMedia
from openmirror.providers.hosted_media import HostedMedia, Poll, Submitted, sniff

log = logging.getLogger(__name__)

API = 'https://generativelanguage.googleapis.com/v1beta'

#: Aspect ratios Imagen and Veo both accept. Gemini's image model takes a
#: longer list, which is handled where it is asked for.
RATIOS = ('1:1', '3:4', '4:3', '9:16', '16:9')


class GoogleMediaProvider(HostedMedia):
    provider_id = 'google'

    def __init__(self, base_url: str = API, api_key: str = '', *, kind: str = 'image', **kw: Any) -> None:
        super().__init__(base_url or API, api_key, **kw)
        self.kind = kind

    def headers(self) -> dict[str, str]:
        h = {'Content-Type': 'application/json'}
        if self.api_key:
            h['x-goog-api-key'] = self.api_key
        return h

    @staticmethod
    def _bare(model: str) -> str:
        return (model or '').removeprefix('models/')

    # -- Veo, which is a job ------------------------------------------------

    async def submit(self, prompt: str, *, model: str, **kw: Any) -> Submitted:
        model = self._bare(model) or 'veo-3.0-generate-001'
        instance: dict[str, Any] = {'prompt': prompt}

        image = kw.get('image')
        if isinstance(image, bytes | bytearray):
            instance['image'] = {
                'bytesBase64Encoded': base64.b64encode(bytes(image)).decode(),
                'mimeType': sniff(bytes(image)),
            }

        parameters: dict[str, Any] = {}
        for key, target in (('aspect_ratio', 'aspectRatio'), ('duration', 'durationSeconds'),
                            ('negative_prompt', 'negativePrompt'), ('resolution', 'resolution'),
                            ('person_generation', 'personGeneration'), ('generate_audio', 'generateAudio')):
            value = kw.get(key)
            if value not in (None, ''):
                parameters[target] = int(value) if target == 'durationSeconds' else value

        body = await self.post_json(
            f'{self.base_url}/models/{model}:predictLongRunning',
            {'instances': [instance], 'parameters': parameters},
        )
        name = str(body.get('name', ''))
        if not name:
            raise RuntimeError(f'google: Veo accepted nothing it would name: {body}')
        return Submitted(id=name, poll_url=f'{self.base_url}/{name}')

    async def check(self, job: Submitted) -> Poll:
        body = await self.get_json(job.poll_url)
        if not body.get('done'):
            # A long-running operation reports no progress at all, so the only
            # true thing to say is that it is still going.
            return Poll(state='running', note='generating at Google')

        if body.get('error'):
            message = (body['error'] or {}).get('message') or 'the operation failed'
            return Poll(state='failed', error=f'google: {message}')

        response = (body.get('response') or {}).get('generateVideoResponse') or (body.get('response') or {})
        samples = response.get('generatedSamples') or response.get('videos') or []
        urls = [
            (s.get('video') or {}).get('uri') or s.get('uri')
            for s in samples
            if isinstance(s, dict)
        ]
        urls = [u for u in urls if u]

        # Veo can decline individual samples for safety and still report done.
        filtered = response.get('raiMediaFilteredCount') or 0
        if not urls:
            reasons = '; '.join(response.get('raiMediaFilteredReasons') or [])
            return Poll(state='failed',
                        error=f'google: Veo returned no video ({filtered} filtered){": " + reasons if reasons else ""}')
        return Poll(state='done', progress=1.0, outputs=urls)

    # -- Imagen and Gemini, which are not -----------------------------------

    async def generate(self, prompt: str, *, model: str = '', n: int = 1, size: str = '',
                       progress: Any = None, **kw: Any) -> list[GeneratedMedia]:
        if self.kind == 'video':
            return await self.run(prompt, model=model, progress=progress, **kw)

        model = self._bare(model) or 'imagen-4.0-generate-001'
        if progress:
            progress(None, 'generating at Google')
        if 'imagen' in model:
            return await self._imagen(prompt, model, n, kw)
        return await self._gemini(prompt, model, kw)

    async def _imagen(self, prompt: str, model: str, n: int, kw: dict[str, Any]) -> list[GeneratedMedia]:
        parameters: dict[str, Any] = {'sampleCount': max(1, min(4, int(n or 1)))}
        for key, target in (('aspect_ratio', 'aspectRatio'), ('negative_prompt', 'negativePrompt'),
                            ('person_generation', 'personGeneration'), ('image_size', 'imageSize')):
            if kw.get(key) not in (None, ''):
                parameters[target] = kw[key]

        body = await self.post_json(
            f'{self.base_url}/models/{model}:predict',
            {'instances': [{'prompt': prompt}], 'parameters': parameters},
            seconds=self.timeout,
        )
        predictions = body.get('predictions') or []
        if not predictions:
            raise RuntimeError(
                'google: Imagen returned no images. This is usually the safety filter rather than '
                'an error — try personGeneration, or a different prompt.'
            )
        return [
            GeneratedMedia(
                data=base64.b64decode(row['bytesBase64Encoded']),
                media_type=row.get('mimeType') or 'image/png',
            )
            for row in predictions
            if row.get('bytesBase64Encoded')
        ]

    async def _gemini(self, prompt: str, model: str, kw: dict[str, Any]) -> list[GeneratedMedia]:
        """The conversational one, which is the one that edits.

        An image handed in alongside the text is the whole mechanism: "make the
        jacket red" against a picture is a complete request here, and does not
        need a mask, a strength or an inpainting mode.
        """
        parts: list[dict[str, Any]] = [{'text': prompt}]
        images = kw.get('image')
        for blob in (images if isinstance(images, list) else [images]):
            if isinstance(blob, bytes | bytearray):
                parts.append({'inline_data': {
                    'mime_type': sniff(bytes(blob)),
                    'data': base64.b64encode(bytes(blob)).decode(),
                }})

        config: dict[str, Any] = {'responseModalities': ['IMAGE']}
        if kw.get('aspect_ratio'):
            config['imageConfig'] = {'aspectRatio': kw['aspect_ratio']}

        body = await self.post_json(
            f'{self.base_url}/models/{model}:generateContent',
            {'contents': [{'role': 'user', 'parts': parts}], 'generationConfig': config},
            seconds=self.timeout,
        )

        out: list[GeneratedMedia] = []
        notes: list[str] = []
        for candidate in body.get('candidates') or []:
            for part in ((candidate.get('content') or {}).get('parts') or []):
                inline = part.get('inlineData') or part.get('inline_data')
                if inline and inline.get('data'):
                    out.append(GeneratedMedia(
                        data=base64.b64decode(inline['data']),
                        media_type=inline.get('mimeType') or inline.get('mime_type') or 'image/png',
                    ))
                elif part.get('text'):
                    notes.append(part['text'])
        if not out:
            said = ' It said: ' + ' '.join(notes)[:300] if notes else ''
            raise RuntimeError(f'google: {model} returned no image.{said}')
        return out

    # -- what it can do -----------------------------------------------------

    async def models(self, kind: str = '') -> list[dict[str, Any]]:
        """Asked of Google and filtered by what each model declares it can do.

        `supportedGenerationMethods` is exactly the fact needed here, which is
        why this is a live list rather than a table: a new Imagen version
        appears in the picker without anyone editing this file.
        """
        kind = kind or self.kind
        try:
            body = await self.get_json(f'{self.base_url}/models?pageSize=200', seconds=30)
        except Exception as exc:  # noqa: BLE001
            log.info('google: model listing failed (%s); offering the documented ids', exc)
            return _fallback(kind, self.provider_id)

        out: list[dict[str, Any]] = []
        for model in body.get('models', []):
            name = self._bare(str(model.get('name', '')))
            methods = model.get('supportedGenerationMethods') or []
            if not name:
                continue
            if kind == 'video':
                if 'predictLongRunning' not in methods:
                    continue
            elif 'predict' in methods and 'imagen' in name:
                pass
            elif 'generateContent' in methods and 'image' in name:
                pass
            else:
                continue
            out.append({
                'id': name,
                'provider': self.provider_id,
                'label': model.get('displayName') or name,
                'note': (model.get('description') or '')[:200],
            })
        return out or _fallback(kind, self.provider_id)

    async def describe(self, model: str = '') -> list[Param]:
        model = self._bare(model)

        if self.kind == 'video':
            return [
                Param('prompt', 'Prompt', 'text', group='prompt',
                      help='Veo 3 generates its own audio from this too — dialogue in quotes is '
                           'spoken, and described sound is produced.'),
                Param('negative_prompt', 'Negative prompt', 'text', group='prompt', advanced=True),
                Param('image', 'Starting frame', 'image', group='prompt',
                      help='Optional. With one, this becomes image-to-video.'),
                Param('aspect_ratio', 'Aspect ratio', 'enum', default='16:9', options=['16:9', '9:16'],
                      group='shape'),
                Param('resolution', 'Resolution', 'enum', default='1080p', options=['720p', '1080p'],
                      group='shape'),
                Param('duration', 'Duration', 'int', default=8, minimum=4, maximum=8, step=1, group='motion',
                      help='Seconds. Veo 3 makes eight.'),
                Param('generate_audio', 'Generate audio', 'bool', default=True, group='motion',
                      help='Veo 3 only, and the reason to choose it — it is the one model here that '
                           'makes sound.'),
                Param('person_generation', 'People', 'enum', default='allow_all',
                      options=['allow_all', 'allow_adult', 'dont_allow'], group='output', advanced=True,
                      help='Google refuses some of these by region regardless of what is set.'),
            ]

        if 'imagen' in model or not model:
            return [p for p in common_image() if p.name != 'seed'] + [
                Param('aspect_ratio', 'Aspect ratio', 'enum', default='1:1', options=list(RATIOS),
                      group='shape'),
                Param('image_size', 'Resolution', 'enum', default='2K', options=['1K', '2K'],
                      group='shape', advanced=True),
                Param('person_generation', 'People', 'enum', default='allow_adult',
                      options=['allow_all', 'allow_adult', 'dont_allow'], group='output', advanced=True),
            ]

        # The conversational image model.
        return [p for p in common_image() if p.name not in ('seed', 'negative_prompt')] + [
            Param('image', 'Images to work from', 'image', group='prompt',
                  help='Give it a picture and say what to change — "make the jacket red". Several can '
                       'be combined: two products and a request to put one in the other\'s scene. '
                       'This is the model to use for editing rather than for making something new.'),
            Param('aspect_ratio', 'Aspect ratio', 'enum', default='1:1',
                  options=['1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'],
                  group='shape'),
        ]


def _fallback(kind: str, provider_id: str) -> list[dict[str, Any]]:
    """Documented ids, for when the listing endpoint cannot be reached.

    Marked as such in the note, following the same rule the catalogue keeps:
    a list that did not come from the service says so, because an id it does
    not know fails at first use rather than at the pick.
    """
    rows = (
        (('veo-3.0-generate-001', 'Veo 3 — video with audio.'),
         ('veo-3.0-fast-generate-001', 'Veo 3 Fast.'),
         ('veo-2.0-generate-001', 'Veo 2.'))
        if kind == 'video' else
        (('imagen-4.0-generate-001', 'Imagen 4.'),
         ('imagen-4.0-ultra-generate-001', 'Imagen 4 Ultra — best prompt adherence, one image at a time.'),
         ('imagen-4.0-fast-generate-001', 'Imagen 4 Fast.'),
         ('gemini-2.5-flash-image', 'Gemini 2.5 Flash Image — conversational editing.'))
    )
    return [
        {'id': i, 'provider': provider_id, 'label': i, 'note': f'{n} (from documentation, not from the service)'}
        for i, n in rows
    ]
