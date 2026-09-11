"""OpenRouter: one key, six kinds of request.

Chat, embeddings and speech in both directions are OpenAI's shapes and go
through `openai_compat` with nothing changed but the base URL. Images and video
are OpenRouter's own: `/images` sizes a picture by aspect ratio and resolution
tier rather than in pixels, and `/videos` is a submit-poll-download job like
every other hosted video API, so it is built on `hosted_media`.

Three things set it apart from an ordinary OpenAI-shaped gateway, and each was
found against the live API rather than read in its documentation:

**Where the models are listed depends on what they are for.** `/models` is the
chat catalogue and nothing else. Embedding models are at `/embeddings/models`,
the speech models are the ones whose output is `speech` or `transcription`,
and images and video each have a listing of their own. One list for all six
would put four hundred chat models in the dictation picker, so each modality
gets an instance that knows which list to ask for.

**Listing models proves nothing about the key.** `/models` answers a wrong key,
and no key at all, with the full catalogue — so a Test button that only listed
would pass a connection whose every real request fails with a 401. `check`
asks `/key`, which does refuse.

**Speech comes back at whatever rate the voice was made at.** The rate is in
the Content-Type — `audio/pcm;rate=24000` from Kokoro and Deepgram, `44100`
from Fish Audio — while the voice call is told one rate when it opens. Audio
at any other rate is resampled here rather than played nearly twice as slow.
"""

from __future__ import annotations

import array
import base64
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from openmirror.media.params import Param
from openmirror.net.transport import Transport
from openmirror.providers.base import GeneratedMedia, ImageProvider, Modality, refused
from openmirror.providers.hosted_media import HostedMedia, Poll, Submitted, data_uri, sniff
from openmirror.providers.openai_compat import OpenAICompatProvider

log = logging.getLogger(__name__)

API = 'https://openrouter.ai/api/v1'

#: Where each modality's models are listed. See the module docstring.
LISTINGS: dict[Modality, str] = {
    Modality.CHAT: '/models',
    Modality.EMBEDDING: '/embeddings/models',
    Modality.STT: '/models?output_modalities=transcription',
    Modality.TTS: '/models?output_modalities=speech',
    Modality.IMAGE: '/images/models',
    Modality.VIDEO: '/videos/models',
}

#: Resolution tiers, cheapest first. A form defaults to the lowest a model
#: offers, because every one of these is billed and the lowest is the one
#: nobody is surprised by.
TIERS = ('480p', '512', '720p', '768p', '1k', '1080p', '2k', '4k')


async def _listing(
    transport: Transport, base_url: str, api_key: str, modality: Modality, provider_id: str
) -> list[dict[str, Any]]:
    """The rows of one of OpenRouter's model listings, as it sent them."""
    headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
    async with transport.session(60) as session:
        async with session.get(f'{base_url}{LISTINGS[modality]}', headers=headers) as resp:
            if resp.status in (401, 403):
                raise refused(provider_id, base_url, resp.status)
            if resp.status != 200:
                return []
            body = await resp.json()
    return [row for row in body.get('data') or [] if isinstance(row, dict) and row.get('id')]


async def _find(rows: Callable[[], Awaitable[list[dict[str, Any]]]], model: str) -> dict[str, Any] | None:
    """One model's row, or None — including when the listing cannot be reached.

    None rather than an error because a form is still worth drawing offline: a
    prompt box and the common controls, instead of an empty panel that only
    fills in once the network is back.
    """
    if not model:
        return None
    try:
        found = await rows()
    except Exception as exc:  # noqa: BLE001 - an offline form beats no form
        log.info('openrouter: could not look up %s: %s', model, exc)
        return None
    return next((row for row in found if row.get('id') == model), None)


def _model_row(row: dict[str, Any], provider_id: str) -> dict[str, Any]:
    description = str(row.get('description') or '')
    return {
        'id': row['id'],
        'provider': provider_id,
        'label': row.get('name') or row['id'],
        # The first sentence: the datalist note is one line, and the rest of
        # these descriptions is marketing.
        'note': description.split('. ', 1)[0][:160],
    }


def _lowest(values: list[str]) -> str:
    rank = {tier: index for index, tier in enumerate(TIERS)}
    return min(values, key=lambda value: rank.get(str(value).lower(), len(rank)))


def _blobs(value: Any) -> list[bytes]:
    items = value if isinstance(value, list) else [value]
    return [bytes(item) for item in items if isinstance(item, bytes | bytearray)]


def _image_url(blob: bytes) -> dict[str, Any]:
    return {'type': 'image_url', 'image_url': {'url': data_uri(blob)}}


def _seed(value: Any) -> int | None:
    """A seed to send, or None for "pick one" — which -1 means, and is not a seed."""
    if value in (None, '', -1, '-1'):
        return None
    return int(value)


def _declared_rate(content_type: str) -> int | None:
    match = re.search(r'rate=(\d+)', content_type)
    return int(match.group(1)) if match else None


def _resample(pcm: bytes, source: int, target: int) -> bytes:
    """16-bit mono PCM from one rate to another, by linear interpolation.

    Not what a studio would use, and it does not need to be: this is speech on
    its way to a laptop speaker, every rate involved is well above what a voice
    needs, and the alternative is a dependency.
    """
    samples = array.array('h')
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    if not samples or source == target:
        return samples.tobytes()
    count = int(len(samples) * target / source)
    step = source / target
    last = len(samples) - 1
    out = array.array('h', bytes(2 * count))
    for index in range(count):
        position = index * step
        left = int(position)
        a = samples[left]
        b = samples[left + 1] if left < last else a
        out[index] = int(a + (b - a) * (position - left))
    return out.tobytes()


# ---------------------------------------------------------------------------
# Chat, embeddings, dictation and speech
# ---------------------------------------------------------------------------


class OpenRouterProvider(OpenAICompatProvider):
    """OpenAI's shapes, with OpenRouter's lists. One instance per modality."""

    def __init__(self, base_url: str = API, api_key: str = '', *, kind: Modality = Modality.CHAT, **kw: Any) -> None:
        super().__init__(base_url or API, api_key, **kw)
        self.kind = kind

    async def models(self) -> list[dict[str, Any]]:
        rows = await _listing(self.transport, self.base_url, self.api_key, self.kind, self.provider_id)
        return [{'id': row['id'], 'provider': self.provider_id} for row in rows]

    async def check(self) -> None:
        """Prove the key, which listing models cannot — see the module docstring."""
        async with self._session(30) as session:
            async with session.get(f'{self.base_url}/key', headers=self._headers()) as resp:
                if resp.status in (401, 403):
                    raise refused(self.provider_id, self.base_url, resp.status)

    async def voices(self) -> list[dict[str, Any]]:
        """Every voice, with the model it belongs to.

        Voices are per model here — `af_heart` is Kokoro's, `aura-2-thalia-en`
        Deepgram's — and asking one model for another's voice is refused, so a
        voice is never listed without its model.
        """
        rows = await _listing(self.transport, self.base_url, self.api_key, Modality.TTS, self.provider_id)
        return [{'id': voice, 'model': row['id']} for row in rows for voice in row.get('supported_voices') or []]

    async def synthesize(
        self, text: str, *, model: str, voice: str, fmt: str = 'pcm16', sample_rate: int = 16_000
    ) -> AsyncIterator[bytes]:
        payload: dict[str, Any] = {'model': model, 'input': text, 'response_format': 'pcm' if fmt == 'pcm16' else fmt}
        # Left out when blank, which OpenRouter accepts for the models that
        # have a default voice and refuses, by name, for the rest.
        if voice:
            payload['voice'] = voice
        async with self._session() as session:
            async with session.post(f'{self.base_url}/audio/speech', json=payload, headers=self._headers()) as resp:
                if resp.status != 200:
                    raise RuntimeError(f'{self.provider_id} tts: HTTP {resp.status}: {(await resp.text())[:300]}')
                rate = _declared_rate(resp.headers.get('Content-Type', ''))
                if fmt != 'pcm16' or rate in (None, self.output_sample_rate):
                    # Streamed as it arrives: closing this iterator is how a
                    # barge-in stops a synthesis already in flight.
                    async for chunk in resp.content.iter_chunked(4096):
                        yield chunk
                    return
                # Another rate. Resampled whole rather than chunk by chunk: it
                # is one sentence, and an interpolator restarted at every chunk
                # boundary clicks at every one of them.
                pcm = await resp.read()
        yield _resample(pcm, rate, self.output_sample_rate)


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

#: The controls `/images` takes, as (label, group, what it does). Offered per
#: model from what that model's listing says it accepts — none of them is on
#: every model, and a control that does nothing teaches people to set it.
IMAGE_CONTROLS: dict[str, tuple[str, str, str]] = {
    'aspect_ratio': ('Aspect ratio', 'shape', ''),
    'resolution': ('Resolution', 'shape',
                   'A tier rather than pixels; the provider picks the exact size. Higher costs more on most models.'),
    'quality': ('Quality', 'quality', 'Higher costs more and takes longer.'),
    'background': ('Background', 'output', 'Transparent needs png or webp output.'),
    'output_format': ('File format', 'output', ''),
    'output_compression': ('Compression', 'output', 'For jpeg and webp only.'),
}


def _image_params(supported: dict[str, Any] | None) -> list[Param]:
    params = [Param('prompt', 'Prompt', 'text', group='prompt',
                    help='What to make. Longer and more specific beats a list of adjectives.')]
    if supported is None:
        # Offline, or a model the listing does not have: the one control every
        # model here takes.
        return params + [Param('aspect_ratio', 'Aspect ratio', 'enum', default='1:1',
                               options=['1:1', '16:9', '9:16', '4:3', '3:4'], group='shape')]

    for name, (label, group, why) in IMAGE_CONTROLS.items():
        spec = supported.get(name)
        if not isinstance(spec, dict):
            continue
        if spec.get('type') == 'enum' and spec.get('values'):
            values = [str(v) for v in spec['values']]
            if name == 'resolution':
                default = _lowest(values)
            elif name == 'aspect_ratio' and '1:1' in values:
                default = '1:1'
            else:
                default = values[0]
            params.append(Param(name, label, 'enum', default=default, options=values, group=group, help=why))
        elif spec.get('type') == 'range':
            params.append(Param(name, label, 'int', minimum=spec.get('min'), maximum=spec.get('max'), step=1,
                                group=group, advanced=True, help=why))

    count = supported.get('n')
    if isinstance(count, dict) and (count.get('max') or 1) > 1:
        params.append(Param('n', 'How many', 'int', default=1, minimum=1, maximum=count['max'], step=1,
                            group='output', help='Images per request. Every one of them costs.'))
    if 'seed' in supported:
        params.append(Param('seed', 'Seed', 'seed', default=-1, group='sampling', advanced=True,
                            help='-1 picks a new one each time. Reusing one with the same prompt and settings '
                                 'reproduces the image, where the provider honours it.'))
    references = supported.get('input_references')
    if isinstance(references, dict) and (references.get('max') or 0) > 0:
        needed = (references.get('min') or 0) > 0
        params.append(Param('image', 'Images to work from', 'image', group='prompt',
                            help=f'Up to {references["max"]}, to edit or to match the look of.'
                                 + (' This model needs at least one.' if needed else '')))
    return params


class OpenRouterImages(ImageProvider):
    """`/images`: a picture from a prompt, sized by ratio and tier."""

    provider_id = 'openrouter'

    def __init__(
        self,
        base_url: str = API,
        api_key: str = '',
        *,
        provider_id: str = '',
        timeout: int = 600,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = (base_url or API).rstrip('/')
        self.api_key = api_key
        if provider_id:
            self.provider_id = provider_id
        self.timeout = timeout
        self.transport = transport or Transport(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        h = {'Content-Type': 'application/json'}
        if self.api_key:
            h['Authorization'] = f'Bearer {self.api_key}'
        return h

    async def _rows(self) -> list[dict[str, Any]]:
        return await _listing(self.transport, self.base_url, self.api_key, Modality.IMAGE, self.provider_id)

    async def models(self, kind: str = '') -> list[dict[str, Any]]:
        if kind == 'video':
            return []
        return [_model_row(row, self.provider_id) for row in await self._rows()]

    async def describe(self, model: str = '') -> list[Param]:
        row = await _find(self._rows, model)
        supported = row.get('supported_parameters') if row else None
        return _image_params(supported if isinstance(supported, dict) else None)

    async def generate(
        self, prompt: str, *, model: str, n: int = 1, size: str = '1024x1024',
        progress: Any = None, **kw: Any
    ) -> list[GeneratedMedia]:
        # `size` is deliberately not sent. The media service hands every image
        # provider one, made up as 1024x1024 when nobody asked, and OpenRouter
        # takes a pixel size as authoritative: sent beside an aspect ratio it
        # is a 400, and sent alone it overrides the ratio the form offered.
        payload: dict[str, Any] = {'model': model, 'prompt': prompt}
        # Only above one: a model that makes one image at a time refuses `n`
        # outright rather than ignoring it.
        if n > 1:
            payload['n'] = n
        for key in IMAGE_CONTROLS:
            value = kw.get(key)
            if value not in (None, '', 'auto'):
                payload[key] = value
        seed = _seed(kw.get('seed'))
        if seed is not None:
            payload['seed'] = seed
        references = _blobs(kw.get('image'))
        if references:
            payload['input_references'] = [_image_url(blob) for blob in references]

        if progress:
            progress(None, f'generating on {self.provider_id}')

        async with self.transport.session(self.timeout) as session:
            async with session.post(f'{self.base_url}/images', json=payload, headers=self._headers()) as resp:
                if resp.status != 200:
                    raise RuntimeError(f'{self.provider_id} image: HTTP {resp.status}: {(await resp.text())[:400]}')
                body = await resp.json()

        out: list[GeneratedMedia] = []
        for row in body.get('data') or []:
            if not row.get('b64_json'):
                continue
            data = base64.b64decode(row['b64_json'])
            out.append(GeneratedMedia(data=data, media_type=row.get('media_type') or sniff(data),
                                      meta={'revised_prompt': row.get('revised_prompt') or ''}))
        return out


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


def _video_params(row: dict[str, Any] | None) -> list[Param]:
    params = [Param('prompt', 'Prompt', 'text', group='prompt',
                    help='What happens, and how the camera sees it. Motion, framing and light matter more '
                         'here than adjectives.')]
    if row is None:
        return params + [Param('first_frame', 'First frame', 'image', group='prompt',
                               help='A picture for the video to start from, on the models that take one.')]

    durations = [str(d) for d in sorted(row.get('supported_durations') or [])]
    if durations:
        params.append(Param('duration', 'Duration (seconds)', 'enum', default=durations[0], options=durations,
                            group='motion', help='Priced by the second on nearly every model, so the shortest '
                                                 'is the default.'))
    resolutions = [str(r) for r in row.get('supported_resolutions') or []]
    if resolutions:
        params.append(Param('resolution', 'Resolution', 'enum', default=_lowest(resolutions), options=resolutions,
                            group='shape', help='Higher costs more and takes longer.'))
    ratios = [str(r) for r in row.get('supported_aspect_ratios') or []]
    if ratios:
        params.append(Param('aspect_ratio', 'Aspect ratio', 'enum', default='16:9' if '16:9' in ratios else ratios[0],
                            options=ratios, group='shape'))
    if row.get('generate_audio'):
        params.append(Param('generate_audio', 'Sound', 'bool', default=False, group='output',
                            help='Sound as well as picture. Off unless asked for, because it costs more on most '
                                 'of the models that can.'))
    if row.get('seed'):
        params.append(Param('seed', 'Seed', 'seed', default=-1, group='sampling', advanced=True,
                            help='-1 picks a new one each time. Not every provider honours a seed exactly.'))
    frames = row.get('supported_frame_images') or []
    if 'first_frame' in frames:
        params.append(Param('first_frame', 'First frame', 'image', group='prompt',
                            help='A picture for the video to start from.'))
    if 'last_frame' in frames:
        params.append(Param('last_frame', 'Last frame', 'image', group='prompt', advanced=True,
                            help='A picture for it to end on.'))
    return params


class OpenRouterVideo(HostedMedia):
    """`/videos`: submit, poll, then fetch the result with the same key."""

    provider_id = 'openrouter'

    def __init__(self, base_url: str = API, api_key: str = '', **kw: Any) -> None:
        super().__init__(base_url or API, api_key, **kw)

    async def _rows(self) -> list[dict[str, Any]]:
        return await _listing(self.transport, self.base_url, self.api_key, Modality.VIDEO, self.provider_id)

    async def submit(self, prompt: str, *, model: str, **kw: Any) -> Submitted:
        payload: dict[str, Any] = {'model': model, 'prompt': prompt}
        for key in ('resolution', 'aspect_ratio'):
            if kw.get(key) not in (None, ''):
                payload[key] = str(kw[key])
        # An integer on the wire, and a closed set per model — the form offers
        # it as a choice, which is why it arrives here as a string.
        if kw.get('duration') not in (None, ''):
            payload['duration'] = int(kw['duration'])
        if kw.get('generate_audio') is not None:
            payload['generate_audio'] = bool(kw['generate_audio'])
        seed = _seed(kw.get('seed'))
        if seed is not None:
            payload['seed'] = seed
        frames = [
            {**_image_url(blobs[0]), 'frame_type': frame}
            for frame in ('first_frame', 'last_frame')
            if (blobs := _blobs(kw.get(frame)))
        ]
        if frames:
            payload['frame_images'] = frames

        body = await self.post_json(f'{self.base_url}/videos', payload)
        ident = str(body.get('id') or '')
        if not ident:
            raise RuntimeError(f'{self.provider_id}: the video was not accepted: {str(body)[:300]}')
        # The poll address is built rather than taken from the response, so the
        # key only ever goes to the host this connection was configured for.
        return Submitted(id=ident, poll_url=f'{self.base_url}/videos/{ident}')

    async def check(self, job: Submitted) -> Poll:
        body = await self.get_json(job.poll_url)
        status = str(body.get('status') or '')
        if status == 'completed':
            # Not presigned: they are on this API and need the key, which
            # `download` sends only because they are.
            urls = [u for u in body.get('unsigned_urls') or [] if isinstance(u, str)]
            return Poll(state='done', progress=1.0,
                        outputs=urls or [f'{self.base_url}/videos/{job.id}/content?index=0'])
        if status in ('failed', 'cancelled', 'expired'):
            return Poll(state='failed', error=f'{self.provider_id}: {body.get("error") or f"the job {status}"}')
        # No percentage anywhere in the response, so none is invented.
        running = status == 'in_progress'
        return Poll(state='running' if running else 'queued',
                    note='generating at OpenRouter' if running else 'queued at OpenRouter')

    async def cancel(self, job: Submitted) -> None:
        """There is no way to cancel a video job on OpenRouter.

        Said here rather than left to the generic implementation, because the
        consequence is real: stopping one in this UI stops the waiting, not
        the job, and a job that finishes is billed.
        """
        log.info('%s: %s was cancelled locally. OpenRouter cannot cancel a video job, so it will run to '
                 'completion and may be charged for.', self.provider_id, job.id)

    async def generate(self, prompt: str, *, model: str = '', n: int = 1, size: str = '',
                       progress: Any = None, **kw: Any) -> list[GeneratedMedia]:
        return await self.run(prompt, model=model, progress=progress, **kw)

    async def models(self, kind: str = '') -> list[dict[str, Any]]:
        if kind == 'image':
            return []
        return [_model_row(row, self.provider_id) for row in await self._rows()]

    async def describe(self, model: str = '') -> list[Param]:
        return _video_params(await _find(self._rows, model))
