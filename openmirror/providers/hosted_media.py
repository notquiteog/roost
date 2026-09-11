"""The shape every hosted image and video API turns out to have.

Replicate, fal, Black Forest Labs, Runway, Luma, Kling, MiniMax, Stability's
video path, Google's Veo and OpenAI's Sora are ten different companies and one
protocol: **post a job, poll until it stops moving, then fetch a URL.** They
disagree about every noun — `id` or `request_id` or `task_id`, `status` or
`state`, `succeeded` or `SUCCEEDED` or `Ready` — and about nothing structural.

Writing that loop ten times would be ten chances to get the same three things
subtly wrong, and they are exactly the three that are invisible until they
matter:

**Cancellation has to reach the other end.** A job cancelled here is still
running there, and on a hosted API it is still *billing* there. So the
cancellation path is one implementation that calls the provider's own cancel
endpoint, and a provider that has none says so by not overriding `cancel`
rather than by silently doing nothing.

**A failure has to carry the provider's own words.** "Generation failed" is
not an error message. `NSFW content detected`, `insufficient credits` and
`invalid aspect ratio for this model` want three different reactions from the
person waiting, and flattening them into one string loses the only useful part.

**Progress must not be invented.** Some of these report a percentage, some
report a queue position, and most report neither. The honest answers are a
fraction, a position, and elapsed seconds — never a synthetic bar that reaches
90% and stops, which is a lie that teaches people to distrust the real ones.

What is deliberately *not* here is anything about what the models do. Sizes,
durations, aspect ratios and samplers belong to each service, because the one
thing worse than ten polling loops is one abstraction that has to be widened
every time a service adds a knob.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from openmirror.net.transport import Transport
from openmirror.providers.base import GeneratedMedia

log = logging.getLogger(__name__)

#: What the poll loop waits between asks, and the ceiling it climbs to. Short
#: at the start because a fast image job is often done on the second ask, and
#: longer later because a four-minute video polled every second is two hundred
#: requests that tell you nothing.
POLL_START = 1.0
POLL_MAX = 5.0

#: Sniffed from the first bytes rather than trusted from the header, because
#: several of these services return `application/octet-stream` for everything
#: and one returns `binary/octet-stream`, and a file stored as `.bin` is a file
#: no gallery will show.
MAGIC: tuple[tuple[bytes, str], ...] = (
    (b'\x89PNG\r\n\x1a\n', 'image/png'),
    (b'\xff\xd8\xff', 'image/jpeg'),
    (b'GIF87a', 'image/gif'),
    (b'GIF89a', 'image/gif'),
)


def sniff(data: bytes, declared: str = '', url: str = '') -> str:
    """What this actually is, preferring evidence over assertion."""
    for magic, media_type in MAGIC:
        if data.startswith(magic):
            return media_type
    if data[4:12] in (b'ftypmp42', b'ftypisom', b'ftypmp41', b'ftypavc1', b'ftypM4V ') or data[4:8] == b'ftyp':
        return 'video/mp4'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    if data[:4] == b'\x1a\x45\xdf\xa3':
        return 'video/webm'
    # Nothing recognised: believe the header, then the extension, then guess
    # by the only distinction that matters downstream.
    declared = (declared or '').split(';')[0].strip().lower()
    if declared.startswith(('image/', 'video/')):
        return declared
    tail = url.split('?', 1)[0].rsplit('.', 1)[-1].lower()
    by_extension = {
        'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'webp': 'image/webp',
        'gif': 'image/gif', 'mp4': 'video/mp4', 'webm': 'video/webm', 'mov': 'video/mp4',
    }
    return by_extension.get(tail, 'application/octet-stream')


@dataclass(slots=True)
class Submitted:
    """A job the other end has accepted."""

    id: str
    #: Where to ask about it. Several services hand back a URL rather than
    #: expecting one to be built from the id, and using theirs matters: fal's
    #: status URL carries the region the job actually landed in.
    poll_url: str = ''
    #: Where the finished thing is, when that is a different URL from the
    #: status one. Empty means the poll response carries it.
    result_url: str = ''
    cancel_url: str = ''
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Poll:
    """One answer to "is it done yet".

    `outputs` are URLs; `inline` is bytes already in hand, which a few services
    return directly in the status body. Both are supported because converting
    one into the other costs a round trip in the direction nobody wants.
    """

    state: str = 'running'            # queued | running | done | failed
    progress: float | None = None     # 0..1, and None when the service does not say
    note: str = ''                    # queue position, current stage, anything true
    outputs: list[str] = field(default_factory=list)
    inline: list[tuple[bytes, str]] = field(default_factory=list)
    error: str = ''
    seed: int | None = None

    @property
    def finished(self) -> bool:
        return self.state in ('done', 'failed')


class HostedMedia(abc.ABC):
    """A media API that takes jobs.

    Subclasses implement three methods and get the loop, the cancellation, the
    downloading and the progress reporting for free.
    """

    #: Shown in errors, so it must be the id the person picked in the UI.
    provider_id: str = 'hosted'

    def __init__(
        self,
        base_url: str,
        api_key: str = '',
        *,
        provider_id: str = '',
        timeout: int = 1800,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        if provider_id:
            self.provider_id = provider_id
        self.timeout = timeout
        self.transport = transport or Transport(timeout=timeout)

    # -- what a subclass must say -------------------------------------------

    def headers(self) -> dict[str, str]:
        """Auth, the way this service spells it."""
        h = {'Content-Type': 'application/json'}
        if self.api_key:
            h['Authorization'] = f'Bearer {self.api_key}'
        return h

    @abc.abstractmethod
    async def submit(self, prompt: str, *, model: str, **kw: Any) -> Submitted: ...

    @abc.abstractmethod
    async def check(self, job: Submitted) -> Poll: ...

    async def cancel(self, job: Submitted) -> None:
        """Stop it at the other end.

        Not abstract: a service with no cancel endpoint should inherit this
        and do nothing, rather than every caller writing a `getattr` dance.
        The default follows `cancel_url` when the submit response gave one,
        which covers Replicate and fal without either of them overriding.
        """
        if not job.cancel_url:
            return
        try:
            async with self.transport.session(30) as session:
                async with session.post(job.cancel_url, headers=self.headers()) as resp:
                    log.info('%s: cancelled %s upstream (HTTP %s)', self.provider_id, job.id, resp.status)
        except Exception as exc:  # noqa: BLE001 - best effort by definition
            log.warning('%s: could not cancel %s upstream: %s', self.provider_id, job.id, exc)

    # -- the loop everything shares -----------------------------------------

    async def run(
        self,
        prompt: str,
        *,
        model: str,
        progress: Any = None,
        **kw: Any,
    ) -> list[GeneratedMedia]:
        """Submit, wait, download. The whole of what a provider does.

        `progress` is called as `progress(fraction | None, note)` and is
        allowed to be None. It is called on every poll even when the fraction
        has not moved, because "still queued, position 4" changing to "still
        queued, position 2" is the useful signal on a busy service and a
        caller that only watches the number would miss it.
        """
        job = await self.submit(prompt, model=model, **kw)
        if progress:
            progress(None, 'queued')

        deadline = time.monotonic() + self.timeout
        wait = POLL_START
        try:
            while True:
                if time.monotonic() > deadline:
                    await self.cancel(job)
                    raise TimeoutError(
                        f'{self.provider_id}: gave up after {self.timeout}s. The job may still be '
                        f'running at the provider — its id is {job.id}.'
                    )
                await asyncio.sleep(wait)
                wait = min(wait * 1.35, POLL_MAX)

                state = await self.check(job)
                if progress and not state.finished:
                    progress(state.progress, state.note or state.state)
                if state.finished:
                    break

            if state.state == 'failed':
                raise RuntimeError(state.error or f'{self.provider_id}: the job failed without saying why')

            if progress:
                progress(1.0, 'downloading')
            return await self.collect(state)
        except asyncio.CancelledError:
            # The person pressed stop. The job is still running — and on a
            # hosted service still billing — until the provider is told.
            await self.cancel(job)
            raise

    async def collect(self, state: Poll) -> list[GeneratedMedia]:
        """Turn a finished poll into bytes.

        Downloads happen here rather than in each subclass so that every
        provider hands the store the same thing: a caller should never have to
        know that one service returns base64 and another a signed URL that
        expires in an hour.
        """
        out = [
            GeneratedMedia(data=data, media_type=media_type, seed=state.seed)
            for data, media_type in state.inline
        ]
        for url in state.outputs:
            data, media_type = await self.download(url)
            out.append(GeneratedMedia(data=data, media_type=media_type, seed=state.seed))
        return out

    async def download(self, url: str) -> tuple[bytes, str]:
        """Fetch one result.

        Sent without this provider's auth header unless the URL is on its own
        API host. Most of these are signed storage URLs on a CDN, and sending
        a bearer token to a third-party bucket is how a key ends up in
        somebody else's access log.
        """
        headers = self.headers() if url.startswith(self.base_url) else {}
        headers.pop('Content-Type', None)
        async with self.transport.session(300) as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    raise RuntimeError(
                        f'{self.provider_id}: could not download the result: HTTP {resp.status}'
                    )
                declared = resp.headers.get('Content-Type', '')
                data = await resp.read()
        return data, sniff(data, declared, url)

    # -- small shared helpers -----------------------------------------------

    async def post_json(self, url: str, payload: dict[str, Any], *, seconds: int = 120) -> dict[str, Any]:
        # Named `seconds` rather than `timeout` because the deadline is the
        # transport's to keep, not this coroutine's — see openmirror.net.transport.
        async with self.transport.session(seconds) as session:
            async with session.post(url, json=payload, headers=self.headers()) as resp:
                body = await resp.text()
                if resp.status >= 300:
                    raise RuntimeError(f'{self.provider_id}: HTTP {resp.status}: {body[:400]}')
                return _as_json(body)

    async def get_json(self, url: str, *, seconds: int = 60) -> dict[str, Any]:
        async with self.transport.session(seconds) as session:
            async with session.get(url, headers=self.headers()) as resp:
                body = await resp.text()
                if resp.status >= 300:
                    raise RuntimeError(f'{self.provider_id}: HTTP {resp.status}: {body[:400]}')
                return _as_json(body)


def _as_json(body: str) -> dict[str, Any]:
    import json

    try:
        parsed = json.loads(body or '{}')
    except ValueError as exc:
        raise RuntimeError(f'the service answered with something that is not JSON: {body[:200]}') from exc
    return parsed if isinstance(parsed, dict) else {'result': parsed}


def data_uri(data: bytes, media_type: str = '') -> str:
    """A reference image as something a JSON API will take.

    Every one of these services accepts a data URI for a file input, and the
    alternative — uploading to a bucket first so there is a URL to hand over —
    means this build would need somewhere public to put a private picture.
    A few megabytes inline is worth not having that.
    """
    import base64

    return f'data:{media_type or sniff(data)};base64,{base64.b64encode(data).decode()}'


def urls_in(value: Any, *, depth: int = 0) -> list[str]:
    """Every media URL in a response, wherever the service decided to put it.

    These APIs return outputs as a string, a list of strings, a list of
    objects with a `url`, an object with `images`, an object with `video.url`,
    or a `sample` — and several of them return a *different* one of those
    depending on the model. Walking the structure is shorter than nine
    special cases, and it fails in the survivable direction: an output shape
    nobody anticipated still yields its URLs.
    """
    if depth > 6:
        return []
    if isinstance(value, str):
        return [value] if value.startswith(('http://', 'https://')) else []
    if isinstance(value, list):
        return [u for item in value for u in urls_in(item, depth=depth + 1)]
    if isinstance(value, dict):
        # Keys that are known to hold the thing itself, tried first so that a
        # response carrying both a result and a thumbnail yields the result
        # first and the gallery does not show a 128px preview as the output.
        out: list[str] = []
        for key in ('url', 'video_url', 'image_url', 'sample', 'signed_url', 'download_url'):
            if isinstance(value.get(key), str):
                out += urls_in(value[key], depth=depth + 1)
        for key, item in value.items():
            if key in ('url', 'video_url', 'image_url', 'sample', 'signed_url', 'download_url'):
                continue
            # Skip the ones that are reliably not the output, so a preview or
            # a poster frame is never mistaken for the result.
            if key in ('thumbnail_url', 'preview_url', 'poster', 'thumb', 'logs', 'cancel'):
                continue
            out += urls_in(item, depth=depth + 1)
        return out
    return []
