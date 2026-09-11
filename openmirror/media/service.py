"""Making an image or a video, whoever is making it.

This is the layer that turns "a prompt and a pile of settings" into a stored
result, and it exists so that three callers — the HTTP API, the agent's tools
and the studio panel — do the same thing rather than three similar things.

The interesting decisions are about *honesty with the person waiting*.

**A request says which provider it went to.** Not for bookkeeping: image
generation is the modality where "local or hosted" is most likely to be a
deliberate choice and least likely to be visible from the result, and a
picture that quietly went to a paid API because the local server was down is
the failure this whole registry design exists to prevent. So the route is
resolved explicitly and the answer carries the provider it used.

**Parameters are validated against what the provider says it has**, not
against a list kept here. Anything the schema does not know is dropped and
reported, so a client that is one version behind is told which of its settings
did nothing rather than silently getting a default.

**Both kinds are jobs.** Video always was — minutes rather than seconds. Images
became one when this build grew past local diffusion: a hosted image model can
sit in a queue for ninety seconds, and an HTTP request held open that long dies
to somebody's proxy, taking the GPU's work with it. So a generation is started,
watched and cancellable whatever it is making, and `generate_image` is kept as
a *synchronous wrapper* over the same path for the agent, whose turn is
easier to reason about when a tool call returns the thing it made.

**Progress is reported, never invented.** Providers say what they honestly
know — a fraction from Runway or Sora, a queue position from fal or ComfyUI,
nothing at all from Veo — and a job carries whichever it got. A bar that
creeps to ninety percent and stops is a lie, and one lie of that kind is
enough to make people stop believing the true ones.

**Cancelling reaches the other end.** A cancelled job is still running at the
provider, and on a hosted one still billing, until the provider is told. So
cancellation is propagated rather than merely stopping the local wait — see
`openmirror.providers.hosted_media`.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from openmirror.media.params import Param, coerce, unknown_keys
from openmirror.media.store import Media, MediaStore
from openmirror.providers.base import Modality
from openmirror.providers.registry import NoProviderError, ProviderRegistry, Route, RouteSet

log = logging.getLogger(__name__)

#: How many finished jobs to keep. They are only ever read by a client that
#: wants to know what happened; the media itself is in the store and outlives
#: all of this. Unbounded, they are a slow leak on a daemon that stays up for
#: weeks.
KEEP_JOBS = 200


@dataclass(slots=True)
class Job:
    """One generation, long enough to be worth watching."""

    id: str
    kind: str
    prompt: str
    provider: str
    model: str
    started_at: float = field(default_factory=time.time)
    state: str = 'queued'           # queued | running | done | failed | cancelled
    #: 0..1, or None when nothing downstream honestly knows. See the module
    #: docstring: None is a real answer here and is rendered as one.
    progress: float | None = None
    #: What is happening, in words. "number 3 in the queue", "rendering on the
    #: GPU", "downloading". Worth more than a number that cannot move.
    note: str = ''
    error: str = ''
    media: list[Media] = field(default_factory=list)
    #: What it was asked with, so a result can be sent back to the form it
    #: came from without the client having kept a copy.
    params: dict[str, Any] = field(default_factory=dict)
    finished_at: float = 0.0
    task: Any = field(default=None, repr=False)

    @property
    def elapsed(self) -> int:
        return int((self.finished_at or time.time()) - self.started_at)

    def to_json(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'kind': self.kind,
            'prompt': self.prompt,
            'provider': self.provider,
            'model': self.model,
            'state': self.state,
            'progress': self.progress,
            'note': self.note,
            'error': self.error,
            'elapsed': self.elapsed,
            'params': self.params,
            'media': [m.to_json() for m in self.media],
        }


class MediaService:
    def __init__(self, store: MediaStore, registry: ProviderRegistry) -> None:
        self.store = store
        self.registry = registry
        self.jobs: dict[str, Job] = {}

    def can_generate(self) -> set[str]:
        """Which kinds of media this install can actually make, right now.

        Asked before the tools are offered. A model shown `generate_video` on
        a machine with no video backend will propose it, be told nothing is
        configured, and spend the turn recovering — the same reason
        `write_file` is left out of a read-only session rather than registered
        and refused. On a small model the cost is worse than a wasted step: a
        tool list full of things that cannot work is a tool list it reasons
        about instead of doing the task.
        """
        available = self.registry.capabilities()
        return {kind for kind in ('image', 'video') if available.get(kind)}

    def providers(self, kind: str = '') -> list[dict[str, Any]]:
        """Who could answer for this kind of generation.

        Per kind rather than one list, because they genuinely differ now: an
        install can have six things that make pictures and two that make
        video, and offering the six in the video picker is offering four
        failures.
        """
        modalities = (
            [Modality.IMAGE, Modality.VIDEO] if not kind
            else [Modality.IMAGE if kind == 'image' else Modality.VIDEO]
        )
        seen: dict[str, dict[str, Any]] = {}
        for modality in modalities:
            for info in self.registry.providers_for(modality):
                seen.setdefault(info.id, {'id': info.id, 'label': info.label, 'local': info.local})
        return sorted(seen.values(), key=lambda p: (not p['local'], p['id']))

    # -- what can be asked for ----------------------------------------------

    def _resolve(self, kind: str, provider: str | None, model: str | None):
        modality = Modality.IMAGE if kind == 'image' else Modality.VIDEO
        routes = (
            RouteSet(routes={modality: Route(provider=provider, model=model or '')})
            if provider else None
        )
        return self.registry.resolve(modality, routes)

    @staticmethod
    async def _models(impl: Any, kind: str) -> list[dict[str, Any]]:
        """A provider's models, asked for this kind of generation where it can be.

        The aggregators host both and their lists barely overlap, so `kind` is
        passed to anything whose signature takes it. Checked by inspection
        rather than by calling and catching TypeError: a TypeError raised
        *inside* a provider's own listing code would otherwise be swallowed and
        retried, which turns a bug into a silently empty dropdown.
        """
        lister = getattr(impl, 'models', None)
        if lister is None:
            return []
        try:
            if 'kind' in inspect.signature(lister).parameters:
                return await lister(kind)
            return await lister()
        except Exception as exc:  # noqa: BLE001
            log.info('%s did not list models: %s', getattr(impl, 'provider_id', impl), exc)
            return []

    async def describe(self, kind: str, provider: str | None = None, model: str | None = None) -> dict[str, Any]:
        """The form for one generator: its models, and its parameters.

        Asked of the provider every time. An install's samplers, upscalers and
        workflow templates are what is on that machine right now, and a form
        built from a remembered list offers a sampler that was uninstalled.
        """
        impl, route, info = self._resolve(kind, provider, model)
        chosen = model or route.model

        models = await self._models(impl, kind)
        if not chosen and models:
            chosen = str(models[0].get('id') or '')

        params = await self._schema(impl, chosen)

        return {
            'provider': info.id,
            'local': info.local,
            'model': chosen,
            'models': models,
            'params': [p.to_json() for p in params],
            'providers': self.providers(kind),
        }

    async def _schema(self, impl: Any, model: str) -> list[Param]:
        describer = getattr(impl, 'describe', None)
        if describer is None:
            # A provider with no opinion gets the common form, so a backend
            # this build has never met is still driveable rather than being
            # offered with no controls at all.
            from openmirror.media.params import common_image

            return common_image()
        return await describer(model)

    # -- making things ------------------------------------------------------

    def start(
        self,
        kind: str,
        prompt: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> Job:
        """Queue a generation and return immediately.

        Returning a job rather than awaiting one is not an optimisation. A
        video is minutes and a hosted image can be a minute and a half; an
        HTTP request that takes minutes is a request that dies to somebody's
        proxy, and a person watching a spinner with no way to stop it will
        kill the browser tab and leave a GPU working on something nobody
        wants any more.
        """
        impl, route, info = self._resolve(kind, provider, model)
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            prompt=prompt,
            provider=info.id,
            model=model or route.model,
        )
        self.jobs[job.id] = job
        self._prune()
        job.task = asyncio.create_task(self._run(job, impl, info, params or {}))
        return job

    async def _run(self, job: Job, impl: Any, info: Any, params: dict[str, Any]) -> None:
        try:
            if not job.model:
                listed = await self._models(impl, job.kind)
                if not listed:
                    raise NoProviderError(self._nothing_to_use(job, info))
                job.model = str(listed[0]['id'])

            schema = await self._schema(impl, job.model)
            clean = coerce(schema, params)
            job.params = dict(clean)
            ignored = unknown_keys(schema, params)
            if ignored:
                log.info('%s ignored %s', info.id, ', '.join(ignored))

            # Reference images are stored media, referred to by id. They are
            # turned into bytes here — once, centrally — because every provider
            # wants them in a different envelope (a data URI, bare base64, a
            # multipart part, an upload to the render box first) and none of
            # them should have to know how this build stores files.
            clean = self._load_images(schema, clean)

            size = clean.pop('size', None) or f'{clean.get("width", 1024)}x{clean.get("height", 1024)}'
            count = int(clean.pop('n', 1) or 1)
            # Not a parameter — it is the list of LoRA names the server has,
            # sent to the form so a person can see what to type into a prompt.
            clean.pop('loras_available', None)

            def report(fraction: float | None, note: str) -> None:
                job.state = 'running'
                job.progress = fraction
                job.note = note

            job.state = 'running'
            started = time.monotonic()
            if job.kind == 'image':
                results = await impl.generate(
                    job.prompt, model=job.model, n=count, size=size, progress=report, **clean
                )
            else:
                results = await impl.generate(job.prompt, model=job.model, progress=report, **clean)
            took = int((time.monotonic() - started) * 1000)

            recipe = self._recipe(schema, clean, size=size, count=count, kind=job.kind)
            job.media = [
                self.store.add(
                    media.data,
                    kind=job.kind,
                    media_type=media.media_type,
                    prompt=job.prompt,
                    provider=job.provider,
                    model=job.model,
                    seed=media.seed,
                    params=recipe,
                    notes=str(media.meta.get('revised_prompt', '')),
                )
                for media in results
            ]
            job.progress = 1.0
            job.note = f'{len(job.media)} made in {took // 1000}s' if job.media else ''
            job.state = 'done' if job.media else 'failed'
            if not job.media:
                job.error = (
                    'it finished and produced nothing this could read. For a ComfyUI template that '
                    'usually means no output node; for a hosted provider it usually means a safety '
                    'filter that returned success with an empty result.'
                )
        except asyncio.CancelledError:
            job.state = 'cancelled'
            job.note = 'stopped'
            # The provider is told, because a job abandoned rather than
            # cancelled keeps a GPU busy and keeps a hosted account billing.
            # Every hosted adapter implements this; ComfyUI has an interrupt
            # endpoint and most local ones have nothing, which is why a
            # missing method is an ordinary outcome rather than an error.
            for name in ('interrupt', 'stop'):
                stopper = getattr(impl, name, None)
                if stopper is not None:
                    try:
                        await stopper()
                    except Exception:  # noqa: BLE001
                        log.debug('could not interrupt %s', info.id, exc_info=True)
                    break
            raise
        except Exception as exc:  # noqa: BLE001 - a failed job is a state, not a crash
            log.exception('%s job %s failed', job.kind, job.id)
            job.state = 'failed'
            # The backend's own message, which is the useful one: "CUDA out of
            # memory", "insufficient credits" and "NSFW content detected" want
            # three different reactions and a generic failure tells you none.
            job.error = str(exc)[:600]
        finally:
            job.finished_at = time.time()

    @staticmethod
    def _recipe(schema: list[Param], clean: dict[str, Any], *, size: str, count: int, kind: str) -> dict[str, Any]:
        """What is worth recording about how this was made.

        A recipe is read by somebody trying to get the same result again, so
        anything in it that cannot produce that is worse than absent. Three
        things were, and all three were visible in the panel before they were
        fixed here:

        **A size nobody asked for.** `generate` takes a size because the image
        contract has always had one, and it is synthesised from the width and
        height when a caller did not give one — so a model that sizes by
        aspect ratio recorded `size 1024x1024`, which is not what it did and
        not a setting it has. Recorded only when the backend actually declared
        one of those controls.

        **`seed -1`.** -1 is a request for a random seed, not a seed. It is the
        first thing anyone looking for reproducibility reads, and it is the one
        value guaranteed not to reproduce anything — the seed that was really
        used is on the media record, where the provider put it.

        **`n 1`.** The default, on the parameter that means "how many of these
        did I ask for". One is what asking once looks like.

        A reference image is dropped too, but for a different reason: it is
        megabytes of picture and it is already in the store under its own id.
        """
        recipe = {k: v for k, v in clean.items() if not isinstance(v, bytes | bytearray)}
        recipe = {k: v for k, v in recipe.items() if not isinstance(v, list)}
        if recipe.get('seed') in (-1, '-1'):
            recipe.pop('seed')
        if kind == 'image':
            sized = {p.name for p in schema} & {'size', 'width', 'height'}
            if sized:
                recipe = {'size': size, **recipe}
            if count > 1:
                recipe = {'n': count, **recipe}
        return recipe

    @staticmethod
    def _nothing_to_use(job: Job, info: Any) -> str:
        if job.kind == 'video' and 'comfy' in info.id.lower():
            return ('no workflow template is installed, so there is nothing to generate with. '
                    'Export a graph from ComfyUI with Save (API format) and import it.')
        return f'{info.id} offered no {job.kind} models, and none was named.'

    def _load_images(self, schema: list[Param], clean: dict[str, Any]) -> dict[str, Any]:
        """Turn stored media ids into bytes, for the controls that take a picture.

        A control of kind `image` carries an id from the media store — never a
        path and never a data URI from the client. That is deliberate: the
        value travels through the same JSON as every other parameter and the
        only thing a client can name is something this server already has, so
        there is nothing here that turns a parameter into a file read.
        """
        wanted = {p.name for p in schema if p.kind == 'image'}
        out: dict[str, Any] = {}
        for key, value in clean.items():
            if key not in wanted or not value:
                out[key] = value
                continue
            ids = [v.strip() for v in str(value).split(',') if v.strip()]
            blobs = [data for data in (self._read(i) for i in ids) if data]
            if not blobs:
                continue
            # One image stays one image. Several providers take a single blob
            # and would be given a list of one, which is a different request.
            out[key] = blobs[0] if len(blobs) == 1 else blobs
        return out

    def _read(self, media_id: str) -> bytes | None:
        path = self.store.path(media_id)
        if path is None:
            log.warning('a generation referred to media %s, which is not in the store', media_id)
            return None
        try:
            return path.read_bytes()
        except OSError as exc:
            log.warning('could not read %s: %s', path, exc)
            return None

    async def generate_image(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an image and wait for it — the shape the agent's tool wants.

        The same job underneath, awaited. An agent turn that has to poll for
        its own tool result is a turn that spends two more round trips saying
        nothing, so the tool gets to block and the panel does not.
        """
        job = self.start('image', prompt, provider=provider, model=model, params=params)
        try:
            await job.task
        except asyncio.CancelledError:
            raise
        if job.state == 'failed':
            raise RuntimeError(job.error)

        impl, _, info = self._resolve('image', provider, model)
        schema = await self._schema(impl, job.model)
        return {
            'media': [m.to_json() for m in job.media],
            'provider': job.provider,
            'local': info.local,
            'model': job.model,
            'took_ms': job.elapsed * 1000,
            'ignored': unknown_keys(schema, params or {}),
        }

    def start_video(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> Job:
        """Kept because the video endpoint and the agent's tool both call it."""
        return self.start('video', prompt, provider=provider, model=model, params=params)

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.task is None or job.task.done():
            return False
        job.task.cancel()
        return True

    def _prune(self) -> None:
        """Forget the oldest finished jobs.

        Only finished ones, and only past the cap: a daemon that stays up for
        a fortnight should not accumulate every generation anyone ever made,
        and the results themselves are in the store regardless.
        """
        if len(self.jobs) <= KEEP_JOBS:
            return
        done = sorted(
            (j for j in self.jobs.values() if j.finished_at),
            key=lambda j: j.finished_at,
        )
        for job in done[: len(self.jobs) - KEEP_JOBS]:
            self.jobs.pop(job.id, None)
