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

**Video is a job, not a request.** Minutes rather than seconds, so it runs as
a task that can be watched and cancelled, and the progress it can report is
coarse because ComfyUI behind Perch only exposes polling. Saying "about a
minute in, still running" is worth more than a spinner, and pretending to a
percentage would be a lie.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from roost.media.params import Param, coerce, unknown_keys
from roost.media.store import Media, MediaStore
from roost.providers.base import Modality
from roost.providers.registry import NoProviderError, ProviderRegistry, Route, RouteSet

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Job:
    """One video generation, which is long enough to need watching."""

    id: str
    kind: str
    prompt: str
    provider: str
    model: str
    started_at: float = field(default_factory=time.time)
    state: str = 'running'          # running | done | failed | cancelled
    error: str = ''
    media: list[Media] = field(default_factory=list)
    task: Any = field(default=None, repr=False)

    def to_json(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'kind': self.kind,
            'prompt': self.prompt,
            'provider': self.provider,
            'model': self.model,
            'state': self.state,
            'error': self.error,
            'elapsed': int(time.time() - self.started_at),
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

    # -- what can be asked for ----------------------------------------------

    def _resolve(self, kind: str, provider: str | None, model: str | None):
        modality = Modality.IMAGE if kind == 'image' else Modality.VIDEO
        routes = (
            RouteSet(routes={modality: Route(provider=provider, model=model or '')})
            if provider else None
        )
        return self.registry.resolve(modality, routes)

    async def describe(self, kind: str, provider: str | None = None, model: str | None = None) -> dict[str, Any]:
        """The form for one generator: its models, and its parameters.

        Asked of the provider every time. An install's samplers, upscalers and
        workflow templates are what is on that machine right now, and a form
        built from a remembered list offers a sampler that was uninstalled.
        """
        impl, route, info = self._resolve(kind, provider, model)
        chosen = model or route.model

        models: list[dict[str, Any]] = []
        lister = getattr(impl, 'models', None)
        if lister is not None:
            try:
                models = await lister()
            except Exception as exc:  # noqa: BLE001
                log.info('%s did not list models: %s', info.id, exc)

        if not chosen and models:
            chosen = str(models[0].get('id') or '')

        describer = getattr(impl, 'describe', None)
        if describer is not None:
            params: list[Param] = await describer(chosen)
        else:
            # A provider with no opinion gets the common form, so a backend
            # this build has never met is still driveable rather than being
            # offered with no controls at all.
            from roost.media.params import common_image

            params = common_image()

        return {
            'provider': info.id,
            'local': info.local,
            'model': chosen,
            'models': models,
            'params': [p.to_json() for p in params],
        }

    # -- making things ------------------------------------------------------

    async def generate_image(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        impl, route, info = self._resolve('image', provider, model)
        chosen = model or route.model
        if not chosen:
            listed = await impl.models() if hasattr(impl, 'models') else []
            chosen = str(listed[0]['id']) if listed else ''

        schema = await self._schema(impl, chosen)
        clean = coerce(schema, params or {})
        ignored = unknown_keys(schema, params or {})

        size = clean.pop('size', None) or f'{clean.get("width", 1024)}x{clean.get("height", 1024)}'
        count = int(clean.pop('n', 1) or 1)
        # Not a parameter — it is the list of LoRA names the server has, sent
        # to the form so a person can see what to type into the prompt.
        clean.pop('loras_available', None)

        started = time.monotonic()
        results = await impl.generate(prompt, model=chosen, n=count, size=size, **clean)
        took = int((time.monotonic() - started) * 1000)

        stored = [
            self.store.add(
                media.data,
                kind='image',
                media_type=media.media_type,
                prompt=prompt,
                provider=info.id,
                model=chosen,
                seed=media.seed,
                params={'size': size, 'n': count, **clean},
                notes=str(media.meta.get('revised_prompt', '')),
            )
            for media in results
        ]

        return {
            'media': [m.to_json() for m in stored],
            'provider': info.id,
            'local': info.local,
            'model': chosen,
            'took_ms': took,
            'ignored': ignored,
        }

    async def _schema(self, impl: Any, model: str) -> list[Param]:
        describer = getattr(impl, 'describe', None)
        if describer is None:
            from roost.media.params import common_image

            return common_image()
        return await describer(model)

    def start_video(
        self,
        prompt: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> Job:
        """Queue a video and return immediately.

        Returning a job rather than awaiting one is not an optimisation. A
        video is minutes, an HTTP request that takes minutes is a request that
        dies to somebody's proxy, and a person watching a spinner for four
        minutes with no way to stop it will kill the browser tab and leave the
        GPU working on something nobody wants any more.
        """
        impl, route, info = self._resolve('video', provider, model)
        chosen = model or route.model
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind='video',
            prompt=prompt,
            provider=info.id,
            model=chosen,
        )
        self.jobs[job.id] = job
        job.task = asyncio.create_task(self._run_video(job, impl, chosen, params or {}))
        return job

    async def _run_video(self, job: Job, impl: Any, model: str, params: dict[str, Any]) -> None:
        try:
            if not model:
                listed = await impl.models() if hasattr(impl, 'models') else []
                if not listed:
                    raise NoProviderError(
                        'no workflow template is installed, so there is nothing to generate with. '
                        'Export a graph from ComfyUI with Save (API format) and import it.'
                    )
                model = str(listed[0]['id'])
                job.model = model

            schema = await self._schema(impl, model)
            clean = coerce(schema, params)

            results = await impl.generate(job.prompt, model=model, **clean)
            job.media = [
                self.store.add(
                    media.data,
                    kind='video',
                    media_type=media.media_type,
                    prompt=job.prompt,
                    provider=job.provider,
                    model=model,
                    seed=media.seed,
                    params=clean,
                )
                for media in results
            ]
            job.state = 'done' if job.media else 'failed'
            if not job.media:
                job.error = 'the workflow finished but produced no output node this could read'
        except asyncio.CancelledError:
            job.state = 'cancelled'
            # Best effort: ComfyUI has an interrupt endpoint and most others
            # do not, so a cancelled job may still be occupying the GPU.
            interrupt = getattr(impl, 'interrupt', None)
            if interrupt is not None:
                try:
                    await interrupt()
                except Exception:  # noqa: BLE001
                    log.debug('could not interrupt the backend', exc_info=True)
            raise
        except Exception as exc:  # noqa: BLE001 - a failed job is a state, not a crash
            log.exception('video job %s failed', job.id)
            job.state = 'failed'
            job.error = str(exc)[:500]

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.task is None or job.task.done():
            return False
        job.task.cancel()
        return True
