"""Generating images and video, and getting them back.

The shape follows what generating actually is now: **a job**, both kinds. It
used to follow what the two kinds *were* — an image was a request and a video
was a job — and that stopped being true the moment this build could reach a
hosted image model, which can sit in a queue for ninety seconds. A
ninety-second HTTP request dies to proxies, laptop lids and browser tabs,
taking the work with it.

`POST /image` is kept synchronous anyway, because the agent's tool calls it
and a tool that returns the thing it made is much easier for a model to reason
about than one that returns a ticket. The panel uses `POST /jobs` for both
kinds and watches. Same code underneath; the difference is only who waits.

One ordering constraint runs through this file: every fixed path — `/jobs`,
`/upload`, `/describe`, `/workflows` — is declared **before** `/{media_id}`.
FastAPI matches in declaration order, so the other way round makes `/upload`
a request for a piece of media with the id "upload", and the 404 that produces
is a genuinely baffling thing to debug.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from openmirror.providers.registry import NoProviderError

log = logging.getLogger(__name__)

router = APIRouter(prefix='/api/media')

# Set at start-up, and None when there is nowhere to put the results. Every
# endpoint checks: a 503 that says why beats an AttributeError.
service: Any = None

#: What an upload may be, and how large. A reference image is a photograph,
#: not an archive — and this endpoint writes a file to disk from an
#: unauthenticated-by-shape request body, so the ceiling is here rather than
#: left to whatever the reverse proxy happens to be configured with.
UPLOADABLE = frozenset({'image/png', 'image/jpeg', 'image/webp', 'image/gif'})
MAX_UPLOAD = 32 * 1024 * 1024


def _service():
    if service is None:
        raise HTTPException(status_code=503, detail='media generation is not configured on this install')
    return service


@router.get('/describe')
async def describe(kind: str = 'image', provider: str | None = None, model: str | None = None) -> dict[str, Any]:
    """The models and the controls for one generator, asked of it now.

    This is what makes "as advanced as you want" possible without a new
    release per backend: the panel is drawn from what comes back here, so an
    A1111 with a new sampler installed grows the option by itself — and so
    does a Replicate model published this morning, whose form comes from the
    schema Replicate publishes for it.
    """
    if kind not in ('image', 'video'):
        raise HTTPException(status_code=400, detail="kind must be 'image' or 'video'")
    try:
        return await _service().describe(kind, provider, model)
    except NoProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get('/providers')
async def providers(kind: str = '') -> dict[str, Any]:
    """Who can answer for this kind of generation, which is not the same set
    as who is connected: six things may make pictures and two of them video."""
    return {'providers': _service().providers(kind)}


class Generate(BaseModel):
    prompt: str
    provider: str | None = None
    model: str | None = None
    # Everything else the backend declared. Loose on purpose: this build must
    # not be the thing that stops someone using a parameter their server has
    # and this version has never heard of.
    params: dict[str, Any] = {}


class StartJob(Generate):
    kind: str = 'image'


@router.post('/jobs')
async def start_job(body: StartJob) -> dict[str, Any]:
    """Start a generation of either kind and return something to watch."""
    if body.kind not in ('image', 'video'):
        raise HTTPException(status_code=400, detail="kind must be 'image' or 'video'")
    if not body.prompt.strip() and body.kind == 'image' and not body.params.get('image'):
        # An image-to-image edit with no words is a real request; a text
        # generation with no words is not.
        raise HTTPException(status_code=400, detail='a prompt is required')
    try:
        job = _service().start(
            body.kind, body.prompt, provider=body.provider, model=body.model, params=body.params
        )
    except NoProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return job.to_json()


@router.post('/image')
async def make_image(body: Generate) -> dict[str, Any]:
    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail='a prompt is required')
    try:
        return await _service().generate_image(
            body.prompt, provider=body.provider, model=body.model, params=body.params
        )
    except NoProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        # The backend's own message, which is the useful one: "CUDA out of
        # memory" and "no such checkpoint" want different reactions and a
        # generic 500 tells the person neither.
        raise HTTPException(status_code=502, detail=str(exc)[:500]) from exc


@router.post('/video')
async def make_video(body: Generate) -> dict[str, Any]:
    if not body.prompt.strip() and not body.params.get('image'):
        raise HTTPException(status_code=400, detail='a prompt or a starting image is required')
    try:
        job = _service().start_video(
            body.prompt, provider=body.provider, model=body.model, params=body.params
        )
    except NoProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return job.to_json()


@router.get('/jobs')
async def list_jobs() -> dict[str, Any]:
    return {'jobs': [j.to_json() for j in _service().jobs.values()]}


@router.get('/jobs/{job_id}')
async def get_job(job_id: str) -> dict[str, Any]:
    job = _service().jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='no such job')
    return job.to_json()


@router.delete('/jobs/{job_id}')
async def cancel_job(job_id: str) -> dict[str, bool]:
    if not _service().cancel(job_id):
        raise HTTPException(status_code=404, detail='no such job, or it has already finished')
    return {'ok': True}


# ---------------------------------------------------------------------------
# Reference images
# ---------------------------------------------------------------------------


@router.post('/upload')
async def upload(request: Request) -> dict[str, Any]:
    """Put a picture into the store so a generation can refer to it by id.

    The body is the file itself rather than a multipart form. Two reasons, and
    the first is the one that decided it: multipart parsing in FastAPI needs a
    dependency this project does not otherwise have, and adding one so that a
    browser can send a single file it already holds as a Blob is a poor trade.
    The second is that it makes the client simpler — `fetch(url, {body: file})`
    and nothing else.

    Ids rather than paths or data URIs are what a parameter carries afterwards,
    which is the part that matters for safety: the only thing a client can name
    in an image field is something this server already put in its own store.
    """
    media_type = (request.headers.get('content-type') or '').split(';')[0].strip().lower()
    if media_type not in UPLOADABLE:
        raise HTTPException(
            status_code=415,
            detail=f'{media_type or "that"} cannot be used as a reference image. '
                   f'Send one of: {", ".join(sorted(UPLOADABLE))}.',
        )

    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail='the upload was empty')
    if len(data) > MAX_UPLOAD:
        raise HTTPException(
            status_code=413,
            detail=f'that is {len(data) // (1024 * 1024)}MB and the limit is {MAX_UPLOAD // (1024 * 1024)}MB.',
        )

    media = _service().store.add(
        data,
        kind='image',
        media_type=media_type,
        prompt=request.headers.get('x-openmirror-filename', '')[:200],
        source='upload',
    )
    return media.to_json()


# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------


@router.get('')
async def list_media(
    limit: int = 60, offset: int = 0, kind: str = '', source: str = 'generated', q: str = ''
) -> dict[str, Any]:
    items = _service().store.list(
        limit=max(1, min(200, limit)), offset=max(0, offset), kind=kind, source=source, search=q
    )
    return {
        'media': [m.to_json() for m in items],
        # Whether asking again with a larger offset is worth it. Cheaper than
        # counting the whole store, and it is the only fact an infinite scroll
        # actually needs.
        'more': len(items) == max(1, min(200, limit)),
    }


@router.get('/{media_id}')
async def get_media(media_id: str) -> dict[str, Any]:
    media = _service().store.get(media_id)
    if media is None:
        raise HTTPException(status_code=404, detail='no such media')
    return media.to_json()


@router.get('/{media_id}/file')
async def get_file(media_id: str) -> FileResponse:
    store = _service().store
    media = store.get(media_id)
    path = store.path(media_id)
    if media is None or path is None:
        raise HTTPException(status_code=404, detail='no such media')
    return FileResponse(path, media_type=media.media_type, filename=path.name)


@router.delete('/{media_id}')
async def delete_media(media_id: str) -> dict[str, bool]:
    if not _service().store.remove(media_id):
        raise HTTPException(status_code=404, detail='no such media')
    return {'ok': True}


# ---------------------------------------------------------------------------
# Workflow templates
# ---------------------------------------------------------------------------


class ImportWorkflow(BaseModel):
    name: str
    # A ComfyUI "Save (API format)" export, whole.
    graph: dict[str, Any]
    overrides: dict[str, Any] | None = None


@router.post('/workflows')
async def import_workflow(body: ImportWorkflow) -> dict[str, Any]:
    """Import a ComfyUI graph as a template, tokenising it on the way in.

    What comes back is the list of tokens it found, which is the only useful
    confirmation: it says which controls the panel will have, and whether the
    importer managed to find the prompt at all.
    """
    from openmirror.media.workflow import install
    from openmirror.providers.comfyui import WORKFLOW_DIR

    if not body.graph:
        raise HTTPException(status_code=400, detail='the graph is empty — did you use Save (API format)?')

    # No provider is constructed. This writes a file into the template
    # directory and never speaks to a ComfyUI, so building something that
    # looks like a connection to reach it would be inventing an address —
    # and an invented address is one that will eventually be dialled.
    try:
        return install(WORKFLOW_DIR, body.name, body.graph, overrides=body.overrides)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f'could not write the template: {exc}') from exc
