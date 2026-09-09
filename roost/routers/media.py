"""Generating images and video, and getting them back.

The shape follows what the two kinds actually are. An image is a request: you
wait a few seconds and get pictures. A video is a job: you start it, watch it,
and may want to stop it. Pretending they are the same would mean either
polling for something that was ready immediately, or holding an HTTP request
open for four minutes — and four-minute requests die to proxies, laptop lids
and browser tabs, taking the GPU's work with them.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from roost.providers.registry import NoProviderError

log = logging.getLogger(__name__)

router = APIRouter(prefix='/api/media')

# Set at start-up, and None when there is nowhere to put the results. Every
# endpoint checks: a 503 that says why beats an AttributeError.
service: Any = None


def _service():
    if service is None:
        raise HTTPException(status_code=503, detail='media generation is not configured on this install')
    return service


@router.get('/describe')
async def describe(kind: str = 'image', provider: str | None = None, model: str | None = None) -> dict[str, Any]:
    """The models and the controls for one generator, asked of it now.

    This is what makes "as advanced as you want" possible without a new
    release per backend: the panel is drawn from what comes back here, so an
    A1111 with a new sampler installed grows the option by itself.
    """
    if kind not in ('image', 'video'):
        raise HTTPException(status_code=400, detail="kind must be 'image' or 'video'")
    try:
        return await _service().describe(kind, provider, model)
    except NoProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


class Generate(BaseModel):
    prompt: str
    provider: str | None = None
    model: str | None = None
    # Everything else the backend declared. Loose on purpose: this build must
    # not be the thing that stops someone using a parameter their server has
    # and this version has never heard of.
    params: dict[str, Any] = {}


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
    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail='a prompt is required')
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
# The library
# ---------------------------------------------------------------------------


@router.get('')
async def list_media(limit: int = 60, kind: str = '') -> dict[str, Any]:
    return {'media': [m.to_json() for m in _service().store.list(limit=limit, kind=kind)]}


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
    from roost.providers.comfyui import ComfyUIProvider

    if not body.graph:
        raise HTTPException(status_code=400, detail='the graph is empty — did you use Save (API format)?')

    # Any ComfyUI provider will do: installing a template is a local file
    # operation and does not touch the server it would be run on.
    provider = ComfyUIProvider('http://127.0.0.1:8188')
    try:
        return provider.install(body.name, body.graph, overrides=body.overrides)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f'could not write the template: {exc}') from exc
