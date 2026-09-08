"""ComfyUI, for video and for the diffusion models that only exist as graphs.

ComfyUI does not take a prompt; it takes a workflow — a graph of nodes with
the prompt buried somewhere inside it. So a template is stored per model
family and the prompt is injected into named nodes. That is why this provider
carries workflow JSON at all, and why adding a new video model is usually a
template rather than code.

One constraint shapes the polling below. Perch's allowlist matches exact
paths, so `/history/{prompt_id}` — the obvious way to check on one job — is
not reachable through it: only bare `/history` is. Since ComfyUI's `/history`
returns every job keyed by id, this polls that and looks its own id up. It
costs a slightly larger response and works against both a direct ComfyUI and
one behind Perch, which is worth more.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import aiohttp

from roost.providers.base import GeneratedMedia, VideoProvider

log = logging.getLogger(__name__)

WORKFLOW_DIR = Path(__file__).parent / 'workflows'

# Where the prompt goes in a template. A workflow marks its own injection
# points with these strings, so a template is editable in ComfyUI itself and
# then saved out without needing code to know its node numbering.
PROMPT_TOKEN = '%prompt%'
NEGATIVE_TOKEN = '%negative%'
SEED_TOKEN = '%seed%'


def _inject(node_tree: Any, prompt: str, negative: str, seed: int) -> Any:
    """Replace the marker strings anywhere they appear in the graph."""
    if isinstance(node_tree, dict):
        return {k: _inject(v, prompt, negative, seed) for k, v in node_tree.items()}
    if isinstance(node_tree, list):
        return [_inject(v, prompt, negative, seed) for v in node_tree]
    if isinstance(node_tree, str):
        if node_tree == SEED_TOKEN:
            return seed
        return node_tree.replace(PROMPT_TOKEN, prompt).replace(NEGATIVE_TOKEN, negative)
    return node_tree


class ComfyUIProvider(VideoProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str = '',
        *,
        provider_id: str = 'comfyui',
        timeout: int = 1800,
        poll_interval: float = 1.5,
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.provider_id = provider_id
        # Video generation is minutes, not seconds, and a default HTTP timeout
        # will end a job that was going to succeed.
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.client_id = str(uuid.uuid4())

    def _headers(self) -> dict[str, str]:
        h = {'Content-Type': 'application/json'}
        if self.api_key:
            h['Authorization'] = f'Bearer {self.api_key}'
        return h

    def _load_workflow(self, model: str) -> dict[str, Any]:
        path = WORKFLOW_DIR / f'{model}.json'
        if not path.is_file():
            available = sorted(p.stem for p in WORKFLOW_DIR.glob('*.json'))
            raise FileNotFoundError(
                f'no workflow template named {model!r}; have: {", ".join(available) or "none"}'
            )
        return json.loads(path.read_text())

    async def generate(self, prompt: str, *, model: str, **kw: Any) -> list[GeneratedMedia]:
        import random

        seed = kw.pop('seed', None)
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        negative = kw.pop('negative_prompt', '')

        workflow = _inject(copy.deepcopy(self._load_workflow(model)), prompt, negative, seed)
        payload = {'prompt': workflow, 'client_id': self.client_id}

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f'{self.base_url}/prompt', json=payload, headers=self._headers()) as resp:
                if resp.status != 200:
                    raise RuntimeError(f'{self.provider_id}: queue failed: {(await resp.text())[:300]}')
                queued = await resp.json()

            prompt_id = queued.get('prompt_id')
            if not prompt_id:
                raise RuntimeError(f'{self.provider_id}: no prompt_id in queue response')

            outputs = await self._await_completion(session, prompt_id)
            return await self._collect(session, outputs, seed)

    async def _await_completion(self, session: aiohttp.ClientSession, prompt_id: str) -> dict[str, Any]:
        """Poll until this job appears in history.

        ComfyUI has a websocket that reports progress, which Perch does not
        proxy — so polling is the only mechanism that works in both
        deployments, and the progress it can report is coarse.
        """
        deadline = asyncio.get_running_loop().time() + self.timeout
        while asyncio.get_running_loop().time() < deadline:
            async with session.get(f'{self.base_url}/history', headers=self._headers()) as resp:
                if resp.status == 200:
                    history = await resp.json()
                    entry = history.get(prompt_id)
                    if entry:
                        status = entry.get('status') or {}
                        if status.get('status_str') == 'error':
                            raise RuntimeError(f'{self.provider_id}: workflow failed: {status.get("messages")}')
                        if entry.get('outputs'):
                            return entry['outputs']
            await asyncio.sleep(self.poll_interval)
        raise TimeoutError(f'{self.provider_id}: job {prompt_id} did not finish within {self.timeout}s')

    async def _collect(
        self, session: aiohttp.ClientSession, outputs: dict[str, Any], seed: int
    ) -> list[GeneratedMedia]:
        media: list[GeneratedMedia] = []
        for node in outputs.values():
            # A video workflow writes under 'gifs' (webp/mp4) or 'videos'
            # depending on the save node; an image one writes under 'images'.
            for key in ('gifs', 'videos', 'images'):
                for item in node.get(key) or []:
                    params = {
                        'filename': item['filename'],
                        'subfolder': item.get('subfolder', ''),
                        'type': item.get('type', 'output'),
                    }
                    async with session.get(
                        f'{self.base_url}/view', params=params, headers=self._headers()
                    ) as resp:
                        if resp.status != 200:
                            log.warning('%s: could not fetch %s', self.provider_id, item['filename'])
                            continue
                        data = await resp.read()
                    media.append(
                        GeneratedMedia(
                            data=data,
                            media_type=resp.headers.get('Content-Type', 'application/octet-stream'),
                            seed=seed,
                            meta={'filename': item['filename']},
                        )
                    )
        return media

    async def interrupt(self) -> None:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await session.post(f'{self.base_url}/interrupt', headers=self._headers())

    async def models(self) -> list[dict[str, Any]]:
        """The workflow templates on disk, not the checkpoints on the server.

        A template is the unit a caller can actually ask for; a checkpoint
        without a graph around it is not runnable.
        """
        WORKFLOW_DIR.mkdir(exist_ok=True)
        return [{'id': p.stem, 'provider': self.provider_id} for p in sorted(WORKFLOW_DIR.glob('*.json'))]
