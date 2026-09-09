"""ComfyUI, for video and for the diffusion models that only exist as graphs.

ComfyUI does not take a prompt; it takes a workflow — a graph of nodes with
the prompt buried somewhere inside it, in a node whose number depends on the
order somebody added things in the editor. So a template is stored per model
family and the prompt is injected into it. That is why this provider carries
workflow JSON at all, and why adding a new video model is a template rather
than code.

The claim in that last sentence used to be aspirational, because a template
still had to be hand-edited and nothing knew what it could be asked for.
`roost.media.workflow` is what makes it true: a template marks its inputs with
`%prompt%`-style tokens, the form is inferred from which tokens are present,
and `install()` takes a ComfyUI "Save (API format)" export and puts the tokens
in for you. So a new model really is one file, and the panel that drives it
appears by itself.

One constraint shapes the polling below. Perch's allowlist matches exact
paths, so `/history/{prompt_id}` — the obvious way to check on one job — is
not reachable through it: only bare `/history` is. Since ComfyUI's `/history`
returns every job keyed by id, this polls that and looks its own id up. It
costs a slightly larger response and works against both a direct ComfyUI and
one behind Perch, which is worth more.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

import aiohttp

from roost.media import workflow as workflow_mod
from roost.media.params import Param
from roost.net.transport import Transport
from roost.providers.base import GeneratedMedia, VideoProvider

log = logging.getLogger(__name__)

WORKFLOW_DIR = Path(__file__).parent / 'workflows'


class ComfyUIProvider(VideoProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str = '',
        *,
        provider_id: str = 'comfyui',
        timeout: int = 1800,
        poll_interval: float = 1.5,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.provider_id = provider_id
        # Video generation is minutes, not seconds, and a default HTTP timeout
        # will end a job that was going to succeed.
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.client_id = str(uuid.uuid4())
        self.transport = transport or Transport(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        h = {'Content-Type': 'application/json'}
        if self.api_key:
            h['Authorization'] = f'Bearer {self.api_key}'
        return h

    def _template_path(self, model: str) -> Path:
        # Resolved and checked against the directory, because a model name
        # arrives from a request and `../../etc/passwd` is a path too.
        path = (WORKFLOW_DIR / f'{model}.json').resolve()
        if WORKFLOW_DIR.resolve() not in path.parents or not path.is_file():
            available = sorted(p.stem for p in WORKFLOW_DIR.glob('*.json') if not p.name.endswith('.roost.json'))
            raise FileNotFoundError(
                f'no workflow template named {model!r}; have: {", ".join(available) or "none"}. '
                'Export one from ComfyUI with Save (API format) and import it — see '
                'roost/providers/workflows/README.md.'
            )
        return path

    def _load_workflow(self, model: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return workflow_mod.load(self._template_path(model))

    async def generate(self, prompt: str, *, model: str, **kw: Any) -> list[GeneratedMedia]:
        import random

        graph, _ = self._load_workflow(model)

        seed = kw.pop('seed', None)
        if seed in (None, -1, ''):
            # ComfyUI needs a number. -1 is our spelling of "surprise me", and
            # it is resolved here rather than sent, so the seed that produced a
            # result is recorded and the result can be reproduced.
            seed = random.randint(0, 2**31 - 1)

        values = {
            'prompt': prompt,
            'negative': kw.pop('negative_prompt', kw.pop('negative', '')),
            'seed': int(seed),
            **{k: v for k, v in kw.items() if v is not None},
        }
        workflow = workflow_mod.inject(graph, values)
        payload = {'prompt': workflow, 'client_id': self.client_id}

        async with self.transport.session() as session:
            async with session.post(f'{self.base_url}/prompt', json=payload, headers=self._headers()) as resp:
                if resp.status != 200:
                    raise RuntimeError(f'{self.provider_id}: queue failed: {(await resp.text())[:300]}')
                queued = await resp.json()

            prompt_id = queued.get('prompt_id')
            if not prompt_id:
                raise RuntimeError(f'{self.provider_id}: no prompt_id in queue response')

            outputs = await self._await_completion(session, prompt_id)
            return await self._collect(session, outputs, int(seed))

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
        async with self.transport.session(15) as session:
            await session.post(f'{self.base_url}/interrupt', headers=self._headers())

    async def models(self) -> list[dict[str, Any]]:
        """The workflow templates on disk, not the checkpoints on the server.

        A template is the unit a caller can actually ask for; a checkpoint
        without a graph around it is not runnable. Each row carries the tokens
        its template uses, so a client can say "this one takes a frame count
        and that one does not" without loading every graph.
        """
        WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
        out: list[dict[str, Any]] = []
        for path in sorted(WORKFLOW_DIR.glob('*.json')):
            if path.name.endswith('.roost.json'):
                continue
            try:
                graph, overrides = workflow_mod.load(path)
            except (OSError, ValueError) as exc:
                log.warning('workflow %s is not loadable: %s', path.name, exc)
                continue
            out.append(
                {
                    'id': path.stem,
                    'provider': self.provider_id,
                    'label': overrides.get('label', path.stem),
                    'kind': overrides.get('kind', 'video'),
                    'tokens': sorted(workflow_mod.tokens_in(graph)),
                    'note': overrides.get('note', ''),
                }
            )
        return out

    async def describe(self, model: str = '') -> list[Param]:
        """The form for one template, from the tokens it actually uses.

        No template named: the union of every token in every installed
        template, which is what a picker needs before a model has been chosen.
        """
        WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
        if model:
            graph, overrides = self._load_workflow(model)
            return workflow_mod.describe(graph, overrides.get('params'))

        merged: dict[str, Param] = {}
        for path in sorted(WORKFLOW_DIR.glob('*.json')):
            if path.name.endswith('.roost.json'):
                continue
            try:
                graph, overrides = workflow_mod.load(path)
            except (OSError, ValueError):
                continue
            for param in workflow_mod.describe(graph, overrides.get('params')):
                merged.setdefault(param.name, param)
        return list(merged.values())

    def install(self, name: str, graph: dict[str, Any], *, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Import a ComfyUI API export as a template.

        Delegates, because installing a template touches the disk and not this
        server — a caller that does not have a connection in hand should call
        `workflow.install` directly rather than inventing one.
        """
        return workflow_mod.install(WORKFLOW_DIR, name, graph, overrides=overrides)
