"""Making pictures and video, from inside a turn.

The tools are thin — the service does the work — but two things about them are
deliberate.

**Generating is a network call, not a write.** It spends money on a hosted
provider and GPU time on a local one, and it produces a file in the media
library rather than in the working root. Grading it `NETWORK` puts it at the
same level as fetching a page: it runs unattended under `trusted`, and it is
confirmed under `ask`, which is the right place for something that can cost
real money per call.

**The model is told what it can turn before it turns anything.** `image_params`
exists because a model asked to "make it more detailed" will otherwise invent
a parameter name, have it silently dropped, and report success. Asking first
costs one step and is the difference between a considered setting and a
plausible-looking one.
"""

from __future__ import annotations

import json
from typing import Any

from openmirror.agent.tools.base import Assessment, Output, Tool, ToolContext, ToolError
from openmirror.protocol.agent import Risk
from openmirror.providers.registry import NoProviderError


class _MediaTool(Tool):
    def __init__(self, service: Any) -> None:
        self.service = service


class MediaParamsTool(_MediaTool):
    name = 'media_params'
    description = (
        'Ask what an image or video generator on this machine can be told to do: which '
        'models it has, and every parameter with its range and what it means. Do this '
        'before generating with anything but a plain prompt — a parameter that is not in '
        'this list is dropped, and you will be told it worked.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'kind': {'type': 'string', 'description': "'image' or 'video'. Default image."},
            'provider': {'type': 'string', 'description': 'Leave empty for the configured one.'},
            'model': {'type': 'string'},
        },
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        return Assessment(risk=Risk.READ, summary=f'ask what the {args.get("kind", "image")} generator can do')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        try:
            info = await self.service.describe(
                args.get('kind', 'image') or 'image', args.get('provider'), args.get('model')
            )
        except NoProviderError as exc:
            raise ToolError(str(exc)) from exc

        lines = [
            f'{info["provider"]}{" (local)" if info["local"] else ""}, model {info["model"] or "(none chosen)"}.',
        ]
        if info['models']:
            names = ', '.join(str(m.get('id')) for m in info['models'][:40])
            lines.append(f'Models: {names}')
        lines.append('')
        for param in info['params']:
            bits = [f'{param["name"]} ({param["kind"]}'
                    + (f', {param["min"]}–{param["max"]}' if param['min'] is not None else '') + ')']
            if param['options']:
                bits.append('one of: ' + ', '.join(param['options'][:20]))
            if param['default'] is not None:
                bits.append(f'default {param["default"]}')
            if param['help']:
                bits.append(param['help'])
            lines.append('  ' + ' — '.join(bits))

        return Output(content='\n'.join(lines), display={'params': info['params'], 'models': info['models']})


class GenerateImageTool(_MediaTool):
    name = 'generate_image'
    description = (
        'Make an image from a description. `params` takes anything media_params listed for '
        'this backend — steps, cfg_scale, sampler, width, height, seed and so on. Say what '
        'you set and why when you report back: a picture with no recipe cannot be iterated on.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'prompt': {'type': 'string'},
            'provider': {'type': 'string'},
            'model': {'type': 'string'},
            'params': {
                'type': 'object',
                'description': 'Backend parameters, from media_params. Anything not in that '
                               'list is ignored and reported back to you.',
            },
        },
        'required': ['prompt'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        prompt = (args.get('prompt') or '').strip()
        if not prompt:
            return Assessment(risk=Risk.NETWORK, summary='', invalid='a prompt is required')
        count = (args.get('params') or {}).get('n', 1)
        shown = prompt if len(prompt) <= 70 else prompt[:67] + '...'
        return Assessment(
            risk=Risk.NETWORK,
            summary=f'generate {count} image(s): {shown!r}',
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        try:
            result = await self.service.generate_image(
                args['prompt'],
                provider=args.get('provider'),
                model=args.get('model'),
                params=args.get('params') or {},
            )
        except NoProviderError as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f'the image backend failed: {exc}') from exc

        media = result['media']
        if not media:
            raise ToolError('the backend returned no images')

        lines = [
            f'Generated {len(media)} image(s) on {result["provider"]} '
            f'({result["model"]}) in {result["took_ms"]}ms.'
        ]
        for item in media:
            seed = f', seed {item["seed"]}' if item.get('seed') is not None else ''
            lines.append(f'  {item["id"]}{seed} — {item["url"]}')
            if item.get('notes'):
                lines.append(f'    the backend revised the prompt to: {item["notes"]}')
        if result['ignored']:
            # Said plainly rather than swallowed. A model that thinks it set
            # the sampler and did not will report a choice it never made.
            lines.append(
                'Ignored, because this backend has no such parameter: '
                + ', '.join(result['ignored'])
            )

        return Output(content='\n'.join(lines), display={'media': media, 'provider': result['provider']})


class GenerateVideoTool(_MediaTool):
    name = 'generate_video'
    description = (
        'Start making a video. This takes minutes, so it returns a job id immediately — '
        'check it with media_job. `model` is a workflow template installed on this machine; '
        'media_params lists them.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'prompt': {'type': 'string'},
            'provider': {'type': 'string'},
            'model': {'type': 'string', 'description': 'The workflow template to run.'},
            'params': {'type': 'object'},
        },
        'required': ['prompt'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        prompt = (args.get('prompt') or '').strip()
        if not prompt:
            return Assessment(risk=Risk.NETWORK, summary='', invalid='a prompt is required')
        shown = prompt if len(prompt) <= 70 else prompt[:67] + '...'
        return Assessment(risk=Risk.NETWORK, summary=f'generate a video: {shown!r} (minutes)')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        try:
            job = self.service.start_video(
                args['prompt'],
                provider=args.get('provider'),
                model=args.get('model'),
                params=args.get('params') or {},
            )
        except NoProviderError as exc:
            raise ToolError(str(exc)) from exc

        return Output(
            content=(
                f'Started video job {job.id} on {job.provider}. This takes minutes. '
                f'Check it with media_job("{job.id}") — do not poll it more than once every '
                'twenty seconds or so, and tell the person it is running rather than sitting '
                'silently waiting for it.'
            ),
            display=job.to_json(),
        )


class MediaJobTool(_MediaTool):
    name = 'media_job'
    description = 'Check on a video job started with generate_video.'
    input_schema = {
        'type': 'object',
        'properties': {'job_id': {'type': 'string'}},
        'required': ['job_id'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        return Assessment(risk=Risk.READ, summary=f'check video job {args.get("job_id", "")}')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        job = self.service.jobs.get(args.get('job_id', ''))
        if job is None:
            raise ToolError(f'no such job: {args.get("job_id")!r}')
        data = job.to_json()
        if job.state == 'running':
            content = f'Still running, {data["elapsed"]}s in.'
        elif job.state == 'done':
            urls = ', '.join(m['url'] for m in data['media'])
            content = f'Finished in {data["elapsed"]}s: {urls}'
        else:
            content = f'{job.state}: {job.error or "no reason given"}'
        return Output(content=content, display=data)


class ImportWorkflowTool(_MediaTool):
    name = 'import_workflow'
    description = (
        'Install a ComfyUI graph as a video or image template. `graph` is the JSON from '
        "ComfyUI's Save (API format) — not plain Save, which writes the editor layout "
        'rather than the graph. The prompt, seed, size and sampler settings are found and '
        'turned into parameters automatically; what it found is returned.'
    )
    input_schema = {
        'type': 'object',
        'properties': {
            'name': {'type': 'string', 'description': 'What to call the template.'},
            'graph': {'type': 'object', 'description': 'The API-format export, whole.'},
            'path': {
                'type': 'string',
                'description': 'A file to read the graph from instead, if you have one on disk.',
            },
        },
        'required': ['name'],
    }

    def assess(self, args: dict[str, Any], ctx: ToolContext) -> Assessment:
        name = (args.get('name') or '').strip()
        if not name:
            return Assessment(risk=Risk.WRITE, summary='', invalid='a name is required')
        if not args.get('graph') and not args.get('path'):
            return Assessment(risk=Risk.WRITE, summary='', invalid='pass either graph or path')
        return Assessment(risk=Risk.WRITE, summary=f'install the workflow template {name!r}')

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> Output:
        from openmirror.agent.tools.base import resolve_in_root
        from openmirror.media.workflow import install
        from openmirror.providers.comfyui import WORKFLOW_DIR

        graph = args.get('graph')
        if not graph:
            path = resolve_in_root(args['path'], ctx)
            try:
                graph = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ToolError(f'could not read {path}: {exc}') from exc

        # A file operation, so no connection is constructed for it — see
        # `workflow.install`.
        result = install(WORKFLOW_DIR, args['name'], graph)
        tokens = ', '.join(result['tokens']) or 'none'
        warning = (
            ''
            if 'prompt' in result['tokens']
            else '\nNo prompt token was found — the importer could not identify the positive '
                 'conditioning node. Put %prompt% into the right text node yourself, or this '
                 'template will always generate the prompt it was exported with.'
        )
        return Output(
            content=f'Installed {result["id"]}. Parameters found: {tokens}.{warning}',
            display=result,
        )


def media_tools(service: Any) -> list[Tool]:
    """The generation tools this install can actually serve.

    Nothing is offered that would fail on its first call. An install with a
    diffusion server and no video backend gets `generate_image` and not
    `generate_video`; one with neither gets none of this at all, and the model
    is never told it can make pictures on a machine that cannot.
    """
    kinds = service.can_generate()
    if not kinds:
        return []

    tools: list[Tool] = [MediaParamsTool(service)]
    if 'image' in kinds:
        tools.append(GenerateImageTool(service))
    if 'video' in kinds:
        # The other three only mean anything with video: a video is a job, so
        # it needs something to check on, and a job needs a workflow template
        # to run — which `import_workflow` is how you get.
        tools += [GenerateVideoTool(service), MediaJobTool(service), ImportWorkflowTool(service)]
    return tools
