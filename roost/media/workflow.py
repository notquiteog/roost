"""ComfyUI workflows as things a form can drive.

ComfyUI does not take a prompt. It takes a graph, and the prompt is a string
buried in a node whose number depends on the order somebody added things in
the editor. That is why the existing provider carries workflow JSON at all —
and why, until now, "add a video model" meant hand-editing a graph.

Two ideas make it a form instead.

**A template marks its own inputs with tokens.** `%prompt%`, `%steps%`,
`%seed%` and so on, written into the graph wherever they belong. The template
stays loadable in ComfyUI itself — a token is just a string — so it can be
opened, adjusted and saved back out.

**The form is inferred from which tokens are present.** A template using
`%frames%` gets a frame-count slider and one that does not, does not. So a new
model is a file, exactly as the provider's docstring has always claimed, and
nothing has to be registered anywhere.

`tokenise` is what makes that practical for a graph somebody already has:
point it at a ComfyUI "Save (API format)" export and it finds the prompt, the
sampler settings and the size, and replaces them with the tokens. It is
deliberately conservative — it matches on ComfyUI's own class names and input
names, and anything it does not recognise it leaves exactly as it was, because
a template that keeps a hard-coded value is merely inflexible while one with a
token in the wrong place is broken.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

from roost.media.params import Param

log = logging.getLogger(__name__)

TOKEN = re.compile(r'%([a-z_]+)%')

# Every token a template may use, and the control it turns into. Adding a row
# here is what makes a new knob available to every template at once.
TOKENS: dict[str, Param] = {
    'prompt': Param('prompt', 'Prompt', 'text', group='prompt',
                    help='What to make.'),
    'negative': Param('negative', 'Negative prompt', 'text', group='prompt', advanced=True,
                      help='What to keep out.'),
    'seed': Param('seed', 'Seed', 'seed', default=-1, group='sampling', advanced=True,
                  help='-1 picks a new one each time.'),
    'steps': Param('steps', 'Steps', 'int', default=20, minimum=1, maximum=150, step=1,
                   group='sampling'),
    'cfg': Param('cfg', 'CFG scale', 'float', default=7.0, minimum=1.0, maximum=30.0, step=0.5,
                 group='sampling',
                 help='How hard to push toward the prompt.'),
    'denoise': Param('denoise', 'Denoise', 'float', default=1.0, minimum=0.0, maximum=1.0,
                     step=0.01, group='sampling', advanced=True),
    'sampler': Param('sampler', 'Sampler', 'string', default='euler', group='sampling',
                     advanced=True),
    'scheduler': Param('scheduler', 'Scheduler', 'string', default='normal', group='sampling',
                       advanced=True),
    'width': Param('width', 'Width', 'int', default=832, minimum=64, maximum=4096, step=8,
                   group='shape'),
    'height': Param('height', 'Height', 'int', default=480, minimum=64, maximum=4096, step=8,
                    group='shape'),
    'frames': Param('frames', 'Frames', 'int', default=49, minimum=1, maximum=1024, step=1,
                    group='motion',
                    help='How many frames to generate. Length in seconds is this divided '
                         'by the frame rate.'),
    'fps': Param('fps', 'Frames per second', 'int', default=16, minimum=1, maximum=60, step=1,
                 group='motion'),
    'motion': Param('motion', 'Motion amount', 'int', default=127, minimum=0, maximum=255, step=1,
                    group='motion', advanced=True,
                    help='For models that take a motion bucket. Higher moves more, and past '
                         'about 180 it mostly wobbles.'),
    'batch': Param('batch', 'Batch size', 'int', default=1, minimum=1, maximum=8, step=1,
                   group='output'),
    'checkpoint': Param('checkpoint', 'Checkpoint', 'string', group='prompt', advanced=True,
                        help='The model file to load, as ComfyUI names it.'),
}

# What each token is worth when nobody says. Kept separate from the Param
# defaults so that a sidecar overriding a label cannot accidentally move a
# default out from under an existing template.
#
# The two text tokens fall back to an empty string rather than to nothing: a
# template with a negative prompt the caller did not fill in must send an empty
# negative, not the literal text "%negative%" — which conditions the image on
# the word "negative" and is the sort of bug that produces a slightly worse
# picture rather than an error.
FALLBACKS: dict[str, object] = {
    name: p.default for name, p in TOKENS.items() if p.default is not None
}
FALLBACKS.update({'prompt': '', 'negative': '', 'checkpoint': ''})


def tokens_in(graph: Any) -> set[str]:
    """Every token used anywhere in a graph."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            found.update(TOKEN.findall(node))

    walk(graph)
    return found & set(TOKENS)


def describe(graph: Any, overrides: dict[str, Any] | None = None) -> list[Param]:
    """The form for one template: the tokens it uses, in a sensible order.

    Ordered by group rather than by discovery, so two templates that use the
    same tokens produce the same form — a panel whose fields move when you
    change model is a panel nobody trusts.
    """
    order = ['prompt', 'shape', 'motion', 'sampling', 'quality', 'output']
    params = [TOKENS[name] for name in tokens_in(graph)]

    for param in params:
        patch = (overrides or {}).get(param.name)
        if not isinstance(patch, dict):
            continue
        # A copy: TOKENS is module state shared by every template, and editing
        # it in place would give one template's sidecar effect over all of them.
        # `fields` rather than `__dict__` — Param has slots, so it has no dict.
        base = {f.name: getattr(param, f.name) for f in fields(param)}
        allowed = {f.name for f in fields(param)}
        params[params.index(param)] = Param(**{**base, **{
            k: v for k, v in patch.items() if k in allowed
        }})

    return sorted(params, key=lambda p: (order.index(p.group) if p.group in order else 99, p.name))


def inject(graph: Any, values: dict[str, Any]) -> Any:
    """Fill a template's tokens in.

    A token that is the whole of a string becomes the value with its own type —
    `"%steps%"` becomes the integer 20, because ComfyUI validates types and a
    string where an int belongs fails inside the queue rather than at the
    request. A token inside a longer string is substituted textually, which is
    what makes `"a photo of %prompt%, cinematic"` work.
    """
    filled = {**FALLBACKS, **{k: v for k, v in values.items() if v is not None}}

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, str):
            whole = TOKEN.fullmatch(node)
            if whole and whole.group(1) in filled:
                return filled[whole.group(1)]
            return TOKEN.sub(lambda m: str(filled.get(m.group(1), m.group(0))), node)
        return node

    return walk(copy.deepcopy(graph))


# ---------------------------------------------------------------------------
# Importing a graph somebody already has
# ---------------------------------------------------------------------------

# ComfyUI's own class and input names. Matched on rather than on node numbers,
# because the numbers are whatever order the editor happened to assign.
_SAMPLER_CLASSES = ('KSampler', 'KSamplerAdvanced', 'SamplerCustom', 'KSamplerSelect')
_LATENT_CLASSES = (
    'EmptyLatentImage', 'EmptySD3LatentImage', 'EmptyHunyuanLatentVideo',
    'EmptyLTXVLatentVideo', 'EmptyMochiLatentVideo', 'EmptyCosmosLatentVideo',
)
_SAVE_CLASSES = ('SaveAnimatedWEBP', 'SaveWEBM', 'VHS_VideoCombine', 'SaveAnimatedPNG')

_SAMPLER_INPUTS = {
    'seed': 'seed', 'noise_seed': 'seed', 'steps': 'steps', 'cfg': 'cfg',
    'denoise': 'denoise', 'sampler_name': 'sampler', 'scheduler': 'scheduler',
}
_LATENT_INPUTS = {'width': 'width', 'height': 'height', 'length': 'frames', 'batch_size': 'batch'}
_SAVE_INPUTS = {'fps': 'fps', 'frame_rate': 'fps'}


def tokenise(graph: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Turn a ComfyUI API export into a template. Returns it and what it found.

    The prompt is the hard part, because both the positive and the negative
    conditioning are the same node class with the same input name — the only
    thing distinguishing them is which sampler input they are wired to. So the
    sampler is found first and its `positive` and `negative` links are
    followed, which is exactly how ComfyUI itself tells them apart. A graph
    whose sampler cannot be found gets its prompts left alone and says so,
    rather than guessing and putting the prompt in the negative.
    """
    graph = copy.deepcopy(graph)
    found: list[str] = []

    def node_id_of(link: Any) -> str | None:
        # A wired input is ["<node id>", <output index>].
        if isinstance(link, list) and link and isinstance(link[0], (str, int)):
            return str(link[0])
        return None

    for node in graph.values():
        if not isinstance(node, dict):
            continue
        cls = node.get('class_type', '')
        inputs = node.get('inputs') or {}

        if cls in _SAMPLER_CLASSES:
            for key, token in _SAMPLER_INPUTS.items():
                if key in inputs and not isinstance(inputs[key], list):
                    inputs[key] = f'%{token}%'
                    found.append(token)

            for side, token in (('positive', 'prompt'), ('negative', 'negative')):
                target = node_id_of(inputs.get(side))
                text_node = graph.get(target) if target else None
                if isinstance(text_node, dict) and 'text' in (text_node.get('inputs') or {}):
                    text_node['inputs']['text'] = f'%{token}%'
                    found.append(token)

        if cls in _LATENT_CLASSES:
            for key, token in _LATENT_INPUTS.items():
                if key in inputs and not isinstance(inputs[key], list):
                    inputs[key] = f'%{token}%'
                    found.append(token)

        if cls in _SAVE_CLASSES:
            for key, token in _SAVE_INPUTS.items():
                if key in inputs and not isinstance(inputs[key], list):
                    inputs[key] = f'%{token}%'
                    found.append(token)

    return graph, sorted(set(found))


def install(
    directory: Path,
    name: str,
    graph: dict[str, Any],
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a ComfyUI export into the template directory, tokenising it.

    A module function rather than a provider method, and that is the point:
    this touches the disk and never the server. It used to be reached by
    constructing a `ComfyUIProvider` pointed at a hard-coded loopback address
    purely to call it — an object that looked like a connection and was not
    the configured one. Nothing leaked, because nothing was sent; but the day
    someone adds "check the graph against the server" to it, that call would
    go to whatever is on local 8188 with no token, and on an install whose
    ComfyUI is elsewhere it would be talking to a stranger.

    The tokens it found come back, because that is the only useful
    confirmation: "imported" says nothing, and "found prompt, seed, steps,
    width, height, frames" tells you at a glance whether it will be driveable
    from a form.
    """
    safe = re.sub(r'[^a-zA-Z0-9._-]', '-', name).strip('-') or 'workflow'
    directory.mkdir(parents=True, exist_ok=True)
    template, found = tokenise(graph)

    path = directory / f'{safe}.json'
    path.write_text(json.dumps(template, indent=2))
    if overrides:
        path.with_suffix('.roost.json').write_text(json.dumps(overrides, indent=2))
    log.info('imported workflow %s with tokens: %s', safe, ', '.join(found) or 'none')
    return {'id': safe, 'tokens': found, 'path': str(path)}


def load(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """A template and its sidecar overrides, if it has any."""
    graph = json.loads(path.read_text())
    sidecar = path.with_suffix('.roost.json')
    overrides: dict[str, Any] = {}
    if sidecar.is_file():
        try:
            overrides = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning('ignoring %s: %s', sidecar.name, exc)
    return graph, overrides
