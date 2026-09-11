"""A model's own input schema, rendered as controls.

Replicate and fal both publish OpenAPI for every endpoint they host, which is
the difference between "we support a few models" and "we support the field".
A schema carries the real ranges, the real enum values, the author's own
description of each knob and the order they meant them to be read in — so a
form built from it is better than one somebody would have typed here, and it
covers a model published this morning.

So this is one translation, used by both, because two would be two sets of the
same bugs. Four of them are worth naming, since each turns a generated form
from useful into actively misleading:

**An enum arrives as a reference.** `allOf: [{$ref: …}]` pointing at a
component that holds the values. A reader that only looks at `type` sees a
string and renders a text box — and a text box for `aspect_ratio` is a field
in which `16:9` and `16x9` look equally plausible and one of them is a 422.

**A file input arrives as `format: uri`.** That is how an image-to-image or
image-to-video model asks for a picture. Rendered as text, the model looks
broken rather than unsupplied.

**A type nobody recognises is left out, not guessed.** A control that sends
the wrong shape produces a 422 the person cannot connect to anything they
touched. A missing control is merely missing, and the API stays open.

**What is required is never folded away.** `advanced` hides a control behind a
disclosure, and a form whose one mandatory field is hidden is a form that
fails on submit for no visible reason.
"""

from __future__ import annotations

from typing import Any

from openmirror.media.params import Param

#: Which section a field belongs in, by the names these models actually use.
#: Matched on the name because no schema says anything about grouping, and an
#: ungrouped form of thirty controls is a wall rather than a panel.
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('prompt', ('prompt', 'negative_prompt', 'system_prompt', 'prompt_2', 'style_preset', 'style',
                'style_id', 'magic_prompt', 'magic_prompt_option', 'enhance_prompt')),
    ('shape', ('width', 'height', 'aspect_ratio', 'size', 'image_size', 'megapixels', 'resolution')),
    ('motion', ('num_frames', 'frames', 'fps', 'frame_rate', 'duration', 'duration_seconds',
                'motion_bucket_id', 'camera', 'camera_motion', 'camera_fixed', 'loop',
                'interpolate', 'motion', 'cond_aug')),
    ('sampling', ('seed', 'num_inference_steps', 'steps', 'guidance_scale', 'guidance', 'cfg',
                  'cfg_scale', 'scheduler', 'sampler', 'shift', 'denoise', 'strength',
                  'prompt_strength', 'image_prompt_strength')),
    ('output', ('num_outputs', 'num_images', 'output_format', 'output_quality', 'compression',
                'go_fast', 'disable_safety_checker', 'enable_safety_checker', 'safety_tolerance',
                'sync_mode', 'raw')),
)

#: Open on first sight of a model. Everything else folds away — a disclosure,
#: never a restriction; see the note on `advanced` in openmirror.media.params.
COMMON = frozenset({
    'prompt', 'negative_prompt', 'aspect_ratio', 'image_size', 'width', 'height',
    'num_outputs', 'num_images', 'duration', 'resolution', 'image', 'image_url',
    'start_image', 'input_image', 'image_prompt', 'first_frame_image', 'reference_images',
})


def input_schema(doc: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The input object of an OpenAPI document, and the components to resolve
    references against.

    Two layouts are in circulation. Replicate names the input schema `Input`
    outright. fal names it after the endpoint and only references it from the
    POST body, so it has to be found by following the path — which is done
    second, because looking for the conventional name first keeps the common
    case one dictionary lookup.
    """
    components = (doc.get('components') or {}).get('schemas') or {}
    if 'Input' in components:
        return components['Input'] or {}, components

    for methods in (doc.get('paths') or {}).values():
        if not isinstance(methods, dict):
            continue
        body = ((methods.get('post') or {}).get('requestBody') or {})
        schema = (((body.get('content') or {}).get('application/json') or {}).get('schema') or {})
        ref = schema.get('$ref')
        if isinstance(ref, str) and ref.startswith('#/components/schemas/'):
            return components.get(ref.rsplit('/', 1)[-1]) or {}, components
        if schema.get('properties'):
            return schema, components
    return {}, components


def params_from_openapi(doc: dict[str, Any]) -> list[Param]:
    """Turn one endpoint's schema into the form for it."""
    spec, components = input_schema(doc)
    properties: dict[str, Any] = spec.get('properties') or {}
    required = set(spec.get('required') or [])

    def resolve(field: dict[str, Any]) -> dict[str, Any]:
        holders = list(field.get('allOf') or [])
        if '$ref' in field:
            holders.append(field)
        for holder in holders:
            ref = holder.get('$ref') if isinstance(holder, dict) else None
            if not isinstance(ref, str) or not ref.startswith('#/components/schemas/'):
                continue
            target = components.get(ref.rsplit('/', 1)[-1]) or {}
            # The reference carries the values; the field carries the default,
            # the title and the description, and the field's version wins —
            # a model that re-describes an enum locally meant that description.
            return {**target, **{k: v for k, v in field.items() if k not in ('allOf', '$ref')}}
        return field

    # `x-order` is the author's own ordering, which is nearly always more
    # sensible than alphabetical: prompt first, then the thing you change next.
    ordered = sorted(properties.items(), key=lambda kv: (_order(kv[1]), kv[0]))
    out: list[Param] = []
    for name, raw in ordered:
        param = param_for(name, resolve(raw if isinstance(raw, dict) else {}), required=name in required)
        if param is not None:
            out.append(param)
    return out


def _order(field: Any) -> float:
    if not isinstance(field, dict):
        return 999
    for key in ('x-order', 'x-fal-order'):
        value = field.get(key)
        if isinstance(value, int | float):
            return float(value)
    return 999


def param_for(name: str, field: dict[str, Any], *, required: bool = False) -> Param | None:
    kind = kind_for(name, field)
    if kind is None:
        return None

    options = [str(v) for v in (field.get('enum') or []) if v is not None]
    default = field.get('default')
    if isinstance(default, dict | list) and kind != 'image':
        # A structured default cannot go in a text box, and putting its JSON
        # there would have somebody edit a string that the API wants typed.
        default = None
    if kind == 'enum' and default is None and options:
        default = options[0]

    return Param(
        name=name,
        label=label(field.get('title') or name),
        kind=kind,
        default=default,
        minimum=field.get('minimum'),
        maximum=field.get('maximum'),
        step=step(kind, field),
        options=options,
        help=' '.join(str(field.get('description') or '').split())[:400],
        advanced=not required and name not in COMMON,
        group=group_for(name),
    )


def kind_for(name: str, field: dict[str, Any]) -> str | None:
    if field.get('enum'):
        return 'enum'
    declared = field.get('type')
    if isinstance(declared, list):
        # `["string", "null"]` — an optional field, written the JSON Schema way.
        declared = next((t for t in declared if t != 'null'), None)
    if declared == 'integer':
        return 'seed' if name == 'seed' else 'int'
    if declared == 'number':
        return 'float'
    if declared == 'boolean':
        return 'bool'
    if declared == 'array':
        items = field.get('items') or {}
        if items.get('format') == 'uri' or 'image' in name:
            return 'image'
        return None
    if declared == 'string':
        if field.get('format') == 'uri' or name.endswith(('_url', '_image')) or name in ('image', 'video'):
            return 'image'
        long_form = (
            name in ('prompt', 'negative_prompt')
            or len(str(field.get('description') or '')) > 90
            or (field.get('maxLength') or 0) > 200
        )
        return 'text' if long_form else 'string'
    return None


def group_for(name: str) -> str:
    for group, names in GROUPS:
        if name in names:
            return group
    if any(word in name for word in ('image', 'video', 'mask', 'reference', 'prompt')):
        return 'prompt'
    return 'quality'


def step(kind: str, field: dict[str, Any]) -> float | None:
    if kind == 'int':
        return 1
    if kind != 'float':
        return None
    low, high = field.get('minimum'), field.get('maximum')
    if low is None or high is None:
        return 0.1
    span = float(high) - float(low)
    # A slider from 0 to 1 wants hundredths; one from 1 to 50 does not.
    return 0.01 if span <= 2 else (0.1 if span <= 100 else 1)


def label(raw: str) -> str:
    return str(raw).replace('_', ' ').strip().capitalize()
