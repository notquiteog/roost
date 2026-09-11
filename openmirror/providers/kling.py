"""Kling, from Kuaishou — the one that understands how things fall.

Kling is in this build for physical motion. Its 2.x models hold up under the
things that break most video models: a person walking without their legs
swapping, water behaving like water, a hand that stays a hand while it picks
something up. It also takes a *tail* image — the frame to end on — and a
structured camera control, both of which are lost when it is reached through
an aggregator.

Its authentication is the one genuinely unusual thing here and the reason this
is a file rather than a catalogue entry: **Kling wants a signed JWT, not an
API key.** You are issued an access key and a secret, and every request
carries a token signed with the secret that expires in half an hour. That is
only three lines of HMAC, which is why there is no dependency here for it —
but it does mean the credential for this connection is two values, so it is
entered as `access_key:secret_key` and split here.

The token is minted per request rather than cached. It costs a SHA-256 over
about a hundred bytes, and a cached token is a token that expires in the
middle of a four-minute poll — which fails as an authentication error on a
job that was running perfectly well.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any

from openmirror.media.params import Param
from openmirror.providers.hosted_media import HostedMedia, Poll, Submitted, urls_in

log = logging.getLogger(__name__)

API = 'https://api-singapore.klingai.com'

#: How long a minted token is good for. Kling's own examples use half an hour;
#: this is shorter because nothing here needs a long-lived credential and a
#: token that outlives the request it was made for is a token in a log.
TOKEN_LIFETIME = 1800

VIDEO_MODELS: tuple[tuple[str, str], ...] = (
    ('kling-v2-1-master', 'Kling 2.1 Master — the best of these at motion that obeys physics.'),
    ('kling-v2-master', 'Kling 2.0 Master.'),
    ('kling-v2-1', 'Kling 2.1 — cheaper, and still ahead of most of the field on motion.'),
    ('kling-v1-6', 'Kling 1.6 — the workhorse. Takes a tail image for a shot that ends somewhere decided.'),
    ('kling-v1-5', 'Kling 1.5.'),
    ('kling-v1', 'Kling 1.0 — the original, kept for prompts tuned against it.'),
)

IMAGE_MODELS: tuple[tuple[str, str], ...] = (
    ('kling-v2', 'Kling Image 2.'),
    ('kling-v1-5', 'Kling Image 1.5.'),
    ('kling-v1', 'Kling Image 1.'),
)


def mint(access_key: str, secret: str, *, lifetime: int = TOKEN_LIFETIME) -> str:
    """A signed JWT, by hand.

    HS256 is a base64url header, a base64url payload and an HMAC-SHA256 of the
    two joined by a dot. Writing it out is shorter than the argument for
    adding a JWT library to a project that needs exactly this one algorithm in
    exactly this one place.

    `nbf` is set slightly in the past on purpose: the clock here and the clock
    at Kling are not the same clock, and a token that is not yet valid by two
    seconds is rejected in a way that reads as a bad secret.
    """
    def segment(data: dict[str, Any]) -> bytes:
        raw = json.dumps(data, separators=(',', ':'), sort_keys=True).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b'=')

    now = int(time.time())
    header = segment({'alg': 'HS256', 'typ': 'JWT'})
    payload = segment({'iss': access_key, 'exp': now + lifetime, 'nbf': now - 5})
    signing_input = header + b'.' + payload
    signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return (signing_input + b'.' + base64.urlsafe_b64encode(signature).rstrip(b'=')).decode()


class KlingProvider(HostedMedia):
    provider_id = 'kling'

    def __init__(self, base_url: str = API, api_key: str = '', *, kind: str = 'video', **kw: Any) -> None:
        super().__init__(base_url or API, api_key, **kw)
        # Kling is the one service here whose image and video models share
        # ids — `kling-v1-5` is both — so the endpoint cannot be chosen from
        # the model name. It is fixed when the provider is built instead, and
        # one connection registers two of these: see openmirror.providers.connections.
        self.kind = kind

    @property
    def credentials(self) -> tuple[str, str]:
        """The access key and the secret, from the one field a connection has.

        Split rather than two fields because every other provider in this
        build takes a single credential, and a second field on the connection
        form for the one service that needs it would be a field that is empty
        and confusing everywhere else.
        """
        raw = (self.api_key or '').strip()
        if ':' not in raw:
            raise RuntimeError(
                'kling: the credential is two values. Enter it as access_key:secret_key — both are '
                'on the API keys page of the Kling console.'
            )
        access, secret = raw.split(':', 1)
        return access.strip(), secret.strip()

    def headers(self) -> dict[str, str]:
        access, secret = self.credentials
        return {'Content-Type': 'application/json', 'Authorization': f'Bearer {mint(access, secret)}'}

    # -- submitting ---------------------------------------------------------

    async def submit(self, prompt: str, *, model: str, **kw: Any) -> Submitted:
        model = (model or ('kling-v2' if self.kind == 'image' else 'kling-v2-1-master')).strip()
        payload: dict[str, Any] = {'model_name': model, 'prompt': prompt}

        for key in ('negative_prompt', 'cfg_scale', 'mode', 'aspect_ratio', 'duration', 'image_fidelity', 'n'):
            if kw.get(key) not in (None, ''):
                payload[key] = kw[key]

        start = kw.get('image')
        tail = kw.get('image_tail')
        for field, value in (('image', start), ('image_tail', tail)):
            if isinstance(value, bytes | bytearray):
                # Bare base64, no data: prefix — Kling rejects the prefix.
                payload[field] = base64.b64encode(bytes(value)).decode()

        # Camera movement is a nested object and only meaningful on the video
        # path. Sent only when something was actually chosen, because
        # `type: ""` is a 400 rather than a default.
        camera = kw.get('camera_control')
        if camera and camera != 'none':
            payload['camera_control'] = {'type': camera}

        if self.kind == 'image':
            path = '/v1/images/generations'
        else:
            path = '/v1/videos/image2video' if 'image' in payload else '/v1/videos/text2video'

        body = await self.post_json(f'{self.base_url}{path}', payload)
        data = body.get('data') or {}
        if body.get('code') not in (0, None):
            raise RuntimeError(f'kling: {body.get("message") or body.get("code")}')
        task = str(data.get('task_id', ''))
        return Submitted(id=task, poll_url=f'{self.base_url}{path}/{task}')

    async def check(self, job: Submitted) -> Poll:
        body = await self.get_json(job.poll_url)
        data = body.get('data') or {}
        status = str(data.get('task_status', '')).lower()

        if status == 'succeed':
            result = data.get('task_result') or {}
            return Poll(state='done', progress=1.0, outputs=urls_in(result))
        if status == 'failed':
            return Poll(state='failed',
                        error=f'kling: {data.get("task_status_msg") or "the task failed without saying why"}')
        return Poll(
            state='running' if status == 'processing' else 'queued',
            note='generating at Kling' if status == 'processing' else 'queued at Kling',
        )

    async def generate(self, prompt: str, *, model: str = '', n: int = 1, size: str = '',
                       progress: Any = None, **kw: Any) -> list[Any]:
        default = 'kling-v2' if self.kind == 'image' else 'kling-v2-1-master'
        return await self.run(prompt, model=model or default, progress=progress, **kw)

    async def models(self, kind: str = '') -> list[dict[str, Any]]:
        rows = IMAGE_MODELS if (kind or self.kind) == 'image' else VIDEO_MODELS
        return [{'id': i, 'provider': self.provider_id, 'label': i, 'note': n} for i, n in rows]

    async def describe(self, model: str = '') -> list[Param]:
        if self.kind == 'image':
            from openmirror.media.params import common_image

            return [p for p in common_image() if p.name != 'seed'] + [
                Param('aspect_ratio', 'Aspect ratio', 'enum', default='16:9',
                      options=['16:9', '9:16', '1:1', '4:3', '3:4', '3:2', '2:3', '21:9'], group='shape'),
                Param('image', 'Reference image', 'image', group='prompt', advanced=True,
                      help='Kling calls this image reference; it keeps a subject or a style.'),
                Param('image_fidelity', 'Reference strength', 'float', default=0.5, minimum=0.0,
                      maximum=1.0, step=0.05, group='sampling', advanced=True,
                      help='How much of the reference to keep.'),
            ]

        return [
            Param('prompt', 'Prompt', 'text', group='prompt',
                  help='Kling rewards describing the motion and the physics — weight, momentum, '
                       'what touches what — rather than only the scene.'),
            Param('negative_prompt', 'Negative prompt', 'text', group='prompt', advanced=True),
            Param('image', 'Starting frame', 'image', group='prompt',
                  help='Supplying one switches this to image-to-video.'),
            Param('image_tail', 'Ending frame', 'image', group='prompt', advanced=True,
                  help='The frame to arrive at. With both, Kling makes the shot between them — '
                       'which is the feature worth coming here for.'),
            Param('aspect_ratio', 'Aspect ratio', 'enum', default='16:9', options=['16:9', '9:16', '1:1'],
                  group='shape'),
            Param('duration', 'Duration', 'enum', default='5', options=['5', '10'], group='motion',
                  help='Seconds.'),
            Param('mode', 'Mode', 'enum', default='std', options=['std', 'pro'], group='quality',
                  help='`pro` is slower, more expensive and noticeably steadier on complex motion.'),
            Param('cfg_scale', 'Prompt adherence', 'float', default=0.5, minimum=0.0, maximum=1.0, step=0.05,
                  group='sampling',
                  help='0 to 1 here, not the 1-to-30 scale diffusion models use. Higher follows the '
                       'prompt harder and moves less.'),
            Param('camera_control', 'Camera', 'enum', default='none',
                  options=['none', 'simple', 'down_back', 'forward_up', 'right_turn_forward',
                           'left_turn_forward'],
                  group='motion', advanced=True,
                  help='A named camera move, which Kling applies rather than inferring from the prompt.'),
        ]
