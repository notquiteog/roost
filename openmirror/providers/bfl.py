"""FLUX, from the people who trained it.

Black Forest Labs is worth a direct adapter rather than reaching FLUX through
an aggregator, for two reasons that only apply to the first party: the newest
FLUX variant appears here first, and `flux-pro-1.1-ultra` is only offered at
4MP by BFL itself. The aggregators catch up, but "catch up" is a week.

The API is unusually plain — submit, poll a URL it hands back, fetch a signed
sample — and two of its details are the sort that cost an afternoon:

**The polling URL must be the one it gave you.** BFL routes a job to a region
and returns a polling URL in that region; rebuilding one from the id against
the global host finds nothing, intermittently, depending on where the job
landed. So the returned URL is used verbatim.

**A moderated request is a distinct status, not an error.** `Request
Moderated` means the prompt was refused and `Content Moderated` means the
image was, and both come back as an ordinary 200. Reporting them as "the job
failed" tells someone to retry the thing that will be refused again, so they
are surfaced in the provider's own words.

The result URL expires about ten minutes after it is signed, which is the real
argument for this build downloading into its own store rather than keeping a
link: a gallery of BFL URLs is a gallery that empties itself overnight.
"""

from __future__ import annotations

from typing import Any

from openmirror.media.params import Param, common_image
from openmirror.providers.hosted_media import HostedMedia, Poll, Submitted

API = 'https://api.bfl.ai/v1'

#: What BFL serves, and what each is for. Kept here rather than asked for
#: because BFL publishes no listing endpoint — so this is documentation, and
#: the model box takes anything, including an endpoint added after this line.
MODELS: tuple[tuple[str, str], ...] = (
    ('flux-pro-1.1-ultra', 'Up to 4MP, and the only one of these that takes `raw` for a less '
                           'processed, more photographic look.'),
    ('flux-kontext-max', 'Editing: give it an image and say what to change. The strongest at '
                         'keeping a character the same across edits.'),
    ('flux-kontext-pro', 'The same editing model, faster and cheaper.'),
    ('flux-pro-1.1', 'The general-purpose one. Six times faster than flux-pro at similar quality.'),
    ('flux-pro', 'The original pro endpoint, kept because some prompts were tuned against it.'),
    ('flux-dev', 'Open weights, guidance-distilled — the same model you can run on your own card.'),
)

#: `ultra` takes an aspect ratio and the rest take pixel dimensions. Offering
#: both to both is how someone sets a width that is silently ignored.
ULTRA = ('flux-pro-1.1-ultra',)
EDITING = ('flux-kontext-max', 'flux-kontext-pro')


class BFLProvider(HostedMedia):
    provider_id = 'bfl'

    def __init__(self, base_url: str = API, api_key: str = '', **kw: Any) -> None:
        super().__init__(base_url or API, api_key, **kw)

    def headers(self) -> dict[str, str]:
        # `x-key`, which is BFL's own spelling and not a bearer token.
        h = {'Content-Type': 'application/json'}
        if self.api_key:
            h['x-key'] = self.api_key
        return h

    async def submit(self, prompt: str, *, model: str, **kw: Any) -> Submitted:
        model = (model or 'flux-pro-1.1').strip()
        payload: dict[str, Any] = {'prompt': prompt}

        for key, value in kw.items():
            if value is None or value == '':
                continue
            # A reference image goes in as bare base64 — no data: prefix,
            # which BFL rejects.
            if isinstance(value, bytes | bytearray):
                import base64

                payload[key] = base64.b64encode(value).decode()
            else:
                payload[key] = value
        if payload.get('seed') in (-1, '-1'):
            payload.pop('seed')

        body = await self.post_json(f'{self.base_url}/{model}', payload)
        polling = body.get('polling_url') or ''
        if not polling and body.get('id'):
            polling = f'{self.base_url}/get_result?id={body["id"]}'
        return Submitted(id=str(body.get('id', '')), poll_url=polling)

    async def check(self, job: Submitted) -> Poll:
        body = await self.get_json(job.poll_url)
        status = str(body.get('status', ''))

        if status == 'Ready':
            sample = (body.get('result') or {}).get('sample')
            return Poll(state='done', progress=1.0, outputs=[sample] if sample else [])

        if status in ('Request Moderated', 'Content Moderated'):
            which = 'prompt' if status.startswith('Request') else 'generated image'
            return Poll(
                state='failed',
                error=f'BFL refused this: the {which} was moderated. Retrying it unchanged will '
                      'be refused again.',
            )
        if status in ('Error', 'Task not found'):
            detail = body.get('details') or body.get('error') or status
            return Poll(state='failed', error=f'BFL: {detail}')

        progress = body.get('progress')
        return Poll(
            state='running' if status == 'Pending' else 'queued',
            progress=float(progress) if isinstance(progress, int | float) else None,
            note='generating at BFL',
        )

    async def generate(self, prompt: str, *, model: str = '', n: int = 1, size: str = '',
                       progress: Any = None, **kw: Any) -> list[Any]:
        """One image per request, so several are several requests.

        BFL takes no batch size. Run in sequence rather than at once: these
        keys are rate-limited per account and a burst of six is how a request
        for six images returns two images and four 429s.
        """
        out: list[Any] = []
        for index in range(max(1, int(n or 1))):
            step = (lambda fraction, note, i=index: progress(
                ((i + (fraction or 0)) / max(1, n)), f'{note} ({i + 1} of {n})'
            )) if progress and n > 1 else progress
            out += await self.run(prompt, model=model or 'flux-pro-1.1', progress=step, **kw)
        return out

    async def models(self, kind: str = '') -> list[dict[str, Any]]:
        if kind == 'video':
            return []
        return [{'id': ident, 'provider': self.provider_id, 'label': ident, 'note': note}
                for ident, note in MODELS]

    async def describe(self, model: str = '') -> list[Param]:
        model = (model or 'flux-pro-1.1').strip()
        params = [p for p in common_image() if p.name != 'negative_prompt']

        if model in ULTRA:
            params.append(Param('aspect_ratio', 'Aspect ratio', 'enum', default='16:9',
                                options=['21:9', '16:9', '4:3', '1:1', '3:4', '9:16', '9:21'],
                                group='shape',
                                help='Ultra sizes by ratio rather than by pixels, and fills 4MP whichever you pick.'))
            params.append(Param('raw', 'Raw mode', 'bool', default=False, group='quality',
                                help='Less processed and more photographic — closer to a snapshot '
                                     'and further from an illustration.'))
        else:
            params += [
                Param('width', 'Width', 'int', default=1024, minimum=256, maximum=1440, step=32, group='shape',
                      help='Multiples of 32.'),
                Param('height', 'Height', 'int', default=768, minimum=256, maximum=1440, step=32, group='shape'),
            ]

        if model in EDITING:
            params.append(Param('input_image', 'Image to edit', 'image', group='prompt',
                                help='Kontext edits what you give it. Say what to change rather than '
                                     'describing the whole picture again — "make the jacket red", not '
                                     '"a man in a red jacket".'))
        else:
            params.append(Param('image_prompt', 'Reference image', 'image', group='prompt', advanced=True,
                                help='Blended with the text prompt as a visual reference.'))

        params += [
            Param('prompt_upsampling', 'Expand the prompt', 'bool', default=False, group='prompt',
                  advanced=True,
                  help='Has the model rewrite your prompt into a longer one first. Helps a short '
                       'prompt and gets in the way of a carefully written one.'),
            Param('safety_tolerance', 'Safety tolerance', 'int', default=2, minimum=0, maximum=6, step=1,
                  group='output', advanced=True,
                  help='0 is strictest, 6 most permissive. Editing endpoints cap this at 2.'),
            Param('output_format', 'File format', 'enum', default='jpeg', options=['jpeg', 'png'],
                  group='output', advanced=True),
        ]
        return params
