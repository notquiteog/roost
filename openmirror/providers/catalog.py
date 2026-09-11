"""The endpoints and embedding models this knows about by name.

Nothing here is a capability. It is a list of *addresses and defaults*, so
that "connect Groq" is a token and a click rather than a base URL somebody has
to find, and so that a person who wants Qwen3-Embedding-8B is told which of
their servers can serve it instead of guessing at an id.

Two rules keep this from rotting into a lie.

**A catalogue entry never decides what a provider can do.** What models exist
is asked of the provider, every time, over the wire — see `models()` on each
adapter. This file only ever supplies *where to ask* and, for the handful of
services with no model listing at all, a documented fallback that is clearly
labelled as not having come from the server.

**A preset is a starting point, not a constraint.** Every field is editable
when a connection is made, because these URLs change and an install pinned to
whatever was true when this file was written is worse than no preset at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from openmirror.providers.base import Modality

#: Which adapter class answers for a host. A shape, not a company: `openai`
#: covers every service that copied that request format, which is most of them.
Adapter = Literal[
    'openai', 'anthropic', 'ollama', 'a1111', 'comfyui', 'voyage', 'google',
    # One host that becomes up to five providers. See openmirror.providers.perch.
    'perch',
    # Image and video, which is where the shapes stop being interchangeable.
    # Every one of these is a submit-poll-download job API that agrees with
    # the others about nothing except that structure — see
    # openmirror.providers.hosted_media, which is the part they do share.
    'replicate', 'fal', 'bfl', 'stability', 'runway', 'luma', 'kling', 'minimax', 'ideogram',
]


@dataclass(frozen=True, slots=True)
class Host:
    """A service someone might connect."""

    id: str
    label: str
    adapter: Adapter
    base_url: str
    env_key: str = ''
    modalities: frozenset[Modality] = field(default_factory=frozenset)
    local: bool = False
    lists_models: bool = True
    note: str = ''
    fallback_models: tuple[str, ...] = ()


CHAT = frozenset({Modality.CHAT})
EMBEDDING = frozenset({Modality.EMBEDDING})
CHAT_AND_EMBED = CHAT | EMBEDDING
STT = frozenset({Modality.STT})
TTS = frozenset({Modality.TTS})
IMAGE = frozenset({Modality.IMAGE})
VIDEO = frozenset({Modality.VIDEO})


HOSTS: tuple[Host, ...] = (
    # First, so it is what the Add form starts on: an install with only Perch
    # configured is a complete install, and this project's own GPU host is the
    # connection most people opening that form have come to make.
    Host(
        id='perch',
        label='Perch',
        adapter='perch',
        base_url='http://127.0.0.1',
        env_key='PERCH_TOKEN',
        modalities=CHAT_AND_EMBED | STT | TTS | IMAGE | VIDEO,
        local=True,
        note=(
            'Your own GPU host, as one connection. The address is the host only — its '
            'services are found on their usual ports (chat 11434, dictation 8080, images '
            '7860, video 8188, speech 8880) and whichever are switched on are connected. '
            'The key is a token from Perch\'s console, and it is checked before anything '
            'is saved.'
        ),
    ),
    Host(
        id='openai',
        label='OpenAI',
        adapter='openai',
        base_url='https://api.openai.com/v1',
        env_key='OPENAI_API_KEY',
        modalities=CHAT_AND_EMBED | STT | TTS | IMAGE | VIDEO,
        note='Chat, embeddings, speech both ways, images, and Sora for video.',
    ),
    Host(
        id='anthropic',
        label='Anthropic',
        adapter='anthropic',
        base_url='https://api.anthropic.com/v1',
        env_key='ANTHROPIC_API_KEY',
        modalities=CHAT,
        note='Chat only — Anthropic publishes no embedding model. Pair it with a separate embedding connection.',
    ),
    Host(
        id='groq',
        label='Groq',
        adapter='openai',
        base_url='https://api.groq.com/openai/v1',
        env_key='GROQ_API_KEY',
        modalities=STT | CHAT,
        note='OpenAI-shaped. Also serves whisper on the transcriptions path.',
    ),
    Host(
        id='openrouter',
        label='OpenRouter',
        adapter='openai',
        base_url='https://openrouter.ai/api/v1',
        env_key='OPENROUTER_API_KEY',
        modalities=CHAT_AND_EMBED,
        note='A gateway to many vendors, so its model list is long and worth searching rather than scrolling.',
    ),
    Host(
        id='fireworks',
        label='Fireworks AI',
        adapter='openai',
        base_url='https://api.fireworks.ai/inference/v1',
        env_key='FIREWORKS_API_KEY',
        modalities=CHAT_AND_EMBED | IMAGE,
        note='Model ids are paths — accounts/fireworks/models/…. Serves FLUX and SDXL on the images path.',
    ),
    Host(
        id='together',
        label='Together AI',
        adapter='openai',
        base_url='https://api.together.xyz/v1',
        env_key='TOGETHER_API_KEY',
        modalities=CHAT_AND_EMBED | IMAGE,
        note='Serves open-weight embedding models, Qwen3-Embedding included.',
    ),
    Host(
        id='nanogpt',
        label='NanoGPT',
        adapter='openai',
        base_url='https://nano-gpt.com/api/v1',
        env_key='NANOGPT_API_KEY',
        modalities=CHAT_AND_EMBED,
    ),
    Host(
        id='deepinfra',
        label='DeepInfra',
        adapter='openai',
        base_url='https://api.deepinfra.com/v1/openai',
        env_key='DEEPINFRA_API_KEY',
        modalities=CHAT_AND_EMBED | IMAGE,
    ),
    Host(
        id='cerebras',
        label='Cerebras',
        adapter='openai',
        base_url='https://api.cerebras.ai/v1',
        env_key='CEREBRAS_API_KEY',
        modalities=CHAT,
    ),
    Host(
        id='mistral',
        label='Mistral',
        adapter='openai',
        base_url='https://api.mistral.ai/v1',
        env_key='MISTRAL_API_KEY',
        modalities=CHAT_AND_EMBED,
    ),
    Host(
        id='xai',
        label='xAI',
        adapter='openai',
        base_url='https://api.x.ai/v1',
        env_key='XAI_API_KEY',
        modalities=CHAT | IMAGE,
        note='Grok for chat, and grok-2-image on the OpenAI-shaped images path.',
    ),
    Host(
        id='google',
        label='Google AI',
        adapter='google',
        base_url='https://generativelanguage.googleapis.com/v1beta',
        env_key='GEMINI_API_KEY',
        modalities=CHAT_AND_EMBED | IMAGE | VIDEO,
        note='Gemini and its embedding models, Imagen and the conversational image model for '
             'stills, and Veo for video — all on the one key.',
    ),
    Host(
        id='voyage',
        label='Voyage AI',
        adapter='voyage',
        base_url='https://api.voyageai.com/v1',
        env_key='VOYAGE_API_KEY',
        modalities=EMBEDDING,
        lists_models=False,
        note='Embeddings only, and it publishes no model-listing endpoint — so the ids offered here come from its documentation, not from the service. An id it does not know fails at first use.',
        fallback_models=('voyage-3-large', 'voyage-3.5', 'voyage-3.5-lite', 'voyage-code-3', 'voyage-law-2', 'voyage-finance-2'),
    ),
    Host(
        id='ollama',
        label='Ollama',
        adapter='ollama',
        base_url='http://127.0.0.1:11434',
        env_key='OLLAMA_API_KEY',
        modalities=CHAT_AND_EMBED,
        local=True,
        note='Local or remote — it is the same API either way. A remote one is only "local" in the privacy sense if the machine is yours, which is why that stays a tick box.',
    ),
    Host(
        id='lmstudio',
        label='LM Studio',
        adapter='openai',
        base_url='http://127.0.0.1:1234/v1',
        modalities=CHAT_AND_EMBED,
        local=True,
    ),
    Host(
        id='vllm',
        label='vLLM',
        adapter='openai',
        base_url='http://127.0.0.1:8000/v1',
        modalities=CHAT_AND_EMBED,
        local=True,
        note='Serves one model per process, so its list is usually one long.',
    ),
    Host(
        id='llamacpp',
        label='llama.cpp server',
        adapter='openai',
        base_url='http://127.0.0.1:8080/v1',
        modalities=CHAT_AND_EMBED,
        local=True,
    ),
    Host(
        id='openwebui',
        label='Open WebUI',
        adapter='openai',
        base_url='http://127.0.0.1:8080/api/v1',
        env_key='OPENWEBUI_API_KEY',
        modalities=CHAT_AND_EMBED,
        note='Every connection configured over there, through one entry here.',
    ),
    Host(
        id='automatic1111',
        label='Stable Diffusion (AUTOMATIC1111)',
        adapter='a1111',
        base_url='http://127.0.0.1:7860',
        modalities=IMAGE,
        local=True,
    ),
    # ---- image and video, first party ------------------------------------
    #
    # These are here rather than left to the two aggregators below because in
    # each case the first party has something the resellers do not: the newest
    # model first, a feature the passthrough drops, or both. Where that is not
    # true, there is no entry.
    Host(
        id='replicate',
        label='Replicate',
        adapter='replicate',
        base_url='https://api.replicate.com/v1',
        env_key='REPLICATE_API_TOKEN',
        modalities=IMAGE | VIDEO,
        note='Most of the field behind one key — and it publishes each model\'s input schema, so '
             'the controls for a model this build has never heard of are still the right ones, with '
             'its real ranges and its author\'s own descriptions. The model box takes any owner/name '
             'from replicate.com, including one published this morning.',
    ),
    Host(
        id='fal',
        label='fal',
        adapter='fal',
        base_url='https://queue.fal.run',
        env_key='FAL_KEY',
        modalities=IMAGE | VIDEO,
        lists_models=False,
        note='The other route to most of the same models, which is worth having: a model being down '
             'or rate-limited on one of these is not the end of the road. fal publishes no listing '
             'endpoint, so the models offered come from its documentation — any endpoint path can be '
             'typed in, and the form is still built from fal\'s own schema for it.',
        fallback_models=('fal-ai/flux-pro/v1.1-ultra', 'fal-ai/flux/dev', 'fal-ai/nano-banana',
                         'fal-ai/veo3', 'fal-ai/kling-video/v2/master/text-to-video', 'fal-ai/wan-t2v'),
    ),
    Host(
        id='bfl',
        label='Black Forest Labs (FLUX)',
        adapter='bfl',
        base_url='https://api.bfl.ai/v1',
        env_key='BFL_API_KEY',
        modalities=IMAGE,
        lists_models=False,
        note='FLUX from the people who trained it: the newest variant lands here first, and ultra is '
             'only offered at 4MP by BFL itself. Kontext is the editing model — give it a picture and '
             'say what to change.',
        fallback_models=('flux-pro-1.1-ultra', 'flux-kontext-max', 'flux-pro-1.1', 'flux-dev'),
    ),
    Host(
        id='stability',
        label='Stability AI',
        adapter='stability',
        base_url='https://api.stability.ai',
        env_key='STABILITY_API_KEY',
        modalities=IMAGE | VIDEO,
        lists_models=False,
        note='Stable Image for stills. Its video model is image-to-video only — there is no text '
             'path — and it takes one of exactly three input sizes.',
        fallback_models=('ultra', 'core', 'sd3'),
    ),
    Host(
        id='runway',
        label='Runway',
        adapter='runway',
        base_url='https://api.dev.runwayml.com/v1',
        env_key='RUNWAYML_API_SECRET',
        modalities=IMAGE | VIDEO,
        lists_models=False,
        note='Tagged reference images: name a person or a place and then refer to it as @sarah in '
             'this and every later prompt. No aggregator passes the tags through, which is the '
             'reason this is a direct connection.',
        fallback_models=('gen4_turbo', 'gen4_image', 'gen4_aleph', 'veo3'),
    ),
    Host(
        id='luma',
        label='Luma (Dream Machine)',
        adapter='luma',
        base_url='https://api.lumalabs.ai/dream-machine/v1',
        env_key='LUMAAI_API_KEY',
        modalities=IMAGE | VIDEO,
        lists_models=False,
        note='The best camera language of anything here, and keyframes — a first frame, a last '
             'frame, or both. Its reference images must be public URLs: Luma has no upload endpoint '
             'and takes no data URI, so those controls are URL boxes rather than file pickers.',
        fallback_models=('ray-2', 'ray-flash-2', 'photon-1'),
    ),
    Host(
        id='kling',
        label='Kling (Kuaishou)',
        adapter='kling',
        base_url='https://api-singapore.klingai.com',
        env_key='KLING_API_KEY',
        modalities=IMAGE | VIDEO,
        lists_models=False,
        note='The strongest of these at motion that obeys physics, and it takes a tail image — the '
             'frame to end on. The credential is two values: enter it as access_key:secret_key, '
             'because Kling signs a JWT rather than taking a bearer token.',
        fallback_models=('kling-v2-1-master', 'kling-v2-master', 'kling-v1-6'),
    ),
    Host(
        id='minimax',
        label='MiniMax (Hailuo)',
        adapter='minimax',
        base_url='https://api.minimax.io/v1',
        env_key='MINIMAX_API_KEY',
        modalities=IMAGE | VIDEO,
        lists_models=False,
        note='Strong prompt adherence, and the Director models execute shot instructions written '
             'into the prompt in square brackets — [Push in], [Pan left], [Tracking shot].',
        fallback_models=('MiniMax-Hailuo-02', 'T2V-01-Director', 'I2V-01-Director'),
    ),
    Host(
        id='ideogram',
        label='Ideogram',
        adapter='ideogram',
        base_url='https://api.ideogram.ai',
        env_key='IDEOGRAM_API_KEY',
        modalities=IMAGE,
        lists_models=False,
        note='The one to use when the picture has words in it. Every other model treats text as '
             'texture; this one treats it as text, which is the whole of why it is here.',
        fallback_models=('v3', 'v2', 'v2-turbo'),
    ),
    Host(
        id='recraft',
        label='Recraft',
        adapter='openai',
        base_url='https://external.api.recraft.ai/v1',
        env_key='RECRAFT_API_KEY',
        modalities=IMAGE,
        note='OpenAI-shaped on the images path. The one that outputs real vectors rather than a '
             'raster that looks like one, and holds a brand style across a set.',
    ),
    Host(
        id='comfyui',
        label='ComfyUI',
        adapter='comfyui',
        base_url='http://127.0.0.1:8188',
        modalities=IMAGE | VIDEO,
        local=True,
        note='Takes a workflow rather than a prompt, so what it can make is the set of templates installed — see openmirror/providers/workflows.',
    ),
)

HOSTS_BY_ID: dict[str, Host] = {h.id: h for h in HOSTS}


@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    """One embedding model, and what a person needs to know to choose it.

    `dimensions` is here because it is the one property that cannot be changed
    later: memory stored with a 3072-dimension model cannot be searched with a
    1024-dimension one, and the failure is silent unless something checks. The
    memory store records the model a vector was made with for exactly that
    reason.
    """

    id: str
    label: str
    dimensions: int
    # Where it can be got. Ids differ between hosts for the same weights, so
    # this maps host id → the id that host knows it by.
    served_by: dict[str, str] = field(default_factory=dict)
    # Some models accept a shorter output vector than they natively produce.
    # Worth surfacing: it is a real memory-size lever, and free.
    truncatable: bool = False
    max_input_tokens: int = 8192
    # Some models express the query/document asymmetry as an instruction in
    # the text rather than as a request field — Qwen3-Embedding and Gemini
    # Embedding 2 both do. The prefix goes on queries only; a document
    # embedded with it is embedded wrongly, which is why this is a property of
    # the model rather than something a caller remembers to pass.
    instruct_queries: bool = False
    #: The model openmirror's memory and recall are built and tested against. Not a
    #: quality ranking and not a default — several entries here are better. It
    #: marks the line below which a FEATURE stops being able to assume its
    #: retrieval is good enough, which is a different question from what an
    #: operator may choose. See FLOOR_CHAT below and docs/PROVIDERS.md.
    floor: bool = False
    note: str = ''


# ---------- The floor ----------
#
# What openmirror's features are built and tested against. Not what will start:
# nothing here refuses a smaller model, and `OPENMIRROR_MODEL` takes any id at all.
# What the floor governs is what a FEATURE may assume.
#
# Below it the agent loop does not get slower, it gets unreliable in ways that
# read as the harness being broken: prose where a tool call was needed, a plan
# the model wrote two turns ago and has already lost.
#
# **The tool budget is part of the floor, not a footnote to it.** gemma4:12b
# completed a five-step browser task in 26 seconds, first try, with the toolset
# narrowed to `browser` + `web` — about ten tools. The same model, same task,
# same prompt, given all thirty, opened the page and then reported it had no
# way to browse. So `TOOLSETS` and the `tools` list on POST /api/sessions are
# the mitigation this floor implies, and handing a floor model everything is a
# way of putting it below the floor without changing the model.
#
# Honest about the evidence: those measurements are gemma4:12b on this machine.
# qwen3.5:9b is named because the rest of this family tests against it, not
# because openmirror has measured it. And even narrowed, a 12B is not enough for
# `autopilot` — see the README.
#
# The other end is a first-class target and needs nothing from this file: a
# frontier model with thinking enabled gets the same tools through the same
# guards, and reasoning is streamed as its own channel rather than folded into
# the answer.
FLOOR_CHAT: tuple[str, ...] = ('qwen3.5:9b', 'gemma4:12b')

#: Roughly the tool count a floor model handles without losing track of what it
#: has. Advisory — nothing enforces it — but it is the number behind the advice
#: to narrow a session's toolsets on a small model.
FLOOR_TOOL_BUDGET = 12


EMBEDDING_MODELS: tuple[EmbeddingModel, ...] = (
    EmbeddingModel(
        id='text-embedding-3-large',
        label='OpenAI text-embedding-3-large',
        dimensions=3072,
        served_by={'openai': 'text-embedding-3-large'},
        truncatable=True,
        note='Takes a `dimensions` argument: ask for 1024 and the vectors are a third the size at close to the same recall.',
    ),
    EmbeddingModel(
        id='gemini-embedding-2',
        label='Gemini Embedding 2',
        dimensions=3072,
        served_by={'google': 'gemini-embedding-2'},
        truncatable=True,
        max_input_tokens=8192,
        instruct_queries=True,
        note='Truncatable to anything from 128 up; 768, 1536 and 3072 are the '
             'sizes Google documents as trained for. Unlike the 001 model it '
             'takes no taskType — the query/document asymmetry is expressed as '
             'an instruction in the text instead, which is why instruct_queries '
             'is set here and not there.',
    ),
    EmbeddingModel(
        id='gemini-embedding-001',
        label='Gemini Embedding 001',
        dimensions=3072,
        served_by={'google': 'gemini-embedding-001'},
        truncatable=True,
        max_input_tokens=2048,
        note='The previous generation, kept because it is the one that takes a '
             'taskType parameter rather than an instruction in the text.',
    ),
    EmbeddingModel(
        id='voyage-3-large',
        label='Voyage voyage-3-large',
        dimensions=1024,
        served_by={'voyage': 'voyage-3-large'},
        truncatable=True,
        max_input_tokens=32000,
        note='Wants an `input_type` of query or document, and the two are not interchangeable — the adapter sets it per call.',
    ),
    EmbeddingModel(
        id='qwen3-embedding-8b',
        label='Qwen3-Embedding-8B',
        dimensions=4096,
        served_by={'ollama': 'qwen3-embedding:8b', 'vllm': 'Qwen/Qwen3-Embedding-8B', 'llamacpp': 'Qwen/Qwen3-Embedding-8B', 'together': 'Qwen/Qwen3-Embedding-8B', 'deepinfra': 'Qwen/Qwen3-Embedding-8B'},
        truncatable=True,
        max_input_tokens=32000,
        instruct_queries=True,
        note='Open weights, so it runs on your own hardware — the only entry here that can be both the best available and never leave the room.',
    ),
    EmbeddingModel(
        id='qwen3-embedding-4b',
        label='Qwen3-Embedding-4B',
        dimensions=2560,
        served_by={'ollama': 'qwen3-embedding:4b', 'vllm': 'Qwen/Qwen3-Embedding-4B', 'llamacpp': 'Qwen/Qwen3-Embedding-4B', 'together': 'Qwen/Qwen3-Embedding-4B'},
        truncatable=True,
        max_input_tokens=32000,
        instruct_queries=True,
        floor=True,
        note='The floor: what memory and recall are built and tested against. '
             'Half the memory of the 8B for most of the quality, and the one to '
             'start with on a single consumer GPU.',
    ),
    EmbeddingModel(
        id='text-embedding-3-small',
        label='OpenAI text-embedding-3-small',
        dimensions=1536,
        served_by={'openai': 'text-embedding-3-small'},
        truncatable=True,
    ),
    EmbeddingModel(
        id='nomic-embed-text',
        label='Nomic Embed Text',
        dimensions=768,
        served_by={'ollama': 'nomic-embed-text', 'perch:chat': 'nomic-embed-text'},
    ),
)

EMBEDDING_BY_ID: dict[str, EmbeddingModel] = {m.id: m for m in EMBEDDING_MODELS}


def embedding_models_for(host_id: str) -> list[EmbeddingModel]:
    """The catalogued models a given connection is known to serve.

    A hint for the picker, never a filter: the live list from the server is
    what is actually offered, and this only adds the dimensions and the notes
    to the ones it recognises.
    """
    return [m for m in EMBEDDING_MODELS if host_id in m.served_by]


# What a query is prefixed with for a model that wants its task in the text.
# One string rather than one per model, deliberately: two spellings of the same
# instruction put queries embedded by different models in subtly different
# places, and a store cannot tell afterwards which spelling made which vector.
QUERY_INSTRUCTION = 'Instruct: Given a search query, retrieve passages that answer it\nQuery: '


def query_prefix(model_id: str) -> str:
    """What to put in front of a query for this model, or an empty string.

    Applied to queries only. The asymmetry is the whole point: the passage goes
    in bare and the question goes in instructed, and doing it to both is the
    same as doing it to neither.
    """
    known = describe_embedding(model_id)
    return QUERY_INSTRUCTION if known and known.instruct_queries else ''


def describe_embedding(model_id: str) -> EmbeddingModel | None:
    """What is known about a model id, from either the catalogue id or a host's."""
    if model_id in EMBEDDING_BY_ID:
        return EMBEDDING_BY_ID[model_id]
    for model in EMBEDDING_MODELS:
        if model_id in model.served_by.values():
            return model
    return None
