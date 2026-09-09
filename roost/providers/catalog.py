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

from roost.providers.base import Modality

#: Which adapter class answers for a host. A shape, not a company: `openai`
#: covers every service that copied that request format, which is most of them.
Adapter = Literal['openai', 'anthropic', 'ollama', 'a1111', 'comfyui', 'voyage', 'google']


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
    Host(
        id='openai',
        label='OpenAI',
        adapter='openai',
        base_url='https://api.openai.com/v1',
        env_key='OPENAI_API_KEY',
        modalities=CHAT_AND_EMBED | STT | TTS | IMAGE,
        note='Chat, embeddings, speech both ways, and images.',
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
        modalities=CHAT_AND_EMBED,
        note='Model ids are paths — accounts/fireworks/models/…',
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
        modalities=CHAT_AND_EMBED,
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
        modalities=CHAT,
    ),
    Host(
        id='google',
        label='Google AI',
        adapter='google',
        base_url='https://generativelanguage.googleapis.com/v1beta',
        env_key='GEMINI_API_KEY',
        modalities=CHAT_AND_EMBED,
        note='Gemini, and the Gemini embedding models.',
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
    Host(
        id='comfyui',
        label='ComfyUI',
        adapter='comfyui',
        base_url='http://127.0.0.1:8188',
        modalities=IMAGE | VIDEO,
        local=True,
        note='Takes a workflow rather than a prompt, so what it can make is the set of templates installed — see roost/providers/workflows.',
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
    #: The model Roost's memory and recall are built and tested against. Not a
    #: quality ranking and not a default — several entries here are better. It
    #: marks the line below which a FEATURE stops being able to assume its
    #: retrieval is good enough, which is a different question from what an
    #: operator may choose. See FLOOR_CHAT below and docs/PROVIDERS.md.
    floor: bool = False
    note: str = ''


# ---------- The floor ----------
#
# What Roost's features are built and tested against. Not what will start:
# nothing here refuses a smaller model, and `ROOST_MODEL` takes any id at all.
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
# because Roost has measured it. And even narrowed, a 12B is not enough for
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
