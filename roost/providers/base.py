"""What a provider is, and the six things one can be asked to do.

Every provider implements only the modalities it has. Nothing here knows
about Anthropic, OpenAI or Ollama; the registry does the choosing, so adding
a provider is one file and one line, and swapping the speech synthesiser
cannot disturb the chat model.

The message format is deliberately closer to Anthropic's than OpenAI's:
content is a list of typed blocks rather than a string, because tool results,
images and reasoning do not fit in a string and every adapter that pretends
otherwise ends up re-parsing its own output. Flattening blocks into an
OpenAI payload is easy; recovering blocks from a flattened one is not.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class Modality(StrEnum):
    CHAT = 'chat'
    EMBEDDING = 'embedding'
    STT = 'stt'
    TTS = 'tts'
    IMAGE = 'image'
    VIDEO = 'video'


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TextBlock:
    text: str
    type: Literal['text'] = 'text'


@dataclass(slots=True)
class ImageBlock:
    """Base64 image data. URLs are resolved before they reach a provider, so
    that one provider being unable to fetch a URL is not a capability
    difference the caller has to know about."""

    data: str
    media_type: str = 'image/png'
    type: Literal['image'] = 'image'


@dataclass(slots=True)
class ThinkingBlock:
    text: str
    signature: str | None = None
    type: Literal['thinking'] = 'thinking'


@dataclass(slots=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: Literal['tool_use'] = 'tool_use'


@dataclass(slots=True)
class ToolResultBlock:
    tool_use_id: str
    content: str
    is_error: bool = False
    type: Literal['tool_result'] = 'tool_result'


ContentBlock = TextBlock | ImageBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock


@dataclass(slots=True)
class Message:
    role: Literal['user', 'assistant']
    content: list[ContentBlock]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(slots=True)
class ChatRequest:
    model: str
    messages: list[Message]
    system: str | None = None
    tools: list[ToolSpec] = field(default_factory=list)
    max_tokens: int = 4096
    temperature: float | None = None
    stop: list[str] = field(default_factory=list)
    # Provider-specific extras, passed through untouched. An escape hatch, and
    # deliberately not a place to put anything the registry needs to read.
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StreamText:
    text: str
    type: Literal['text'] = 'text'


@dataclass(slots=True)
class StreamThinking:
    text: str
    type: Literal['thinking'] = 'thinking'


@dataclass(slots=True)
class StreamToolUse:
    """A complete tool call.

    Providers stream tool arguments as partial JSON, which is useless to a
    caller until it parses — so adapters accumulate and emit once. The cost is
    that a tool call appears all at once; the alternative is every consumer
    writing its own incremental JSON parser.
    """

    id: str
    name: str
    input: dict[str, Any]
    type: Literal['tool_use'] = 'tool_use'


@dataclass(slots=True)
class StreamDone:
    stop_reason: str = 'end_turn'
    usage: dict[str, int] = field(default_factory=dict)
    type: Literal['done'] = 'done'


StreamEvent = StreamText | StreamThinking | StreamToolUse | StreamDone


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class ChatProvider(abc.ABC):
    @abc.abstractmethod
    def stream(self, req: ChatRequest) -> AsyncIterator[StreamEvent]: ...

    @abc.abstractmethod
    async def models(self) -> list[dict[str, Any]]: ...


class EmbeddingProvider(abc.ABC):
    @abc.abstractmethod
    async def embed(self, texts: list[str], model: str) -> list[list[float]]: ...


@dataclass(slots=True)
class Transcript:
    text: str
    # True while the person is still talking. Streaming backends emit several
    # partials and one final; batch backends emit one final and never a partial.
    partial: bool = False
    language: str | None = None


class STTProvider(abc.ABC):
    @abc.abstractmethod
    async def transcribe(self, audio: bytes, *, model: str, language: str | None = None) -> Transcript: ...

    def supports_streaming(self) -> bool:
        """Whether `stream_transcribe` is real.

        Most local backends are batch-only — whisper.cpp behind Perch included —
        and the gateway compensates by cutting on VAD and transcribing each
        utterance. Saying so here lets it choose without a try/except.
        """
        return False

    def stream_transcribe(
        self, chunks: AsyncIterator[bytes], *, model: str, language: str | None = None
    ) -> AsyncIterator[Transcript]:
        raise NotImplementedError


class TTSProvider(abc.ABC):
    # The rate of the audio this provider actually returns, which is not
    # necessarily the rate anyone asked for. OpenAI's `pcm` is 24 kHz and so is
    # Kokoro's; a client that assumes 16 kHz plays them about a third too slow,
    # which sounds like a fault in the model rather than in the plumbing. So it
    # is declared rather than guessed.
    output_sample_rate: int = 24_000

    @abc.abstractmethod
    def synthesize(
        self, text: str, *, model: str, voice: str, fmt: str = 'pcm16', sample_rate: int = 16_000
    ) -> AsyncIterator[bytes]:
        """Yield audio as it is produced.

        An implementation that can only return one buffer yields once. It must
        still be an iterator, because the caller cancels by closing it — that
        is how barge-in stops a synthesis already in flight.
        """

    @abc.abstractmethod
    async def voices(self) -> list[dict[str, Any]]: ...


@dataclass(slots=True)
class GeneratedMedia:
    data: bytes
    media_type: str
    seed: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class ImageProvider(abc.ABC):
    @abc.abstractmethod
    async def generate(
        self, prompt: str, *, model: str, n: int = 1, size: str = '1024x1024', **kw: Any
    ) -> list[GeneratedMedia]: ...


class VideoProvider(abc.ABC):
    @abc.abstractmethod
    async def generate(self, prompt: str, *, model: str, **kw: Any) -> list[GeneratedMedia]: ...


@dataclass(slots=True)
class ProviderInfo:
    """What a provider is and what it can do, for the registry and the UI."""

    id: str
    label: str
    modalities: set[Modality]
    # True when the model runs on hardware the operator controls. The UI marks
    # these, because "which of these choices sends my screen to a company" is
    # the question people actually want answered.
    local: bool = False
    base_url: str | None = None
