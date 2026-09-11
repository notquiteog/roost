"""Stripping chat-template markers that leak into content.

Local models occasionally emit their own control tokens as ordinary text —
`<|channel|>`, `<|start|>` and friends — when the template that was supposed
to consume them does not. It is intermittent and depends on the model, the
template and how long the generation ran, so it cannot be fixed by asking the
model nicely.

It matters more than a cosmetic blemish because assistant text is also what
the voice pipeline speaks aloud, and "less than pipe channel pipe greater
than" is not a thing anyone should hear their computer say.

The list is deliberately narrow: these are special-token *spellings*, with
the pipe-and-angle-bracket shape a tokenizer uses and prose does not. Nothing
here is a general-purpose filter, because stripping anything broader would
eventually eat somebody's code.
"""

from __future__ import annotations

import re

_NAMES = (
    r'channel|message|start|end|return|constrain|assistant|user|system|'
    r'im_start|im_end|eot_id|start_header_id|end_header_id|reserved_special_token_\d+'
)

# At least one pipe is required, and that requirement is the whole safety
# margin. Making both pipes optional also matches `<user>` and `<system>` —
# ordinary tags that turn up in XML, in templating, and in any conversation
# about prompts. A marker with no pipe at all is indistinguishable from
# content, so it is left alone.
CONTROL_TOKEN = re.compile(
    rf'(?:<\|(?:{_NAMES})\|>|<\|(?:{_NAMES})>|<(?:{_NAMES})\|>)'
)


def strip_control_tokens(text: str) -> str:
    """Remove leaked template markers from model output.

    Whitespace is deliberately not tidied afterwards: this runs on streaming
    deltas, where a chunk may be a single space that matters to the sentence
    being assembled.
    """
    if '<' not in text:
        # The overwhelmingly common case, and this runs on every delta.
        return text
    return CONTROL_TOKEN.sub('', text)
