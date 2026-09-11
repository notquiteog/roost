"""Leaked chat-template markers.

Observed for real: gemma4:12b through Ollama emitted `<channel|>` at the head
of an answer during a multi-step tool turn. It reaches the transcript, and
because assistant text is what the voice pipeline speaks, it also reaches the
speakers.
"""

from __future__ import annotations

import pytest

from openmirror.providers.control_tokens import strip_control_tokens


@pytest.mark.parametrize(
    'raw,expected',
    [
        ('<channel|>I have updated the file.', 'I have updated the file.'),
        ('<|channel|>final<|message|>Done.', 'finalDone.'),
        ('<|im_start|>Hello<|im_end|>', 'Hello'),
        ('<|eot_id|>', ''),
        ('<|reserved_special_token_12|>x', 'x'),
    ],
)
def test_markers_are_removed(raw, expected):
    assert strip_control_tokens(raw) == expected


@pytest.mark.parametrize(
    'text',
    [
        'Use the <div> element here.',
        'if (a < b) { return a > c; }',
        'The channel is busy.',
        'Vec<String> is the type.',
        'a < b and c > d',
        '',
        'ordinary prose with no angle brackets at all',
    ],
)
def test_real_content_is_left_alone(text):
    """The filter is narrow on purpose: anything broader eats someone's code."""
    assert strip_control_tokens(text) == text


def test_whitespace_is_preserved():
    """This runs on streaming deltas, where a lone space matters to the
    sentence being assembled."""
    assert strip_control_tokens(' ') == ' '
    assert strip_control_tokens('word ') == 'word '
    assert strip_control_tokens('<|end|> tail') == ' tail'


def test_html_in_a_code_block_survives():
    code = '<html>\n  <body>\n    <p>hi</p>\n  </body>\n</html>'
    assert strip_control_tokens(code) == code


@pytest.mark.parametrize(
    'text',
    [
        '<user>alice</user>',
        '<system>config</system>',
        '<message>hi</message>',
        'the <start> and <end> of the range',
        'an <assistant> tag in some XML',
    ],
)
def test_pipeless_tags_are_content_not_markers(text):
    """A pipe is what distinguishes a special token from an ordinary tag.

    Without requiring one, this filter would quietly rewrite anyone's XML —
    and conversations about prompt formats are exactly where these words show
    up as plain text.
    """
    assert strip_control_tokens(text) == text
